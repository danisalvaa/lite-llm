"""Tests for the V2 ``OpenTelemetryV2`` CustomLogger adapter.

Exercises the callback surface the existing call sites use: the LLM-call span
opened at the ``pre_call`` boundary and closed at async success/failure, service
hooks, proxy SERVER span lifecycle (start + setters), parent-context resolution
(ambient context), and Baggage promotion onto child spans.
"""

import asyncio
import contextlib
from datetime import datetime, timezone

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind  # noqa: E402
from opentelemetry.trace.status import StatusCode  # noqa: E402

from litellm.integrations.otel import (  # noqa: E402
    GenAI,
    HTTP,
    LiteLLM,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel import providers  # noqa: E402
from litellm.integrations.otel.logger import (  # noqa: E402
    LITELLM_PROXY_REQUEST_SPAN_NAME,
    OpenTelemetryV2,
)
from litellm.integrations.otel.spans import SpanRole  # noqa: E402
from litellm.integrations.otel.utils import to_ns, to_seconds  # noqa: E402

# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #


def _payload(**overrides):
    payload = {
        "call_type": "acompletion",
        "custom_llm_provider": "openai",
        "model": "gpt-4o",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "stream": False,
        "model_parameters": {"temperature": 0.7, "max_tokens": 256},
        "response": {
            "id": "resp_1",
            "model": "gpt-4o-2024",
            "choices": [{"finish_reason": "stop"}],
        },
        "metadata": {
            "team_id": "t1",
            "team_alias": "team one",
            "user_api_key_hash": "hsh",
        },
        "api_base": "https://api.openai.com:443/v1",
        "status": "success",
        "litellm_call_id": "call_1",
        "response_cost": 0.002,
        "hidden_params": {},
    }
    payload.update(overrides)
    return payload


def _kwargs(payload=None):
    return {
        # ``litellm_call_id`` (here carried inside the payload) correlates the
        # pre_call boundary with the close callback — the carrier is keyed by it.
        "standard_logging_object": payload if payload is not None else _payload(),
        "litellm_params": {"metadata": {}},
    }


def _logger(legacy_compat=True):
    cfg = OpenTelemetryV2Config(exporter="in_memory", legacy_compat=legacy_compat)
    exporter = InMemorySpanExporter()
    tracer_provider = providers.build_tracer_provider(cfg, exporter=exporter)
    return OpenTelemetryV2(config=cfg, tracer_provider=tracer_provider), exporter


def _emit_llm(logger, kwargs=None, *, ambient=None, fail=False):
    """Drive the real boundary flow: open at ``pre_call`` then close at the async
    callback. ``ambient``, if given, is the span that is the active OTel context
    while ``pre_call`` runs (the server span) so the LLM span parents to it."""
    if kwargs is None:
        kwargs = _kwargs()
    payload = kwargs.get("standard_logging_object") or {}
    with (
        trace.use_span(ambient, end_on_exit=False)
        if ambient is not None
        else contextlib.nullcontext()
    ):
        logger.log_pre_api_call(model=payload.get("model"), messages=[], kwargs=kwargs)
    hook = logger.async_log_failure_event if fail else logger.async_log_success_event
    asyncio.run(hook(kwargs, None, None, None))
    return kwargs


# --------------------------------------------------------------------------- #
#  Time helpers
# --------------------------------------------------------------------------- #


def test_to_ns_handles_datetime_and_float():
    dt = datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc)
    assert to_ns(dt) == int(dt.timestamp() * 1e9)
    assert to_ns(1.5) == 1_500_000_000
    assert to_ns(None) is None
    assert to_ns(True) is None  # bool is rejected — not a real epoch value


def test_to_seconds_parses_string_formats():
    assert to_seconds("2026-05-26 12:00:00.123") is not None
    assert to_seconds("2026-05-26 12:00:00") is not None
    assert to_seconds("nonsense") is None
    assert to_seconds(None) is None
    assert to_seconds(1.5) == 1.5


# --------------------------------------------------------------------------- #
#  LLM-call callbacks
# --------------------------------------------------------------------------- #


def test_async_log_success_event_emits_llm_call_span():
    logger, exporter = _logger()
    _emit_llm(logger)
    (span,) = exporter.get_finished_spans()
    assert span.name == "chat gpt-4o"
    assert span.kind is SpanKind.CLIENT
    assert span.attributes[GenAI.OPERATION_NAME] == "chat"
    assert span.attributes[GenAI.REQUEST_MODEL] == "gpt-4o"
    assert span.attributes[LiteLLM.CALL_ID] == "call_1"
    # Success leaves status UNSET (semconv default), not forced OK.
    assert span.status.status_code is StatusCode.UNSET


def test_async_log_failure_event_marks_error_status():
    logger, exporter = _logger()
    payload = _payload(
        status="failure",
        error_information={"error_class": "RateLimitError", "error_message": "429"},
    )
    _emit_llm(logger, _kwargs(payload=payload), fail=True)
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "RateLimitError"


def test_sync_log_event_is_noop():
    """V2 closes the span async-only; the sync callback runs out-of-context, so
    it no-ops (the span stays open on the carrier until the async callback)."""
    logger, exporter = _logger()
    kwargs = _kwargs()
    logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
    logger.log_success_event(kwargs, None, None, None)
    logger.log_failure_event(kwargs, None, None, None)
    assert exporter.get_finished_spans() == ()


def test_missing_standard_logging_object_is_noop():
    """No carrier (``pre_call`` never ran) → the callback emits nothing."""
    logger, exporter = _logger()
    asyncio.run(
        logger.async_log_success_event({"litellm_params": {}}, None, None, None)
    )
    assert exporter.get_finished_spans() == ()


def test_no_span_when_pre_call_never_ran():
    """A request rejected before the upstream call — at the auth/budget gate, or
    blocked by a pre-call guardrail — never reaches ``pre_call``, so there is no
    carrier and the failure log produces no phantom CLIENT span. This replaces the
    old post-hoc heuristics: "did pre_call run?" is the only signal needed."""
    logger, exporter = _logger()
    payload = _payload(
        status="failure",
        error_information={"error_class": "ProxyException", "error_code": "401"},
    )
    # No log_pre_api_call: the call never started.
    asyncio.run(
        logger.async_log_failure_event(_kwargs(payload=payload), None, None, None)
    )
    assert exporter.get_finished_spans() == ()  # no phantom LLM span


def test_real_llm_failure_still_emitted():
    """A genuine LLM failure: ``pre_call`` ran (the call was attempted), so the
    CLIENT span is opened at the boundary and closed ERROR."""
    logger, exporter = _logger()
    payload = _payload(
        status="failure",
        error_information={"error_class": "RateLimitError", "error_code": "429"},
    )
    _emit_llm(logger, _kwargs(payload=payload), fail=True)
    (span,) = exporter.get_finished_spans()
    assert span.name == "chat gpt-4o"
    assert span.status.status_code is StatusCode.ERROR


def test_idempotent_on_repeat_callback():
    """The carrier is the dedup: once the async callback closes the span and
    clears the carrier, a second callback firing emits nothing."""
    logger, exporter = _logger()
    kwargs = _kwargs()
    logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    assert len(exporter.get_finished_spans()) == 1


def test_pre_call_idempotent_keeps_first_span():
    """A retried call may re-enter ``pre_call`` with the same call id; the first
    span (with the true start time) is kept, not replaced."""
    logger, _ = _logger()
    kwargs = _kwargs()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    with trace.use_span(server, end_on_exit=False):
        logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
        first = logger._open_llm_calls["call_1"]
        logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
        second = logger._open_llm_calls["call_1"]
    server.end()
    assert first is second  # not overwritten


# --------------------------------------------------------------------------- #
#  Parent resolution — ambient context at the boundary (no metadata threading)
# --------------------------------------------------------------------------- #


def test_llm_span_parents_to_ambient_server_span():
    """The span is opened at ``pre_call`` while the server span is the active
    context, so it nests under it natively (no ``litellm_parent_otel_span``)."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    _emit_llm(logger, ambient=server)
    server.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    llm_span = by_name["chat gpt-4o"]
    assert llm_span.parent is not None
    assert llm_span.parent.span_id == server.get_span_context().span_id


def test_llm_span_is_root_without_ambient_server_span():
    """No server span at ``pre_call`` → creation is deferred and the span is a
    root of its own trace (the SDK / no-proxy path)."""
    logger, exporter = _logger()
    _emit_llm(logger)
    (span,) = exporter.get_finished_spans()
    assert span.parent is None  # standalone (no proxy server span) → root


def test_real_logging_pre_call_opens_span_end_to_end():
    """Regression guard: a real ``LiteLLMLoggingObj.pre_call`` must fire
    ``log_pre_api_call`` on the V2 logger (via ``litellm.input_callback``), so the
    boundary span is opened and then closed by the success callback. If the logger
    is not wired into ``input_callback``, no span is produced at all."""
    import litellm
    from litellm.litellm_core_utils.litellm_logging import Logging

    logger, exporter = _logger()
    # Register exactly this logger as the (only) input callback pre_call iterates.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(litellm, "input_callback", [logger], raising=False)
    try:
        logging_obj = Logging(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            call_type="acompletion",
            start_time=datetime.now(),
            litellm_call_id="call_e2e",
            function_id="fn",
        )
        # The wrapper always runs this before pre_call — it's what seeds
        # ``litellm_params`` and ``litellm_call_id`` into ``model_call_details``
        # (the call id is how the close callback correlates back to this span).
        logging_obj.update_environment_variables(
            litellm_params={"metadata": {}},
            optional_params={},
            model="gpt-4o",
        )
        # pre_call fires log_pre_api_call → opens the boundary span on the obj.
        logging_obj.pre_call(input="hi", api_key="sk-test")
        # The success callback closes it, reading the typed payload.
        logging_obj.model_call_details["standard_logging_object"] = _payload(
            litellm_call_id="call_e2e"
        )
        asyncio.run(
            logger.async_log_success_event(
                logging_obj.model_call_details, None, None, None
            )
        )
    finally:
        monkeypatch.undo()
    (span,) = exporter.get_finished_spans()
    assert span.name == "chat gpt-4o"


def test_deferred_span_parents_to_ambient_at_close():
    """When ``pre_call`` runs off the request task (a sync-only provider driven
    through a thread pool, where contextvars don't follow), no ambient parent is
    visible there, so span creation is deferred. The async callback — whose worker
    context was copied from the request task and so still carries the server span —
    then creates it parented to that server span, not as an orphan root."""
    logger, exporter = _logger()
    kwargs = _kwargs()
    # pre_call with NO ambient span (the thread-pool case) → deferred.
    logger.log_pre_api_call(model="gpt-4o", messages=[], kwargs=kwargs)
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    # The close callback runs with the (worker-copied) server span ambient.
    with trace.use_span(server, end_on_exit=False):
        asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    server.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    llm_span = by_name["chat gpt-4o"]
    assert llm_span.parent.span_id == server.get_span_context().span_id


# Inbound ``traceparent`` propagation is now the FastAPI instrumentor's job
# (see proxy_server's startup mount + ``test_otel_v2_mount``), not the logger's.


# --------------------------------------------------------------------------- #
#  Baggage promotion (LLM call writes identity into baggage so child spans
#  inherit team/key/model attrs).
# --------------------------------------------------------------------------- #


def test_baggage_identity_promoted_onto_llm_call():
    """On the deferred (SDK / no-proxy) path the callback seeds identity Baggage
    from the payload so the span is still labeled with team/key. (On the proxy
    boundary path identity rides in from auth-seeded ambient Baggage instead.)"""
    logger, exporter = _logger()
    _emit_llm(logger)
    (span,) = exporter.get_finished_spans()
    assert span.attributes[LiteLLM.TEAM_ID] == "t1"
    assert span.attributes[LiteLLM.TEAM_ALIAS] == "team one"
    assert span.attributes[GenAI.REQUEST_MODEL] == "gpt-4o"


class _Auth:
    """Stub matching the ``UserAPIKeyAuth`` fields the logger reads."""

    team_id = "t1"
    team_alias = "team one"
    api_key = "hash1"
    user_id = "u1"
    org_id = None
    key_alias = "k1"
    end_user_id = None


def test_pre_call_hook_seeds_baggage_onto_server_and_child_spans():
    """The pre-call hook seeds identity Baggage in the request context so the
    server span (stamped directly) AND later child spans (service here, via the
    Baggage processor) carry identity — not just the LLM-call span."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )

    async def _flow():
        # pre-call seeds baggage + stamps the active server span
        await logger.async_pre_call_hook(
            _Auth(), None, {"model": "gpt-4o"}, "completion"
        )
        # a later service call (same task) must inherit the identity
        await logger.async_service_success_hook(
            payload=_ServicePayload("redis", "set"), parent_otel_span=server
        )

    with trace.use_span(server, end_on_exit=False):
        asyncio.run(_flow())
    server.end()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    redis = spans["redis set"]
    assert redis.attributes[LiteLLM.TEAM_ID] == "t1"
    assert redis.attributes[LiteLLM.KEY_HASH] == "hash1"
    assert redis.attributes[f"{LiteLLM.METADATA_PREFIX}user_api_key_user_id"] == "u1"
    srv = spans[LITELLM_PROXY_REQUEST_SPAN_NAME]
    assert (
        srv.attributes[LiteLLM.TEAM_ID] == "t1"
    )  # stamped directly on the server span
    assert srv.attributes[f"{LiteLLM.METADATA_PREFIX}user_api_key_user_id"] == "u1"


# --------------------------------------------------------------------------- #
#  Service hooks (Phase 3)
# --------------------------------------------------------------------------- #


class _Service:
    """Stub matching ``ServiceTypes(str, Enum)``."""

    def __init__(self, value):
        self.value = value


class _ServicePayload:
    def __init__(self, service="redis", call_type="set", error=None):
        self.service = _Service(service)
        self.call_type = call_type
        self.error = error


def _service_parent(logger):
    """Helper: a live PROXY_REQUEST span to parent service spans under."""
    return logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )


def test_async_service_success_hook_emits_service_span():
    logger, exporter = _logger()
    parent = _service_parent(logger)
    try:
        asyncio.run(
            logger.async_service_success_hook(
                payload=_ServicePayload("redis", "set"),
                parent_otel_span=parent,
                event_metadata={"key1": "val1"},
            )
        )
    finally:
        parent.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    # Name disambiguates calls to the same service; redis is an outbound
    # datastore call, so it's a CLIENT span with db.* semconv.
    span = by_name["redis set"]
    assert span.kind is SpanKind.CLIENT
    assert span.attributes["db.system.name"] == "redis"
    assert span.attributes["db.operation.name"] == "set"
    assert span.attributes[LiteLLM.SERVICE_NAME] == "redis"
    assert span.attributes[LiteLLM.SERVICE_CALL_TYPE] == "set"
    # Canonical (V2) namespaced metadata key
    assert span.attributes[f"{LiteLLM.METADATA_PREFIX}key1"] == "val1"
    # V1 bare key (legacy dual-emit)
    assert span.attributes["key1"] == "val1"
    assert span.attributes["service"] == "redis"  # V1 bare key
    assert span.attributes["call_type"] == "set"  # V1 bare key
    # Success leaves status UNSET (semconv default), not forced OK.
    assert span.status.status_code is StatusCode.UNSET


def test_async_service_failure_hook_marks_error_status():
    logger, exporter = _logger()
    parent = _service_parent(logger)
    try:
        asyncio.run(
            logger.async_service_failure_hook(
                payload=_ServicePayload("postgres", "query"),
                error="boom",
                parent_otel_span=parent,
            )
        )
    finally:
        parent.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    span = by_name["postgres query"]
    assert span.kind is SpanKind.CLIENT
    assert span.attributes["db.system.name"] == "postgresql"
    assert span.status.status_code is StatusCode.ERROR
    # Without an explicit error_type from the payload, V2 stamps the fallback.
    assert span.attributes["error.type"] == "error"
    assert span.attributes[LiteLLM.SERVICE_NAME] == "postgres"


def test_async_service_failure_hook_preserves_payload_error_over_override():
    """When the payload itself carries an error, that takes precedence over the override."""
    logger, exporter = _logger()
    parent = _service_parent(logger)
    try:
        asyncio.run(
            logger.async_service_failure_hook(
                payload=_ServicePayload("postgres", "query", error="db-down"),
                error="override-only-used-when-payload-clean",
                parent_otel_span=parent,
            )
        )
    finally:
        parent.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    span = by_name["postgres query"]
    assert span.status.status_code is StatusCode.ERROR
    assert "db-down" in (span.status.description or "")


def test_metrics_only_ping_without_timing_or_parent_is_noop():
    """A success with no timing and no parent is a prometheus-only ping (the
    per-request ``self`` latency hook, in-memory queue gauges) — not a traceable
    operation, so no span is emitted."""
    logger, exporter = _logger()
    asyncio.run(
        logger.async_service_success_hook(
            payload=_ServicePayload(), parent_otel_span=None
        )
    )
    assert exporter.get_finished_spans() == ()


def test_background_service_call_with_timing_emits_root_span():
    """A background datastore call (no request → no parent) but with real timing
    still emits — as its own root trace — instead of being dropped."""
    logger, exporter = _logger()
    asyncio.run(
        logger.async_service_success_hook(
            payload=_ServicePayload("postgres", "query"),
            parent_otel_span=None,
            start_time=1.0,
            end_time=2.0,
        )
    )
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["postgres query"]
    # No parent → it's a root span of its own trace.
    assert spans[0].parent is None
    assert spans[0].kind is SpanKind.CLIENT


def test_internal_service_call_is_internal_kind_without_db_attrs():
    """A genuine internal service (background job) is an INTERNAL span, no db.*."""
    logger, exporter = _logger()
    asyncio.run(
        logger.async_service_success_hook(
            payload=_ServicePayload("reset_budget_job", "reset_budget"),
            parent_otel_span=None,
            start_time=1.0,
            end_time=2.0,
        )
    )
    span = exporter.get_finished_spans()[0]
    assert span.name == "reset_budget_job reset_budget"
    assert span.kind is SpanKind.INTERNAL
    assert "db.system.name" not in span.attributes
    assert span.attributes[LiteLLM.SERVICE_NAME] == "reset_budget_job"


def test_metrics_only_services_emit_no_span():
    """self / router / proxy_pre_call / auth duplicate gen-AI spans or get a live
    phase span — they are metrics-only and must not produce a service span."""
    for service in ("self", "router", "proxy_pre_call", "auth"):
        logger, exporter = _logger()
        asyncio.run(
            logger.async_service_success_hook(
                payload=_ServicePayload(service, "x"),
                parent_otel_span=None,
                start_time=1.0,
                end_time=2.0,
            )
        )
        assert exporter.get_finished_spans() == (), f"{service} should emit no span"


def test_service_span_inherits_parent_when_provided():
    logger, exporter = _logger()
    parent = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    try:
        asyncio.run(
            logger.async_service_success_hook(
                payload=_ServicePayload(), parent_otel_span=parent
            )
        )
    finally:
        parent.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    assert (
        by_name["redis set"].parent.span_id
        == by_name[LITELLM_PROXY_REQUEST_SPAN_NAME].get_span_context().span_id
    )


def test_service_span_prefers_ambient_context_over_threaded_parent():
    """Service/DB spans parent to the active (ambient) span when there is one, so
    they nest under whatever phase is active (e.g. a DB lookup under the live
    ``auth`` span). The threaded ``parent_otel_span`` is only a fallback for when
    ambient has no live span (a background service call)."""
    logger, exporter = _logger()
    ambient = logger._emitter.start_span(SpanRole.LLM_CALL, "chat gpt-4o")
    threaded = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    try:
        with trace.use_span(ambient, end_on_exit=False):
            asyncio.run(
                logger.async_service_success_hook(
                    payload=_ServicePayload("redis", "get"),
                    parent_otel_span=threaded,
                )
            )
    finally:
        ambient.end()
        threaded.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    assert by_name["redis get"].parent.span_id == ambient.get_span_context().span_id


# --------------------------------------------------------------------------- #
#  Proxy SERVER span lifecycle
# --------------------------------------------------------------------------- #


def test_create_proxy_request_started_span_returns_ambient_span():
    """V2 doesn't create a server span (the instrumentor does), but it returns
    the active server span so the proxy can thread it as the service-span parent
    — service logging only fires the OTel hook when that parent is non-None."""
    logger, exporter = _logger()
    # No ambient recordable span → None (and creates nothing).
    assert (
        logger.create_litellm_proxy_request_started_span(
            start_time=datetime.now(timezone.utc), headers={"traceparent": "x"}
        )
        is None
    )
    assert exporter.get_finished_spans() == ()
    # With an active server span, return it (do NOT create a new one).
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    with trace.use_span(server, end_on_exit=False):
        got = logger.create_litellm_proxy_request_started_span(
            start_time=datetime.now(timezone.utc), headers=None
        )
    server.end()
    assert got is server


def test_proxy_span_setters_are_noops():
    """The FastAPI instrumentor owns the server span; the setters write nothing
    (and must tolerate any span / None without raising).
    """
    logger, exporter = _logger()
    span = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    OpenTelemetryV2.set_proxy_request_route_attributes(
        span, url_path="/chat/completions", http_route="/chat/completions"
    )
    OpenTelemetryV2.set_response_status_code_attribute(span, 200)
    OpenTelemetryV2.set_preprocessing_duration_attribute(
        span,
        {"first_api_call_start_time": 1.0, "metadata": {"litellm_received_at": 0.5}},
    )
    span.end()
    (finished,) = exporter.get_finished_spans()
    assert HTTP.URL_PATH not in finished.attributes
    assert HTTP.ROUTE not in finished.attributes
    assert HTTP.RESPONSE_STATUS_CODE not in finished.attributes
    assert LiteLLM.PREPROCESSING_MS not in finished.attributes
    # None span is tolerated too.
    OpenTelemetryV2.set_proxy_request_route_attributes(None, http_route="/x")
    OpenTelemetryV2.set_response_status_code_attribute(None, 200)
    OpenTelemetryV2.set_preprocessing_duration_attribute(None, {})


# --------------------------------------------------------------------------- #
#  Constructor / proxy global guard
# --------------------------------------------------------------------------- #


def test_constructor_accepts_v1_compatible_kwargs():
    """Mirrors V1's positional shape — config / callback_name / providers / **kwargs."""
    cfg = OpenTelemetryV2Config(exporter="in_memory")
    tp = providers.build_tracer_provider(cfg)
    logger = OpenTelemetryV2(
        config=cfg,
        callback_name="otel",
        tracer_provider=tp,
        logger_provider=None,
        meter_provider=None,
        turn_off_message_logging=True,
    )
    assert logger.callback_name == "otel"
    assert logger.turn_off_message_logging is True
    assert logger.tracer is not None


def test_default_config_reads_env(monkeypatch):
    """No explicit config → reads env (exporter=console by default)."""
    monkeypatch.delenv("OTEL_EXPORTER", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
    logger = OpenTelemetryV2(
        tracer_provider=providers.build_tracer_provider(
            OpenTelemetryV2Config(exporter="in_memory")
        )
    )
    assert logger.config.exporter == "console"


def test_proxy_global_first_registered_wins(monkeypatch):
    """``_init_otel_logger_on_litellm_proxy`` claims the global only when empty."""
    proxy_server = pytest.importorskip("litellm.proxy.proxy_server")
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", None, raising=False)
    cfg = OpenTelemetryV2Config(exporter="in_memory")
    tp = providers.build_tracer_provider(cfg)

    first = OpenTelemetryV2(config=cfg, tracer_provider=tp)
    assert proxy_server.open_telemetry_logger is first

    second = OpenTelemetryV2(config=cfg, tracer_provider=tp)
    # Global still points at the first registration.
    assert proxy_server.open_telemetry_logger is first
    assert second is not first


def test_registers_into_litellm_service_callback(monkeypatch):
    """The logger must mutate ``litellm.service_callback`` in place. An empty
    list is falsy, so a ``getattr(..) or []`` would append to a throwaway local
    and service spans (Redis, …) would silently never fire on this logger.
    """
    import litellm

    pytest.importorskip("litellm.proxy.proxy_server")
    monkeypatch.setattr(litellm, "service_callback", [], raising=False)
    cfg = OpenTelemetryV2Config(exporter="in_memory")
    tp = providers.build_tracer_provider(cfg)

    first = OpenTelemetryV2(config=cfg, tracer_provider=tp)
    assert first in litellm.service_callback

    # A second OTel logger sees one is already registered and does not duplicate.
    OpenTelemetryV2(config=cfg, tracer_provider=tp)
    otel_registrations = [
        cb
        for cb in litellm.service_callback
        if cb.__class__.__module__.startswith("litellm.integrations.otel")
    ]
    assert len(otel_registrations) == 1


def test_registers_into_litellm_input_callback(monkeypatch):
    """The logger must land in ``litellm.input_callback`` — the list
    ``Logging.pre_call`` iterates to fire ``log_pre_api_call``. Without this the
    boundary hook never runs and the gen-AI span is never opened (the span goes
    completely missing). Deduped like ``service_callback``.
    """
    import litellm

    pytest.importorskip("litellm.proxy.proxy_server")
    monkeypatch.setattr(litellm, "input_callback", [], raising=False)
    cfg = OpenTelemetryV2Config(exporter="in_memory")
    tp = providers.build_tracer_provider(cfg)

    first = OpenTelemetryV2(config=cfg, tracer_provider=tp)
    assert first in litellm.input_callback

    OpenTelemetryV2(config=cfg, tracer_provider=tp)
    otel_registrations = [
        cb
        for cb in litellm.input_callback
        if cb.__class__.__module__.startswith("litellm.integrations.otel")
    ]
    assert len(otel_registrations) == 1


# --------------------------------------------------------------------------- #
#  Management endpoint hooks — no-ops: management endpoints are ordinary FastAPI
#  routes, so the mounted instrumentor spans them. The hooks must not emit.
# --------------------------------------------------------------------------- #


def test_management_hooks_are_noops():
    logger, exporter = _logger()

    class _Payload:
        route = "/key/generate"
        request_data = {"models": "gpt-4o"}
        response = {"key": "sk-123"}
        exception = ValueError("nope")
        start_time = end_time = None

    asyncio.run(logger.async_management_endpoint_success_hook(_Payload()))
    asyncio.run(logger.async_management_endpoint_failure_hook(_Payload()))
    assert exporter.get_finished_spans() == ()


# --------------------------------------------------------------------------- #
#  Guardrail span placement: request-level parent + real execution timestamps
# --------------------------------------------------------------------------- #


def _guardrail_request_data(*, start, end):
    return {
        "metadata": {
            "standard_logging_guardrail_information": [
                {
                    "guardrail_name": "openai-moderation",
                    "guardrail_mode": "pre_call",
                    "guardrail_status": "success",
                    "start_time": start,
                    "end_time": end,
                    "duration": end - start,
                }
            ],
        }
    }


def test_guardrail_span_parents_to_ambient_server_span():
    """The post-call hook runs in the request task with the server span ambient,
    so the guardrail span parents to it natively — no span threaded through
    metadata. (Auth already finished, so no phase span is active.)"""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    data = _guardrail_request_data(start=1000.0, end=1000.5)
    try:
        with trace.use_span(server, end_on_exit=False):
            asyncio.run(
                logger.async_post_call_success_hook(data, _Auth(), {"ok": True})
            )
    finally:
        server.end()
    g = {s.name: s for s in exporter.get_finished_spans()}[
        "execute_guardrail openai-moderation"
    ]
    assert g.parent.span_id == server.get_span_context().span_id


def test_guardrail_span_uses_actual_execution_timestamps():
    """A pre_call guardrail's span carries its real start/end (from the logging
    entry), so it sorts before the LLM call instead of at post-call emit time."""
    logger, exporter = _logger()
    server = logger._emitter.start_span(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    data = _guardrail_request_data(start=1700.0, end=1700.25)
    try:
        with trace.use_span(server, end_on_exit=False):
            asyncio.run(
                logger.async_post_call_success_hook(data, _Auth(), {"ok": True})
            )
    finally:
        server.end()
    g = {s.name: s for s in exporter.get_finished_spans()}[
        "execute_guardrail openai-moderation"
    ]
    assert g.start_time == to_ns(1700.0)
    assert g.end_time == to_ns(1700.25)
