"""``CustomLogger`` adapter on the OpenTelemetry span engine.

Thin adapter: it translates litellm's logging callbacks into typed ``*SpanData``
and hands them to the engine (:mod:`emitter`), with multi-tenant tracer routing
in :mod:`routing`. It emits the gen-ai spans (LLM call, guardrail, service).

The proxy server span is NOT owned here. It is created by the FastAPI
instrumentation mounted in ``proxy_server``'s startup event, which stamps the
``http.*`` attributes and handles inbound context propagation. The proxy-span
methods below are therefore no-ops: routes never modify spans.

Gen-ai spans parent to that server span via the ambient OTel context rather
than a ``Span`` threaded through a request-metadata dict. The LLM-call span in
particular is **born at the call boundary**: ``log_pre_api_call`` opens it while
the request task is still on the stack (so the live server span is genuinely
ambient and becomes its parent), and the async success/failure callback closes
it once the typed ``StandardLoggingPayload`` is available. The open span is held
in a bounded cache keyed by ``litellm_call_id`` (present in the callback kwargs at
both ``pre_call`` and close) until it is closed — no live span ever travels
through ``litellm_params``, and the logging object need not be reachable from the
callback. For the boundary hook to fire, the logger is registered into
``litellm.input_callback`` (the list ``Logging.pre_call`` iterates).

When ``pre_call`` runs off the request task (a provider with no async support is
driven through a thread pool, where contextvars don't propagate), no ambient
server span is visible, so creation is **deferred** to the async callback —
whose worker context was copied from the request task at enqueue and so still
carries the server span. Either way the parent comes from real ambient context.

"Did ``pre_call`` run?" is therefore the single signal for "did the LLM call
actually happen": a request rejected at the auth/budget gate or blocked by a
pre-call guardrail never reaches ``pre_call``, so no span is opened and the
failure log can't produce a phantom CLIENT span — no post-hoc heuristics needed.
"""

from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterator, Mapping, cast

from opentelemetry.context import attach, get_current
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, Tracer, get_current_span, use_span

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.otel.baggage import promoted_baggage
from litellm.integrations.otel.config import OpenTelemetryV2Config
from litellm.integrations.otel.context import (
    is_recordable_span,
    resolve_parent_context,
    set_request_baggage,
)
from litellm.integrations.otel.emitter import SpanEmitter
from litellm.integrations.otel.mappers import resolve_mappers
from litellm.integrations.otel.payloads import (
    GuardrailSpanData,
    LLMCallSpanData,
    RequestIdentity,
    ServiceSpanData,
    SpanError,
)
from litellm.integrations.otel.providers import build_tracer_provider, get_tracer
from litellm.integrations.otel.routing import TenantTracerCache
from litellm.integrations.otel.semconv import resolve_operation
from litellm.integrations.otel.spans import SpanRole, span_role_for_service
from litellm.integrations.otel.utils import as_str, to_ns

if TYPE_CHECKING:
    from litellm.types.utils import StandardLoggingGuardrailInformation

LITELLM_TRACER_NAME = "litellm"
LITELLM_PROXY_REQUEST_SPAN_NAME = "Received Proxy Server Request"

# Any callback whose class belongs to one of these modules is "the OTel
# callback" for proxy-global-registration purposes.
_OTEL_MODULES = (
    "litellm.integrations.otel",
    "litellm.integrations.opentelemetry",
)


# Cap on the open-call carrier map. A span opened at ``pre_call`` that never
# reaches a success/failure callback (e.g. a stream that only fires stream
# events) would otherwise linger; bounding the map evicts the oldest so memory
# stays flat on a long-running proxy while covering every concurrent in-flight
# call.
_OPEN_CALLS_MAX = 10_000


class _LLMCallSpan:
    """The state carried from the ``pre_call`` boundary to span close.

    ``span`` is the live span when it could be opened at the boundary (the server
    span was ambient), or ``None`` when creation was deferred because no ambient
    parent was visible — in which case the async callback creates it against its
    own (worker-copied) ambient context using ``start_time_ns``. The presence of
    a carrier for a call at all is the proof that ``pre_call`` ran, i.e. that an
    upstream call was actually attempted.
    """

    __slots__ = ("span", "start_time_ns")

    def __init__(self, span: "Span | None", start_time_ns: int | None) -> None:
        self.span = span
        self.start_time_ns = start_time_ns


def _call_id(kwargs: Mapping[str, Any]) -> str | None:
    """The ``litellm_call_id`` correlating ``pre_call`` with the close callback.

    Present in ``model_call_details`` at ``pre_call`` and in both the kwargs and
    the ``standard_logging_object`` at success/failure, so it's a stable key for
    the open-call carrier — no back-reference to the logging object required (the
    object isn't reachable from the callback kwargs at ``pre_call`` time).
    """
    payload = kwargs.get("standard_logging_object")
    if isinstance(payload, Mapping):
        call_id = as_str(payload.get("litellm_call_id")) or as_str(payload.get("id"))
        if call_id:
            return call_id
    return as_str(kwargs.get("litellm_call_id"))


def _provisional_llm_span_name(kwargs: Mapping[str, Any]) -> str:
    """A best-effort ``"{operation} {model}"`` name known at ``pre_call`` time.

    The span is renamed from the typed payload at close (``finish_span``); this
    only needs to be reasonable for a span that never gets closed (a leak).
    """
    operation = resolve_operation(as_str(kwargs.get("call_type")))
    model = as_str(kwargs.get("model")) or ""
    return f"{operation.value} {model}".strip()


class OpenTelemetryV2(CustomLogger):
    """The ``CustomLogger`` for OpenTelemetry.

    The constructor accepts an optional config, callback name, and pre-built
    OTel providers; when a provider is omitted it is built from the config.
    ``logger_provider`` and ``meter_provider`` are accepted but reserved for
    future OTel logs and metrics support.
    """

    def __init__(
        self,
        config: OpenTelemetryV2Config | None = None,
        callback_name: str | None = None,
        tracer_provider: TracerProvider | None = None,
        logger_provider: Any | None = None,  # reserved for OTel logs
        meter_provider: Any | None = None,  # reserved for metrics
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Build the config from any settings passed through ``kwargs`` so
        # ``callback_settings.otel.*`` in config.yaml (e.g. ``baggage_promoted_keys``,
        # ``capture_message_content``) configures the logger. ``OpenTelemetryV2Config``
        # ignores extra keys, so unrelated kwargs are dropped harmlessly.
        self.config: OpenTelemetryV2Config = config or OpenTelemetryV2Config(**kwargs)
        self.callback_name = callback_name
        self._tracer_provider: TracerProvider = (
            tracer_provider
            if tracer_provider is not None
            else build_tracer_provider(self.config)
        )
        self.tracer: Tracer = get_tracer(self._tracer_provider, LITELLM_TRACER_NAME)
        self._emitter = SpanEmitter(
            self.tracer, self.config, mappers=resolve_mappers(self.config.mapper_names)
        )
        self._tenant_tracers = TenantTracerCache(
            self.config, callback_name, LITELLM_TRACER_NAME
        )
        # LLM-call spans opened at the ``pre_call`` boundary, keyed by
        # ``litellm_call_id`` until the success/failure callback closes them.
        # Bounded so a call that never closes can't grow it without limit.
        self._open_llm_calls: "OrderedDict[str, _LLMCallSpan]" = OrderedDict()
        self._init_otel_logger_on_litellm_proxy()

    # ====================================================================== #
    #  Proxy global registration
    # ====================================================================== #

    def _register_in_callback_list(self, callbacks: list) -> None:
        """Append ``self`` to a global litellm callback list, in place and deduped.

        Mutates the list in place rather than via ``getattr(..) or []`` — an empty
        list is falsy, so the latter would bind a throwaway local and the append
        would never reach the global. Skips when an OTel-module callback is already
        registered so a second logger doesn't double up.
        """
        already_otel = any(
            cb.__class__.__module__.startswith(_OTEL_MODULES)
            for cb in callbacks
            if hasattr(cb, "__class__")
        )
        if not already_otel:
            callbacks.append(self)

    def _init_otel_logger_on_litellm_proxy(self) -> None:
        """Claim ``proxy_server.open_telemetry_logger`` if no one else has."""
        try:
            from litellm.proxy import proxy_server
        except Exception:
            return
        try:
            # ``service_callback`` drives the Redis/Postgres service spans.
            self._register_in_callback_list(litellm.service_callback)
            # ``input_callback`` is the list ``Logging.pre_call`` iterates to fire
            # ``log_pre_api_call`` — where the LLM-call span is opened at the call
            # boundary. Without this the boundary hook never runs and the gen-AI
            # span is never created. (It's the *sync* input list, matching our
            # sync ``log_pre_api_call``.)
            self._register_in_callback_list(litellm.input_callback)
        except Exception:
            pass
        if getattr(proxy_server, "open_telemetry_logger", None) is None:
            setattr(proxy_server, "open_telemetry_logger", self)

    # ====================================================================== #
    #  LLM-call callbacks — the span is opened at the ``pre_call`` boundary and
    #  closed here. See ``log_pre_api_call``.
    # ====================================================================== #

    def log_pre_api_call(self, model, messages, kwargs):
        """Open the LLM-call span at the call boundary.

        Runs synchronously inside the request task, before the upstream call —
        the one place where the live server span is genuinely the ambient OTel
        context — so the span parents to it natively, with no span threaded
        through a metadata dict. The open span is stashed on the per-request
        ``LiteLLMLoggingObj`` (a typed object) and closed in the async callback.

        When no recordable ambient span is visible (``pre_call`` was driven from
        a thread pool for a sync-only provider, where contextvars don't follow),
        creation is deferred: only the start time is recorded, and the async
        callback — whose worker context was copied from the request task and so
        still carries the server span — creates the span then.
        """
        call_id = _call_id(kwargs)
        if call_id is None:
            return
        # Idempotent: a retried call may re-enter ``pre_call`` with the same
        # call id; keep the first span so its start time is the true one.
        if call_id in self._open_llm_calls:
            return
        start_time_ns = to_ns(datetime.now())
        span: Span | None = None
        if is_recordable_span(get_current_span()):
            span = self._emitter.start_span(
                SpanRole.LLM_CALL,
                _provisional_llm_span_name(kwargs),
                parent_context=resolve_parent_context(),
                start_time_ns=start_time_ns,
                tracer=self._tenant_tracers.tracer_for(
                    self.tracer, kwargs.get("standard_callback_dynamic_params")
                ),
            )
        self._open_llm_calls[call_id] = _LLMCallSpan(
            span=span, start_time_ns=start_time_ns
        )
        # Evict the oldest open call if the map is over budget. A call that opens
        # but never closes (a stream that only fires stream events) would linger
        # otherwise; the evicted span is simply dropped (never exported).
        if len(self._open_llm_calls) > _OPEN_CALLS_MAX:
            self._open_llm_calls.popitem(last=False)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        return None

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        return None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._close_llm_call(kwargs, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._close_llm_call(kwargs, start_time, end_time)

    def _close_llm_call(
        self,
        kwargs: Mapping[str, Any],
        start_time: datetime | float | None,
        end_time: datetime | float | None,
    ) -> Span | None:
        """Finish the LLM-call span opened at ``pre_call`` (or create it deferred).

        No carrier for this call id means ``pre_call`` never ran — the request was
        rejected at the gate or blocked by a pre-call guardrail before any upstream
        call — so there is nothing to record and no phantom span.
        """
        call_id = _call_id(kwargs)
        # ``pop`` is the dedup: this method runs from both the success and failure
        # paths, and whichever fires first removes the carrier and closes the span.
        carrier = self._open_llm_calls.pop(call_id, None) if call_id else None
        if carrier is None:
            return None
        payload = kwargs.get("standard_logging_object")
        if not payload:
            if carrier.span is not None:
                # Opened at the boundary but the payload never materialized — end
                # it (named provisionally) so it isn't leaked as an open span.
                carrier.span.end(end_time=to_ns(end_time))
            return None
        data = LLMCallSpanData.from_standard_logging_payload(
            cast("Any", payload), capture_content=self.config.capture_span_content
        )
        end_time_ns = to_ns(end_time)
        if carrier.span is not None:
            # Born at the boundary: stamp attributes from the typed payload, set
            # status, and end it. Its parent (the server span) was captured at
            # creation from real ambient context.
            self._emitter.finish_span(
                SpanRole.LLM_CALL, carrier.span, data, end_time_ns=end_time_ns
            )
            return carrier.span
        # Deferred: ``pre_call`` had no ambient parent, so create the span now
        # against this callback's ambient context (the worker copied the request
        # task's context, which carries the server span). Seed identity Baggage so
        # the span — and the SDK path, which has none — is labeled consistently.
        parent_ctx = resolve_parent_context()
        bag = promoted_baggage(
            data.identity,
            data.request_model,
            promoted_keys=tuple(self.config.baggage_promoted_keys),
            metadata_keys=tuple(self.config.baggage_metadata_keys),
        )
        if bag:
            parent_ctx = set_request_baggage(bag, context=parent_ctx)
        return self._emitter.emit(
            SpanRole.LLM_CALL,
            data,
            parent_context=parent_ctx,
            start_time_ns=carrier.start_time_ns,
            end_time_ns=end_time_ns,
            tracer=self._tenant_tracers.tracer_for(
                self.tracer, kwargs.get("standard_callback_dynamic_params")
            ),
        )

    # ====================================================================== #
    #  Service hooks
    # ====================================================================== #

    async def async_service_success_hook(
        self,
        payload: Any,
        parent_otel_span: Span | None = None,
        start_time: datetime | float | None = None,
        end_time: datetime | float | None = None,
        event_metadata: dict | None = None,
    ) -> None:
        self._emit_service(
            payload,
            parent_otel_span=parent_otel_span,
            start_time=start_time,
            end_time=end_time,
            event_metadata=event_metadata,
            error_override=None,
        )

    async def async_service_failure_hook(
        self,
        payload: Any,
        error: str | None = "",
        parent_otel_span: Span | None = None,
        start_time: datetime | float | None = None,
        end_time: datetime | float | None = None,
        event_metadata: dict | None = None,
    ) -> None:
        self._emit_service(
            payload,
            parent_otel_span=parent_otel_span,
            start_time=start_time,
            end_time=end_time,
            event_metadata=event_metadata,
            error_override=error or "error",
        )

    def _emit_service(
        self,
        payload: Any,
        *,
        parent_otel_span: Span | None,
        start_time: datetime | float | None,
        end_time: datetime | float | None,
        event_metadata: dict | None,
        error_override: str | None,
    ) -> Span | None:
        data = ServiceSpanData.from_payload(payload, event_metadata=event_metadata)
        # Decide whether this service call is a span at all, and of what kind.
        # ``None`` means metrics-only (framework instrumentation that duplicates a
        # gen-AI span — ``self``/``router``/``proxy_pre_call`` — or ``auth``, which
        # gets a live phase span instead). Those still feed Prometheus/Datadog via
        # their own hooks; they just never enter the trace.
        role = span_role_for_service(data.service_name)
        if role is None:
            return None
        # A metrics-only ping with neither timing nor a parent (in-memory queue
        # gauges) is not a traceable operation; a span for it would be a
        # zero-duration root with no context, so skip it. Real background work
        # (budget/reset jobs, spend flush) passes start/end times and still emits
        # as a root; anything with a parent emits regardless.
        if (
            error_override is None
            and start_time is None
            and end_time is None
            and parent_otel_span is None
        ):
            return None
        if error_override is not None and data.error is None:
            data = ServiceSpanData(
                service_name=data.service_name,
                call_type=data.call_type,
                error=SpanError(message=error_override),
                event_metadata=data.event_metadata,
            )
        # Parent like every other span: ambient context first (so identity Baggage
        # rides along and the call nests under whatever request phase is active —
        # e.g. a DB lookup under the live ``auth`` span), falling back to the
        # server span the proxy threaded as ``parent_otel_span``. A background
        # service call has neither, so it starts its own root trace.
        parent_context = resolve_parent_context(threaded=parent_otel_span)
        return self._emitter.emit(
            role,
            data,
            parent_context=parent_context,
            start_time_ns=to_ns(start_time),
            end_time_ns=to_ns(end_time),
        )

    # ====================================================================== #
    #  async_post_call_* hooks — emit guardrail spans. The server span's status
    #  / errors are the FastAPI instrumentor's job, so we don't touch it here.
    # ====================================================================== #

    def seed_request_identity(self, user_api_key_dict: Any, model: Any = None) -> None:
        """Attach request-identity Baggage to the current context + server span.

        Seeding identity into Baggage makes **every** span emitted afterwards for
        this request — LLM call, guardrail, DB call — inherit it via
        ``LiteLLMBaggageSpanProcessor``. Called once at the auth boundary (as soon
        as the key resolves) so post-auth spans are labeled consistently; the
        Baggage rides the request task's contextvar from there on. Auth-internal
        DB lookups that run before the key is known stay unlabeled — identity
        isn't determined yet, which is correct.
        """
        try:
            identity = RequestIdentity.from_user_api_key_auth(user_api_key_dict)
            bag = promoted_baggage(
                identity,
                model,
                promoted_keys=tuple(self.config.baggage_promoted_keys),
                metadata_keys=tuple(self.config.baggage_metadata_keys),
            )
            if bag:
                # Attach (no detach): the contextvar is scoped to this request's
                # asyncio task and is reclaimed when the task ends.
                attach(set_request_baggage(bag, context=get_current()))
                # The server span was started by the instrumentor before this ran,
                # so the Baggage processor (which only fires at span start) won't
                # backfill it — stamp identity on it directly.
                server_span = get_current_span()
                if is_recordable_span(server_span):
                    for key, value in bag.items():
                        server_span.set_attribute(key, value)
        except Exception:
            pass

    @contextmanager
    def start_phase_span(self, name: str) -> "Iterator[Span]":
        """Open a live, **active** INTERNAL span for a request phase (e.g. auth).

        Unlike the post-hoc service spans (emitted from start/end timestamps after
        the fact), this span is the active OTel context for the duration of the
        ``with`` block. Service/DB calls fired inside it — even via
        ``asyncio.create_task``, which copies the active context — therefore nest
        under it instead of flattening onto the server span.
        """
        span = self._emitter.start_span(SpanRole.SERVICE, name)
        with use_span(span, end_on_exit=True):
            yield span

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: Any,
    ) -> dict:
        """Re-seed identity Baggage in the request task.

        Identity is first seeded at the auth boundary (``seed_request_identity``),
        but this hook re-seeds with the request ``model`` now known and covers
        entrypoints that don't pass through that boundary (e.g. the SDK). Idempotent.
        """
        self.seed_request_identity(
            user_api_key_dict,
            model=data.get("model") if isinstance(data, dict) else None,
        )
        return data

    async def async_post_call_success_hook(
        self,
        data: Mapping[str, Any],
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        self._emit_guardrail_spans(data)
        return response

    async def async_post_call_failure_hook(
        self,
        request_data: Mapping[str, Any],
        original_exception: BaseException | None,
        user_api_key_dict: Any,
        traceback_str: str | None = None,
    ) -> None:
        self._emit_guardrail_spans(request_data)

    def _emit_guardrail_spans(self, request_data: Mapping[str, Any]) -> None:
        # The post-call hooks run inside the request task, where the server span
        # is the ambient OTel context, so each guardrail span parents to it
        # natively — no span threaded through metadata. Emit with the guardrail's
        # actual execution window so a pre_call guardrail is placed before the LLM
        # call rather than at post-call emission time.
        metadata = request_data.get("metadata")
        guardrails: list[Any] = []
        if isinstance(metadata, dict):
            info = metadata.get("standard_logging_guardrail_information")
            if isinstance(info, list):
                guardrails = info
            elif isinstance(info, dict):
                guardrails = [info]
        if not guardrails:
            return
        parent_ctx = resolve_parent_context()
        for entry in guardrails:
            if not isinstance(entry, dict):
                continue
            data = GuardrailSpanData.from_logging_entry(
                cast("StandardLoggingGuardrailInformation", entry)
            )
            self._emitter.emit(
                SpanRole.GUARDRAIL,
                data,
                parent_context=parent_ctx,
                start_time_ns=to_ns(data.start_time),
                end_time_ns=to_ns(data.end_time),
            )

    # ====================================================================== #
    #  Management endpoint hooks — no-ops. Management endpoints are ordinary
    #  FastAPI routes, so the mounted instrumentor already spans them.
    # ====================================================================== #

    async def async_management_endpoint_success_hook(
        self,
        logging_payload: Any,
        parent_otel_span: Span | None = None,
    ) -> None:
        return None

    async def async_management_endpoint_failure_hook(
        self,
        logging_payload: Any,
        parent_otel_span: Span | None = None,
    ) -> None:
        return None

    # ====================================================================== #
    #  Proxy SERVER-span API — no-ops. The FastAPI instrumentor owns the server
    #  span (creation, http.* attributes, inbound propagation) and gen-ai spans
    #  parent to it via ambient context. These methods are the surface the
    #  proxy and auth call sites invoke; they intentionally do nothing.
    # ====================================================================== #

    def create_litellm_proxy_request_started_span(
        self, start_time: datetime, headers: Mapping[str, str] | None
    ) -> Span | None:
        """Return the active server span instead of creating one.

        The FastAPI instrumentor owns the server span, so V2 creates nothing
        here. But the proxy threads this return value as ``litellm_parent_otel_span``
        — and service logging (Redis, Postgres, …) only invokes the OTel service
        hook when that parent is non-None. Returning the ambient server span lets
        service spans nest under it. The proxy must NOT ``.end()`` this span (the
        instrumentor does); ``_close_dangling_otel_server_span`` skips it under V2.
        """
        span = get_current_span()
        return span if is_recordable_span(span) else None

    @staticmethod
    def set_proxy_request_route_attributes(
        span: Span | None,
        *,
        url_path: str | None = None,
        http_route: str | None = None,
    ) -> None:
        """No-op: the FastAPI instrumentor stamps ``http.route`` / ``url.path``."""

    @staticmethod
    def set_response_status_code_attribute(
        span: Span | None, status_code: int | None
    ) -> None:
        """No-op: the FastAPI instrumentor stamps ``http.response.status_code``."""

    @staticmethod
    def set_preprocessing_duration_attribute(span: Span | None, container: Any) -> None:
        """No-op: the server span belongs to the FastAPI instrumentor."""


# ====================================================================== #
#  Module-level seam for proxy-core call sites (auth, …). These resolve the
#  registered V2 logger and no-op when V2 is not the active logger, so the
#  proxy can call them unconditionally without importing the OTel SDK or
#  knowing whether V2 is enabled.
# ====================================================================== #


def _registered_v2_logger() -> "OpenTelemetryV2 | None":
    """The proxy's registered logger if it is the V2 ``OpenTelemetryV2``, else None."""
    try:
        from litellm.proxy import proxy_server
    except Exception:
        return None
    logger = getattr(proxy_server, "open_telemetry_logger", None)
    return logger if isinstance(logger, OpenTelemetryV2) else None


def seed_request_identity(user_api_key_dict: Any, model: Any = None) -> None:
    """Seed request-identity Baggage at the auth boundary (no-op without V2)."""
    logger = _registered_v2_logger()
    if logger is not None:
        logger.seed_request_identity(user_api_key_dict, model=model)


@contextmanager
def phase_span(name: str) -> "Iterator[Span | None]":
    """Run a request phase inside a live active span so its DB/service calls nest.

    A no-op (yields ``None``) when V2 is not the active logger, so proxy-core
    call sites can wrap a phase unconditionally.
    """
    logger = _registered_v2_logger()
    if logger is None:
        yield None
        return
    with logger.start_phase_span(name) as span:
        yield span
