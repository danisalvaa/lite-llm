# Strict marketplace authentication evidence

Interactive Claude Code `/plugin marketplace add http://127.0.0.1:4000/claude-code/marketplace.json` against a local proxy and PostgreSQL database, with `general_settings.claude_code_marketplace_auth_required: true` in both runs

Before: `e73f949fbb735b344f75e5f8c395344bf4cfdb54`, marketplace added without credentials

After: `97fcc19a32b47def0d1ba6de28e70915adabd00e`, anonymous access rejected with HTTP 401

Captured on 2026-09-23 using Claude Code 2.1.269. These are captures of the actual interactive terminal window. No model inference was needed for catalog discovery

The local Windows launcher uses a separate console and a native process-existence check to prevent signal-zero process polling from interrupting the console. The same launcher was used for both runs. Marketplace authentication and database access were not mocked
