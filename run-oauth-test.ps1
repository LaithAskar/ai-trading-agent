# V8c manual test — OpenRouter OAuth round-trip.
#
# OpenRouter only redirects OAuth callbacks to https:443 or http://localhost:3000,
# so we force the callback URL to localhost:3000 and run Streamlit on that port.
# (No .env edit needed: a shell env var wins over the .env default.)
#
# Run it:   .\run-oauth-test.ps1
# Then:
#   1. Open http://localhost:3000
#   2. Sidebar -> "Sign in with OpenRouter" -> authorize on openrouter.ai
#   3. Confirm it returns to the app showing "Signed in with OpenRouter"
#   4. Open "Run Agent", run a small goal, confirm >=1 iteration completes
#      (this is what validates the Anthropic-via-OpenRouter call path end to end)
#   5. Sign out, expand "Advanced", paste an Anthropic key, confirm that path still runs
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:OPENROUTER_CALLBACK_URL = "http://localhost:3000"
& ".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.port 3000
