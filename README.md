# Store Assistant (Sarvam AI)
Voice agent for shop owners: live customer<->owner translation + voice transaction logging.

## Run
    pip install -r requirements.txt
    export SARVAM_API_KEY=your_key   # from dashboard.sarvam.ai
    uvicorn app.main:app --reload --port 8000
Open http://localhost:8000 (mic needs localhost or HTTPS; on a phone use ngrok/cloudflared).

## Sarvam APIs used
STT `POST /speech-to-text` (saaras:v4, language auto-detect) · Translate `POST /translate` (sarvam-translate:v1) ·
TTS `POST /text-to-speech` (bulbul:v3) · Chat `POST /v1/chat/completions` (sarvam-30b; set SARVAM_LLM=sarvam-105b for accuracy).

## Modes
- **Talk**: customer speech -> detected language (kept sticky) -> owner's language text+voice. Owner replies -> customer's language.
- **Log sale**: speak -> LLM extracts JSON (validated with Pydantic, 1 retry) -> read back -> tap Save or say "yes"; say a correction to amend. Missing amount/items triggers a spoken follow-up. Duplicate saves ignored (idempotent id). Undo last supported.
- **Ask**: "aaj ki sale kitni?", "Ramesh ka udhaar?" -> LLM picks one of 4 fixed tools (parameterized SQL, never raw SQL).
- **Dashboard**: today's totals, credit balances, 7-day list, CSV export.

## Notes
Money is stored as integer paise. REST STT handles audio under ~30s. Mode switching is manual (tabs).
