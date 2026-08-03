# KCAL Tracker

A minimal calorie tracking app with a single-file Python backend and an inline React frontend.

## Running

```bash
docker compose up -d --build
```

The app is available at `http://localhost:8765`.

## Protein estimation (optional)

When an entry is added, an LLM estimates its protein content from the description
and returns a per-item breakdown. Set an [OpenRouter](https://openrouter.ai/keys)
key to enable it:

```bash
echo 'OPENROUTER_API_KEY=sk-or-...' > .env
docker compose up -d --build
```

Without a key the app behaves exactly as before and no protein is shown.

Estimation is best-effort and never blocks anything. The entry is saved and its
calories counted immediately; the estimate arrives a second or two later. If the
call fails, the entry keeps its calories and the row offers a retry. The model is
`deepseek/deepseek-v4-flash-0731`, called once per entry with no retries.

## API

All read-only endpoints. Base URL: `http://localhost:8765/api`

### Get a day's data

Returns entries, total kcal, limit, burn rate, and skip status for a given date.

```bash
curl http://localhost:8765/api/days/2026-07-16
```

```json
{
  "date": "2026-07-16",
  "limit": 1700,
  "burn": 2200,
  "total": 2465,
  "total_protein_g": 47.8,
  "protein_complete": true,
  "skipped": false,
  "entries": [
    {
      "id": 112, "kcal": 600, "description": "3 eggs, 2 slices of protein bread", "time": "12:34",
      "protein_g": 29.8,
      "protein_items": [
        { "name": "3 eggs", "protein_g": 18.0 },
        { "name": "2 slices of protein bread", "protein_g": 11.8 }
      ],
      "protein_state": "ok"
    },
    { "id": 113, "kcal": 450, "description": "dinner", "time": "19:10",
      "protein_g": 18.0, "protein_items": [], "protein_state": "ok" }
  ]
}
```

`protein_state` is one of `pending` (estimate in flight), `ok`, `failed`
(retryable), or `skipped` (estimation disabled, or an entry predating the
feature). `protein_complete` is `false` when any entry lacks an estimate, which
makes `total_protein_g` a partial figure.

### Retry a protein estimate

Re-runs estimation for one entry. Returns 503 if no API key is configured.

```bash
curl -X POST http://localhost:8765/api/entries/112/protein
```

### Average daily intake

Returns the average kcal per day over the last N days (excluding today). Skipped (cheat) days count as 4000 kcal.

```bash
curl http://localhost:8765/api/average/7
```

```json
{
  "days_requested": 7,
  "days_counted": 7,
  "average_kcal": 1944.4,
  "days": [
    { "date": "2026-07-10", "total": 1710, "skipped": false },
    { "date": "2026-07-11", "total": 2087, "skipped": false },
    { "date": "2026-07-17", "total": 4000, "skipped": true }
  ]
}
```

### Cumulative weight change

Returns the total estimated weight change since the first tracked day, based on daily deficit/surplus against the configured burn rate (7.7 kcal per gram of fat). Skipped days count as 4000 kcal consumed.

```bash
curl http://localhost:8765/api/cumulative
```

```json
{
  "total_grams": 1234.567,
  "days_counted": 19,
  "days": [
    {
      "date": "2026-06-28",
      "consumed": 2050,
      "burn": 2200,
      "deficit": 150,
      "grams": 19.481,
      "skipped": false
    },
    {
      "date": "2026-07-17",
      "consumed": 4000,
      "burn": 2200,
      "deficit": -1800,
      "grams": -233.766,
      "skipped": true
    }
  ]
}
```
