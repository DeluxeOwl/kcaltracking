# KCAL Tracker

A minimal calorie tracking app with a single-file Python backend and an inline React frontend.

## Running

```bash
docker compose up -d --build
```

The app is available at `http://localhost:8765`.

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
  "skipped": false,
  "entries": [
    { "id": 112, "kcal": 600, "description": "lunch", "time": "12:34" },
    { "id": 113, "kcal": 450, "description": "dinner", "time": "19:10" }
  ]
}
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
