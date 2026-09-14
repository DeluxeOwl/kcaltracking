# KCAL Tracker — agent context

## Start here

This is a small, actively used local calorie tracker. Treat the running app and
its SQLite database as user data, not as a disposable development fixture.

- `main.py` is the source of truth for the application. It contains the
  FastAPI backend, SQLite repository, LLM estimators, API schemas/routes, and
  the complete inline React/TypeScript frontend in `HTML`.
- `docker-compose.yml` defines the local deployment. It runs one service,
  `kcaltracking`, publishes port `8765`, passes the optional
  `OPENROUTER_API_KEY`, and bind-mounts `./data` at `/app/data`.
- `Dockerfile` is the source of truth for the container command: it runs
  `uv run --script main.py` against `/app/data/kcal.db` on port `8765`.
- `README.md` contains older protein-only examples and can be stale. Confirm
  behavior, field names, model names, and routes in `main.py` before relying on
  it.

The important runtime path is:

```text
browser
  -> http://localhost:8765
  -> Docker Compose: kcaltracking
  -> uv / main.py / FastAPI
  -> /app/data/kcal.db  <->  host ./data/kcal.db
```

The current service has no authentication. The Compose port mapping publishes
`8765` on the Docker host; treat it as a local app and do not add remote access
implicitly.

## What the app does

### Stored data and calculations

The repository stores four kinds of durable state in SQLite:

- `entries`: positive integer kcal, a non-blank description (maximum 500
  characters), a local calendar date, and an optional `HH:MM` time.
- `daily_limits`: a calorie limit effective from its date forward. Looking up a
  date uses the latest row whose `entry_date` is less than or equal to it.
- `daily_burns`: the same effective-from-date lookup for a burn rate.
- `day_marks`: an optional mark for a date: `cheat` scores the date as exactly
  4000 kcal, while `excluded` removes it from average and cumulative
  calculations. Clearing the mark returns the date to ordinary entry scoring.

An ordinary day's displayed kcal total is the sum of its entries. A marked day
keeps its entries in the database, but the mark overrides how aggregate
calculations score them. The average covers the previous N calendar days,
excluding today, but counts only dates with entries or a cheat mark; empty
ordinary dates do not count. The cumulative calculation skips today and future
 dates, requires an effective burn rate, and converts burn minus consumed kcal
at 7.7 kcal per gram. Excluded dates contribute neither deficit nor surplus.

### Macro estimation

When `OPENROUTER_API_KEY` is present, adding an entry saves it immediately and
starts a FastAPI background task. One LLM call estimates per-item protein, fat,
and fiber. The entry state is `pending`, `ok`, or retryable `failed`. Without a
key it is `skipped`. Macro failures never remove or change the entry's kcal.
The frontend polls a day while an estimate is pending.

`POST /api/estimate` is separate: it estimates kcal, macros, weights, and
per-100g densities for free text, computes totals in Python, and saves nothing.
It returns `503` when the key is unavailable and `502` when the call fails.

### Frontend shape

The browser UI is not a separate frontend project. It is served from the
`HTML` string in `main.py`; Babel, Tailwind, React, React Query, and other
frontend dependencies load from CDNs at runtime. UI/API changes usually need
matching edits to the backend schemas/routes, the inline client types and
methods, and the relevant components in the same file.

The UI supports local-date navigation, entry add/delete, effective limit and
burn setters, cheat/excluded day marks, average and cumulative cards, a quick
kcal estimator, macro breakdown expansion/retry, keyboard day navigation, and a
light/dark theme. Date helpers deliberately avoid UTC conversion around
midnight.

### API surface

The API router is mounted at `/api` and currently provides:

- `GET /api/days/{YYYY-MM-DD}`
- `POST /api/entries`, `POST /api/entries/{id}/macros`,
  `DELETE /api/entries/{id}`
- `PUT /api/limits`, `PUT /api/burns`, `PUT /api/day-mark`
- `POST /api/estimate`
- `GET /api/average/{days}`, `GET /api/cumulative`

Pydantic validates real ISO dates, 24-hour times, positive bounded kcal values,
and non-blank descriptions. The catch-all route serves the SPA HTML for
non-API paths.

## SQLite and migration safety

`SqliteKcalRepository` opens one shared SQLite connection with
`check_same_thread=False`, enables WAL mode, creates missing tables, performs
startup migrations, and commits them. A lock protects writes because macro
background tasks can write while request handlers run. Startup is therefore a
potential data/schema mutation, not a read-only initialization.

The built-in migrations currently do the following, in this order:

1. Rename legacy `skipped_days` to `day_marks` when needed, then add a missing
   `mark` column with the historical `cheat` default.
2. Create the current `day_marks` table.
3. Rename legacy `protein_items`/`protein_state` entry columns to
   `macro_items`/`macros_state` when the new names are absent.
4. Add missing `protein_g`, `fat_g`, `fiber_g`, `macro_items`, and
   `macros_state` columns, with old rows defaulting to `skipped` where
   appropriate.

`CREATE TABLE IF NOT EXISTS` does not alter an existing table. Any future
schema change needs an explicit, idempotent, forward migration in the
repository initialization path, with old rows and legacy names accounted for.
Keep the migration ordering around `day_marks` intact.

### Required procedure for data or schema work

1. Inspect the live state first with `docker compose ps` and recent service
   logs. Use the existing service as the single writer; an extra local server
   must use a separate database and port.
2. Make a consistent SQLite backup before changing schema or transforming
   data. Prefer SQLite's online backup command while the service is running:

   ```bash
   backup="data/kcal.db.bak-$(date +%Y%m%d%H%M%S)"
   sqlite3 data/kcal.db ".backup '$backup'"
   ```

   If the host lacks the `sqlite3` CLI, use Python's `Connection.backup()` API
   from the container or another environment with SQLite support. Verify that
   the backup opens and contains the expected tables. Keep the backup until
   the running app and representative API reads have been checked.
3. Test a new migration or data transformation against the backup first. Use
   a separate database path and, if starting a test server, a separate port.
   Preserve ambiguous or destructive data and ask the user before choosing a
   lossy conversion.
4. Apply the smallest idempotent change. On the live deployment, restart the
   one Compose service deliberately so the startup migration runs, then check
   logs and `http://localhost:8765` or a read-only API request.
5. Treat `kcal.db-wal` and `kcal.db-shm` as part of the live SQLite state. Use
   SQLite backup/checkpoint mechanisms rather than copying or deleting only
   `kcal.db` while the process is active. Preserve the bind-mounted `data`
   directory and existing backup files.

Destructive database resets, dropping columns/tables, deleting the database,
or changing the mounted database location require explicit user confirmation.
Do not use a Compose operation that removes the data volume as a convenience.

## Development and verification

For a backend-only edit, inspect the route/schema and repository paths that the
change touches, then run a syntax check such as:

```bash
python -m py_compile main.py
```

For container behavior, prefer the existing deployment workflow:

```bash
docker compose up -d --build kcaltracking
docker compose ps
docker compose logs --tail=100 kcaltracking
```

Smoke-test with read-only requests such as `GET /api/days/<date>`,
`GET /api/average/7`, and `GET /api/cumulative`. Do not start a second process
on port `8765` or point a test process at `data/kcal.db`. There is currently no
separate frontend build or test suite in the repository, so changes to the
inline UI should be verified through the running container and browser/API
smoke tests.

When changing a response, update the Pydantic response model, the inline
TypeScript interface/client, and every renderer that consumes the field. When
changing persistence, account for existing WAL-backed data and add or update a
migration before claiming the change is complete.
