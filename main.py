#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click==8.4.2",
#     "fastapi==0.141.1",
#     "pydantic==2.13.4",
#     "pydantic-ai-slim[openrouter]==2.22.0",
#     "uvicorn==0.52.1",
# ]
# ///

import asyncio
import json
import logging
import os
import re
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Annotated

import click
import uvicorn
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import AfterValidator, BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

log = logging.getLogger("kcal")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

# Upper bound for the rolling-average window (~10 years). Without it, a huge
# value overflows date arithmetic and surfaces as a 500.
MAX_AVERAGE_DAYS = 3650


def _validate_date_str(value: str) -> str:
    if not _DATE_RE.match(value):
        raise ValueError("date must be in YYYY-MM-DD format")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError("date must be a real calendar date")
    return value


def _validate_time_str(value: str | None) -> str | None:
    if value is None:
        return None
    if not _TIME_RE.match(value):
        raise ValueError("time must be in HH:MM 24-hour format")
    return value


def _validate_description(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("description must not be blank")
    return stripped


def require_valid_date(value: str) -> str:
    """Validate a date taken from a URL path, raising HTTP 400 on failure."""
    try:
        return _validate_date_str(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


DateStr = Annotated[str, AfterValidator(_validate_date_str)]
TimeStr = Annotated[str | None, AfterValidator(_validate_time_str)]
DescriptionStr = Annotated[str, Field(max_length=500), AfterValidator(_validate_description)]


# ── Domain ────────────────────────────────────────────────────────────

# Macro estimates are best-effort: an entry is always saved and counted for kcal
# even when the LLM is unavailable or returns nonsense. Protein, fat and fiber
# come from one call, so a single state covers all three.
MACROS_PENDING = "pending"    # estimate in flight
MACROS_OK = "ok"              # estimate stored
MACROS_FAILED = "failed"      # LLM call failed; retryable
MACROS_SKIPPED = "skipped"    # estimator disabled (no API key), or a legacy row

# The macros tracked per entry, in display order. Adding one here carries it
# through the repository, the API and the UI without further plumbing.
MACRO_NAMES = ("protein", "fat", "fiber")


def totals_from_items(items: list[dict]) -> dict[str, float]:
    """Sum a per-item breakdown into one total per macro."""
    return {
        name: round(sum(item.get(f"{name}_g", 0) or 0 for item in items), 1)
        for name in MACRO_NAMES
    }


class KcalEntry:
    def __init__(
        self,
        kcal: int,
        description: str,
        entry_date: str,
        created_at: str | None = None,
        macros: dict[str, float] | None = None,
        macro_items: list[dict] | None = None,
        macros_state: str = MACROS_SKIPPED,
    ) -> None:
        self.kcal = kcal
        self.description = description
        self.entry_date = entry_date
        self.created_at = created_at or datetime.now().strftime("%H:%M")
        # Totals per macro name, e.g. {"protein": 29.8, "fat": 18.0, "fiber": 4.2}.
        # Empty when there is no estimate.
        self.macros = macros or {}
        self.macro_items = macro_items or []
        self.macros_state = macros_state
        self.id: int | None = None


# ── Repository ────────────────────────────────────────────────────────

class KcalRepository(ABC):
    @abstractmethod
    def add_entry(self, entry: KcalEntry) -> KcalEntry: ...

    @abstractmethod
    def delete_entry(self, entry_id: int) -> None: ...

    @abstractmethod
    def list_entries(self, entry_date: str) -> list[KcalEntry]: ...

    @abstractmethod
    def get_entry(self, entry_id: int) -> KcalEntry | None: ...

    @abstractmethod
    def set_macros(self, entry_id: int, items: list[dict] | None, state: str) -> bool: ...

    @abstractmethod
    def get_limit(self, entry_date: str) -> int | None: ...

    @abstractmethod
    def set_limit(self, entry_date: str, limit_kcal: int) -> None: ...

    @abstractmethod
    def get_burn(self, entry_date: str) -> int | None: ...

    @abstractmethod
    def set_burn(self, entry_date: str, burn_kcal: int) -> None: ...

    @abstractmethod
    def is_skipped(self, entry_date: str) -> bool: ...

    @abstractmethod
    def set_skipped(self, entry_date: str, skipped: bool) -> None: ...

    @abstractmethod
    def cumulative_weight_change(self) -> dict: ...

    @abstractmethod
    def average_intake(self, days: int) -> dict: ...


class SqliteKcalRepository(KcalRepository):
    def __init__(self, db_path: str = "kcal.db") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        # Protein estimates are written from background tasks, so writes can race
        # with request handlers on this single shared connection.
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  kcal INTEGER NOT NULL,"
            "  description TEXT NOT NULL,"
            "  entry_date TEXT NOT NULL,"
            "  created_at TEXT NOT NULL DEFAULT '00:00'"
            ")"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS daily_limits ("
            "  entry_date TEXT PRIMARY KEY,"
            "  limit_kcal INTEGER NOT NULL"
            ")"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS daily_burns ("
            "  entry_date TEXT PRIMARY KEY,"
            "  burn_kcal INTEGER NOT NULL"
            ")"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS skipped_days ("
            "  entry_date TEXT PRIMARY KEY"
            ")"
        )
        self._migrate_entries()
        self._conn.commit()

    def _migrate_entries(self) -> None:
        """Bring an existing database up to the current schema.

        CREATE TABLE IF NOT EXISTS never alters a table that already exists, so
        databases created before a column was introduced need an explicit ALTER.
        Existing rows land on the column default.
        """
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(entries)")}

        # The first version of this feature tracked protein alone, so its columns
        # were named for it. They now hold all three macros.
        renames = {"protein_items": "macro_items", "protein_state": "macros_state"}
        for old_name, new_name in renames.items():
            if old_name in existing and new_name not in existing:
                self._conn.execute(f"ALTER TABLE entries RENAME COLUMN {old_name} TO {new_name}")
                existing.discard(old_name)
                existing.add(new_name)

        wanted = {
            **{f"{name}_g": "REAL" for name in MACRO_NAMES},
            "macro_items": "TEXT",
            "macros_state": f"TEXT NOT NULL DEFAULT '{MACROS_SKIPPED}'",
        }
        for column, ddl in wanted.items():
            if column not in existing:
                self._conn.execute(f"ALTER TABLE entries ADD COLUMN {column} {ddl}")

    _MACRO_COLUMNS = tuple(f"{name}_g" for name in MACRO_NAMES)

    _SELECT_ENTRY = (
        "SELECT id, kcal, description, entry_date, created_at, "
        + ", ".join(_MACRO_COLUMNS)
        + ", macro_items, macros_state FROM entries"
    )

    @staticmethod
    def _row_to_entry(row: tuple) -> KcalEntry:
        row_id, kcal, description, d, created_at = row[:5]
        totals = row[5:5 + len(MACRO_NAMES)]
        macro_items, macros_state = row[5 + len(MACRO_NAMES):]
        try:
            items = json.loads(macro_items) if macro_items else []
        except ValueError:
            items = []
        macros = {name: value for name, value in zip(MACRO_NAMES, totals) if value is not None}
        e = KcalEntry(kcal, description, d, created_at, macros, items, macros_state)
        e.id = row_id
        return e

    def add_entry(self, entry: KcalEntry) -> KcalEntry:
        columns = ", ".join(self._MACRO_COLUMNS)
        placeholders = ", ".join("?" * len(MACRO_NAMES))
        with self._lock:
            cur = self._conn.execute(
                f"INSERT INTO entries (kcal, description, entry_date, created_at, "
                f"{columns}, macro_items, macros_state) "
                f"VALUES (?, ?, ?, ?, {placeholders}, ?, ?)",
                (
                    entry.kcal,
                    entry.description,
                    entry.entry_date,
                    entry.created_at,
                    *(entry.macros.get(name) for name in MACRO_NAMES),
                    json.dumps(entry.macro_items) if entry.macro_items else None,
                    entry.macros_state,
                ),
            )
            self._conn.commit()
        entry.id = cur.lastrowid
        return entry

    def delete_entry(self, entry_id: int) -> None:
        with self._lock:
            cur = self._conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(f"No entry with id {entry_id}")

    def list_entries(self, entry_date: str) -> list[KcalEntry]:
        rows = self._conn.execute(
            f"{self._SELECT_ENTRY} WHERE entry_date = ? ORDER BY id",
            (entry_date,),
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_entry(self, entry_id: int) -> KcalEntry | None:
        row = self._conn.execute(
            f"{self._SELECT_ENTRY} WHERE id = ?", (entry_id,)
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def set_macros(self, entry_id: int, items: list[dict] | None, state: str) -> bool:
        """Store a macro estimate. Returns False if the entry no longer exists.

        Totals are derived from the per-item breakdown so the two can never
        disagree. An entry can be deleted while its estimate is still in flight,
        which is a normal outcome rather than an error.
        """
        totals = totals_from_items(items) if items else {}
        assignments = ", ".join(f"{col} = ?" for col in self._MACRO_COLUMNS)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE entries SET {assignments}, macro_items = ?, macros_state = ? "
                f"WHERE id = ?",
                (
                    *(totals.get(name) for name in MACRO_NAMES),
                    json.dumps(items) if items else None,
                    state,
                    entry_id,
                ),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def get_limit(self, entry_date: str) -> int | None:
        row = self._conn.execute(
            "SELECT limit_kcal FROM daily_limits "
            "WHERE entry_date <= ? ORDER BY entry_date DESC LIMIT 1",
            (entry_date,),
        ).fetchone()
        return row[0] if row else None

    def set_limit(self, entry_date: str, limit_kcal: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO daily_limits (entry_date, limit_kcal) VALUES (?, ?) "
                "ON CONFLICT(entry_date) DO UPDATE SET limit_kcal = excluded.limit_kcal",
                (entry_date, limit_kcal),
            )
            self._conn.commit()

    def get_burn(self, entry_date: str) -> int | None:
        row = self._conn.execute(
            "SELECT burn_kcal FROM daily_burns "
            "WHERE entry_date <= ? ORDER BY entry_date DESC LIMIT 1",
            (entry_date,),
        ).fetchone()
        return row[0] if row else None

    def set_burn(self, entry_date: str, burn_kcal: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO daily_burns (entry_date, burn_kcal) VALUES (?, ?) "
                "ON CONFLICT(entry_date) DO UPDATE SET burn_kcal = excluded.burn_kcal",
                (entry_date, burn_kcal),
            )
            self._conn.commit()

    def is_skipped(self, entry_date: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM skipped_days WHERE entry_date = ?",
            (entry_date,),
        ).fetchone()
        return row is not None

    def set_skipped(self, entry_date: str, skipped: bool) -> None:
        with self._lock:
            if skipped:
                self._conn.execute(
                    "INSERT OR IGNORE INTO skipped_days (entry_date) VALUES (?)",
                    (entry_date,),
                )
            else:
                self._conn.execute(
                    "DELETE FROM skipped_days WHERE entry_date = ?",
                    (entry_date,),
                )
            self._conn.commit()

    def average_intake(self, days: int) -> dict:
        """Compute average daily kcal intake over the last N days (excluding today).

        Skipped (cheat) days count as 4000 kcal consumed.
        """
        from datetime import timedelta
        SKIPPED_DAY_KCAL = 4000
        today = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        skipped_rows = self._conn.execute(
            "SELECT entry_date FROM skipped_days "
            "WHERE entry_date >= ? AND entry_date < ?",
            (start_date, today),
        ).fetchall()
        skipped_set = {r[0] for r in skipped_rows}

        rows = self._conn.execute(
            "SELECT entry_date, SUM(kcal) FROM entries "
            "WHERE entry_date >= ? AND entry_date < ? "
            "GROUP BY entry_date ORDER BY entry_date",
            (start_date, today),
        ).fetchall()

        # Merge entry dates and skipped dates so skipped days with no entries are included
        entry_totals = {entry_date: total for entry_date, total in rows}
        all_dates = sorted(set(entry_totals.keys()) | skipped_set)

        day_totals = []
        for entry_date in all_dates:
            if entry_date in skipped_set:
                day_totals.append({"date": entry_date, "total": SKIPPED_DAY_KCAL, "skipped": True})
            else:
                day_totals.append({"date": entry_date, "total": entry_totals[entry_date], "skipped": False})

        counted = len(day_totals)
        avg = round(sum(d["total"] for d in day_totals) / counted, 1) if counted > 0 else 0

        return {
            "days_requested": days,
            "days_counted": counted,
            "average_kcal": avg,
            "days": day_totals,
        }

    def cumulative_weight_change(self) -> dict:
        """Compute cumulative weight change across all completed days.

        Skipped (cheat) days count as 4000 kcal consumed.
        """
        KCAL_PER_GRAM_FAT = 7.7
        SKIPPED_DAY_KCAL = 4000

        # Get all dates that have entries
        rows = self._conn.execute(
            "SELECT entry_date, SUM(kcal) FROM entries GROUP BY entry_date ORDER BY entry_date"
        ).fetchall()

        # Get all burn rates (sorted by date)
        burn_rows = self._conn.execute(
            "SELECT entry_date, burn_kcal FROM daily_burns ORDER BY entry_date"
        ).fetchall()

        # Get skipped days
        skipped_rows = self._conn.execute(
            "SELECT entry_date FROM skipped_days"
        ).fetchall()
        skipped_set = {r[0] for r in skipped_rows}

        def get_burn_for_date(date: str) -> int | None:
            """Replicate the <= lookup logic for burn rate."""
            result = None
            for bd, bk in burn_rows:
                if bd <= date:
                    result = bk
                else:
                    break
            return result

        today = datetime.now().strftime("%Y-%m-%d")

        # Merge entry dates and skipped dates so skipped days with no entries are included
        entry_totals = {entry_date: consumed for entry_date, consumed in rows}
        all_dates = sorted(set(entry_totals.keys()) | skipped_set)

        total_grams = 0.0
        day_details = []

        for entry_date in all_dates:
            if entry_date >= today:
                # Skip today and future days (not yet complete)
                continue

            if entry_date in skipped_set:
                consumed = SKIPPED_DAY_KCAL
            else:
                consumed = entry_totals.get(entry_date, 0)

            burn = get_burn_for_date(entry_date)
            if burn is None:
                continue

            deficit = burn - consumed
            grams = deficit / KCAL_PER_GRAM_FAT
            total_grams += grams
            day_details.append({
                "date": entry_date,
                "consumed": consumed,
                "burn": burn,
                "deficit": deficit,
                "grams": round(grams, 3),
                "skipped": entry_date in skipped_set,
            })

        return {
            "total_grams": round(total_grams, 3),
            "days_counted": len(day_details),
            "days": day_details,
        }


# ── Macro estimation ──────────────────────────────────────────────────

MACRO_MODEL = "google/gemini-3.5-flash-lite"
MACRO_TIMEOUT_S = 30
# Atwater factors: the energy each macronutrient supplies per gram.
KCAL_PER_G_PROTEIN = 4
KCAL_PER_G_FAT = 9
# Fiber is a subset of carbohydrate and yields ~2 kcal/g.
KCAL_PER_G_FIBER = 2
# Protein and fat together can account for nearly all of a meal's energy, so an
# estimate whose combined energy overshoots the stated calories is not physically
# possible. The slack absorbs rounding and the fact that the user's own kcal
# figure is itself a rough estimate.
MACRO_SLACK = 1.25

MACRO_INSTRUCTIONS = """\
You estimate the macronutrient content of meals for a calorie tracking app.

Given a short meal description and its approximate calorie count, break the meal
into its distinct food items and estimate the grams of protein, fat and fiber in
each one.

Rules:
- One entry per distinct food, in the order mentioned in the description.
- Echo the quantity in the name exactly as the user framed it: "3 eggs",
  "2 slices of protein bread", "a handful of almonds".
- Assume ordinary supermarket products and typical serving sizes when the
  description is vague. Never ask for clarification.
- Include items that contribute nothing (black coffee, water) with 0 grams so the
  breakdown accounts for the whole description.
- Only plant foods contain fiber; meat, fish, eggs and dairy have none.
- The calorie count is a hint about portion size; keep your estimate consistent
  with it. Protein supplies 4 kcal per gram and fat 9 kcal per gram, so protein
  and fat together can never exceed the meal's calories.
"""


class MacroItem(BaseModel):
    """A single food item within a meal."""

    name: str = Field(description="The food including its quantity, e.g. '3 eggs'")
    protein_g: float = Field(ge=0, description="Estimated grams of protein in this item")
    fat_g: float = Field(ge=0, description="Estimated grams of fat in this item")
    fiber_g: float = Field(ge=0, description="Estimated grams of dietary fiber in this item")


class MacroEstimate(BaseModel):
    """A per-item macronutrient breakdown of a meal."""

    items: list[MacroItem]


class MacroEstimator:
    """Estimates per-item macros for a meal description via a single LLM call.

    All three macros come from one call: they are a single judgement about what
    the meal contains, and splitting them would triple latency and cost for no
    gain in quality.

    Deliberately one-shot and failure-tolerant: callers treat any exception as
    "no estimate available" rather than an error worth surfacing.
    """

    def __init__(self, api_key: str, model_name: str = MACRO_MODEL) -> None:
        model = OpenRouterModel(model_name, provider=OpenRouterProvider(api_key=api_key))
        # Tool output (the default) rather than NativeOutput: OpenRouter does not
        # advertise native json_schema support for these models, and tool calls
        # are measurably faster here anyway.
        self._agent = Agent(
            model,
            output_type=MacroEstimate,
            instructions=MACRO_INSTRUCTIONS,
            retries=0,
        )

    async def estimate(self, description: str, kcal: int) -> list[dict]:
        result = await asyncio.wait_for(
            self._agent.run(f"Meal (~{kcal} kcal): {description}"),
            timeout=MACRO_TIMEOUT_S,
        )
        return self._sanitise(result.output.items, kcal)

    @staticmethod
    def _sanitise(items: list[MacroItem], kcal: int) -> list[dict]:
        """Round to 1dp and scale the breakdown down to fit the meal's energy."""
        clean = [
            {
                "name": item.name.strip(),
                "protein_g": max(0.0, item.protein_g),
                "fat_g": max(0.0, item.fat_g),
                "fiber_g": max(0.0, item.fiber_g),
            }
            for item in items
            if item.name.strip()
        ]

        # Protein and fat share the meal's energy budget, so they are scaled
        # together to preserve their ratio rather than clamped independently.
        energy = sum(i["protein_g"] * KCAL_PER_G_PROTEIN + i["fat_g"] * KCAL_PER_G_FAT for i in clean)
        ceiling = kcal * MACRO_SLACK
        if energy > ceiling > 0:
            log.warning(
                "Estimate implies %.0f kcal of protein+fat for a %d kcal meal; scaling down",
                energy, kcal,
            )
            scale = ceiling / energy
            for i in clean:
                i["protein_g"] *= scale
                i["fat_g"] *= scale

        # Fiber has its own, far looser bound; it only catches outright nonsense.
        fiber = sum(i["fiber_g"] for i in clean)
        fiber_ceiling = kcal / KCAL_PER_G_FIBER
        if fiber > fiber_ceiling > 0:
            log.warning("Fiber estimate %.1fg exceeds the %.1fg a %d kcal meal allows; scaling down",
                        fiber, fiber_ceiling, kcal)
            scale = fiber_ceiling / fiber
            for i in clean:
                i["fiber_g"] *= scale

        return [
            {"name": i["name"], **{f"{n}_g": round(i[f"{n}_g"], 1) for n in MACRO_NAMES}}
            for i in clean
        ]


estimator: MacroEstimator | None = None


def build_estimator() -> MacroEstimator | None:
    """Build the estimator, or return None so the app runs without an API key."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        log.warning("OPENROUTER_API_KEY not set — macro estimation disabled")
        return None
    try:
        return MacroEstimator(api_key)
    except Exception:
        log.exception("Could not build macro estimator — macro estimation disabled")
        return None


async def estimate_macros_task(entry_id: int, description: str, kcal: int) -> None:
    """Background task: estimate macros for an entry and store the result.

    Never raises. A failure leaves the entry marked 'failed', which the UI offers
    to retry; the entry's calories are unaffected either way.
    """
    if estimator is None:
        return
    try:
        items = await estimator.estimate(description, kcal)
    except Exception as exc:
        log.warning("Macro estimate failed for entry %s: %s", entry_id, exc)
        try:
            repo.set_macros(entry_id, None, MACROS_FAILED)
        except Exception:
            log.exception("Could not mark entry %s as failed", entry_id)
        return
    try:
        if not repo.set_macros(entry_id, items, MACROS_OK):
            log.info("Entry %s was deleted before its macro estimate arrived", entry_id)
    except Exception:
        log.exception("Could not store macro estimate for entry %s", entry_id)


# ── Schemas ───────────────────────────────────────────────────────────

class AddEntryRequest(BaseModel):
    kcal: int = Field(gt=0, le=100_000)
    description: DescriptionStr
    date: DateStr
    time: TimeStr = None

class SetLimitRequest(BaseModel):
    limit: int = Field(gt=0, le=100_000)
    date: DateStr

class SetBurnRequest(BaseModel):
    burn: int = Field(gt=0, le=100_000)
    date: DateStr

class SetSkippedRequest(BaseModel):
    skipped: bool
    date: DateStr

class MacroItemResponse(BaseModel):
    name: str
    # Null rather than zero when a macro is genuinely unknown, which is the case
    # for entries estimated before fat and fiber were tracked.
    protein_g: float | None = None
    fat_g: float | None = None
    fiber_g: float | None = None

class EntryResponse(BaseModel):
    id: int
    kcal: int
    description: str
    time: str
    # Totals per macro name; empty when this entry has no estimate.
    macros: dict[str, float] = {}
    macro_items: list[MacroItemResponse] = []
    macros_state: str = MACROS_SKIPPED

class DayResponse(BaseModel):
    date: str
    limit: int | None
    burn: int | None
    total: int
    total_macros: dict[str, float]
    # False when at least one entry has no estimate, so the UI can flag the
    # macro totals as partial figures.
    macros_complete: bool
    skipped: bool
    entries: list[EntryResponse]


# ── API ───────────────────────────────────────────────────────────────

api = APIRouter(prefix="/api")
repo: SqliteKcalRepository


def _entry_response(entry: KcalEntry) -> EntryResponse:
    return EntryResponse(
        id=entry.id,
        kcal=entry.kcal,
        description=entry.description,
        time=entry.created_at,
        macros=entry.macros,
        macro_items=[MacroItemResponse(**item) for item in entry.macro_items],
        macros_state=entry.macros_state,
    )


@api.get("/days/{day}", response_model=DayResponse)
async def get_day(day: str):
    day = require_valid_date(day)
    entries = repo.list_entries(day)
    limit = repo.get_limit(day)
    total = sum(e.kcal for e in entries)
    burn = repo.get_burn(day)
    skipped = repo.is_skipped(day)
    return DayResponse(
        date=day,
        limit=limit,
        burn=burn,
        total=total,
        # A macro is omitted entirely when no entry has a figure for it, so the
        # UI can say "unknown" rather than imply a real zero.
        total_macros={
            name: round(sum(e.macros.get(name, 0) for e in entries), 1)
            for name in MACRO_NAMES
            if any(name in e.macros for e in entries)
        },
        macros_complete=all(
            e.macros_state == MACROS_OK and all(name in e.macros for name in MACRO_NAMES)
            for e in entries
        ),
        skipped=skipped,
        entries=[_entry_response(e) for e in entries],
    )


@api.post("/entries", response_model=EntryResponse, status_code=201)
async def add_entry(body: AddEntryRequest, background: BackgroundTasks):
    # The entry is saved and returned immediately; the protein estimate lands a
    # second or two later so adding an entry stays instant.
    pending = estimator is not None
    entry = KcalEntry(
        body.kcal,
        body.description,
        body.date,
        created_at=body.time,
        macros_state=MACROS_PENDING if pending else MACROS_SKIPPED,
    )
    repo.add_entry(entry)
    if pending:
        background.add_task(estimate_macros_task, entry.id, entry.description, entry.kcal)
    return _entry_response(entry)


@api.post("/entries/{entry_id}/macros", response_model=EntryResponse, status_code=202)
async def retry_macros(entry_id: int, background: BackgroundTasks):
    entry = repo.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    if estimator is None:
        raise HTTPException(status_code=503, detail="Macro estimation is not configured")
    repo.set_macros(entry_id, None, MACROS_PENDING)
    entry.macros, entry.macro_items, entry.macros_state = {}, [], MACROS_PENDING
    background.add_task(estimate_macros_task, entry_id, entry.description, entry.kcal)
    return _entry_response(entry)


@api.delete("/entries/{entry_id}", status_code=204)
async def delete_entry(entry_id: int):
    try:
        repo.delete_entry(entry_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Entry not found")


@api.put("/limits", status_code=200)
async def set_limit(body: SetLimitRequest):
    repo.set_limit(body.date, body.limit)
    return {"date": body.date, "limit": body.limit}


@api.put("/burns", status_code=200)
async def set_burn(body: SetBurnRequest):
    repo.set_burn(body.date, body.burn)
    return {"date": body.date, "burn": body.burn}


@api.put("/skip", status_code=200)
async def set_skipped(body: SetSkippedRequest):
    repo.set_skipped(body.date, body.skipped)
    return {"date": body.date, "skipped": body.skipped}


@api.get("/cumulative")
async def get_cumulative():
    return repo.cumulative_weight_change()


@api.get("/average/{days}")
async def get_average(days: int):
    if days < 1:
        raise HTTPException(status_code=400, detail="Days must be >= 1")
    if days > MAX_AVERAGE_DAYS:
        raise HTTPException(status_code=400, detail=f"Days must be <= {MAX_AVERAGE_DAYS}")
    return repo.average_intake(days)


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI()
app.include_router(api)

HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>KCAL</title>

  <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.3.3/dist/index.global.js"></script>

  <style type="text/tailwindcss">
    @theme inline {
      --font-sans: "IBM Plex Mono", "SF Mono", "Fira Code", ui-monospace, monospace;
      --font-mono: "IBM Plex Mono", "SF Mono", "Fira Code", ui-monospace, monospace;
      --color-background: var(--background);
      --color-foreground: var(--foreground);
      --color-primary: var(--primary);
      --color-primary-foreground: var(--primary-foreground);
      --color-secondary: var(--secondary);
      --color-secondary-foreground: var(--secondary-foreground);
      --color-muted: var(--muted);
      --color-muted-foreground: var(--muted-foreground);
      --color-accent: var(--accent);
      --color-accent-foreground: var(--accent-foreground);
      --color-destructive: var(--destructive);
      --color-card: var(--card);
      --color-card-foreground: var(--card-foreground);
      --color-border: var(--border);
      --color-input: var(--input);
      --color-ring: var(--ring);
      --radius-sm: 0px;
    }
    input[type="number"]::-webkit-inner-spin-button,
    input[type="number"]::-webkit-outer-spin-button {
      -webkit-appearance: none;
      margin: 0;
    }
    input[type="number"] {
      -moz-appearance: textfield;
      --radius-md: 0px;
      --radius-lg: 0px;
      --radius-xl: 0px;
      --radius-2xl: 0px;
    }
    :root {
      --background: #ffffff;
      --foreground: #0a0a0a;
      --primary: #0a0a0a;
      --primary-foreground: #ffffff;
      --secondary: #f0f0f0;
      --secondary-foreground: #0a0a0a;
      --muted: #f5f5f5;
      --muted-foreground: #737373;
      --accent: #f0f0f0;
      --accent-foreground: #0a0a0a;
      --destructive: #dc2626;
      --card: #ffffff;
      --card-foreground: #0a0a0a;
      --border: #0a0a0a;
      --input: #0a0a0a;
      --ring: #0a0a0a;
      --radius: 0px;
    }
    @layer base {
      * { @apply border-border; }
      body { @apply bg-background text-foreground; }
    }
  </style>

  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />

  <script type="importmap">
  {
    "imports": {
      "react": "https://esm.sh/react@19.2.8",
      "react/jsx-runtime": "https://esm.sh/react@19.2.8/jsx-runtime",
      "react/jsx-dev-runtime": "https://esm.sh/react@19.2.8/jsx-dev-runtime",
      "react-dom": "https://esm.sh/react-dom@19.2.8?deps=react@19.2.8",
      "react-dom/client": "https://esm.sh/react-dom@19.2.8/client?deps=react@19.2.8",
      "shadcn": "https://esm.sh/shadcn-ui-bundled@0.1.0/standalone?deps=react@19.2.8,react-dom@19.2.8",
      "@tanstack/react-query": "https://esm.sh/@tanstack/react-query@5.101.4?deps=react@19.2.8",
      "react-error-boundary": "https://esm.sh/react-error-boundary@6.1.2?deps=react@19.2.8",
      "ky": "https://esm.sh/ky@2.0.2",
      "react-hook-form": "https://esm.sh/react-hook-form@7.84.0?deps=react@19.2.8"
    }
  }
  </script>

  <script src="https://unpkg.com/@babel/standalone@7.29.8/babel.min.js"></script>
  <script>
    Babel.registerPreset("tsx-auto", {
      presets: [
        [Babel.availablePresets["react"], { runtime: "automatic" }],
        [
          Babel.availablePresets["typescript"],
          {
            // Required so the parser runs in TSX mode (JSX + TS).
            isTSX: true,
            allExtensions: true
          }
        ],
      ],
    });
  </script>
</head>
<body>
  <div id="root"></div>

  <script type="text/babel" data-type="module" data-presets="tsx-auto">
    import { Suspense, useState, useEffect, useCallback } from "react";
    import { createRoot } from "react-dom/client";
    import {
      QueryClient,
      QueryClientProvider,
      useSuspenseQuery,
      useMutation,
      useQueryClient,
    } from "@tanstack/react-query";
    import { ErrorBoundary } from "react-error-boundary";
    import ky, { HTTPError } from "ky";
    import { useForm } from "react-hook-form";
    import {
      Button,
      Input,
      Separator,
      Spinner,
      Alert, AlertDescription,
    } from "shadcn";

    // ── Types ────────────────────────────────────────────────────

    type MacrosState = "pending" | "ok" | "failed" | "skipped";

    type Macros = Record<string, number>;

    interface MacroItem {
      name: string;
      protein_g: number;
      fat_g: number;
      fiber_g: number;
    }

    interface Entry {
      id: number;
      kcal: number;
      description: string;
      time: string;
      macros: Macros;
      macro_items: MacroItem[];
      macros_state: MacrosState;
    }

    interface DayData {
      date: string;
      limit: number | null;
      burn: number | null;
      total: number;
      total_macros: Macros;
      macros_complete: boolean;
      skipped: boolean;
      entries: Entry[];
    }

    // Display order and short labels. "FAT" and "FIB" are spelled out rather than
    // both reduced to "F", which would be ambiguous.
    const MACROS = [
      { key: "protein", short: "P", label: "PROTEIN" },
      { key: "fat", short: "FAT", label: "FAT" },
      { key: "fiber", short: "FIB", label: "FIBER" },
    ] as const;

    // Aligns a row's sub-lines under its description: time (w-11) + gap-3 +
    // kcal (w-12) + gap-3.
    const MACRO_INDENT = "ml-[7.25rem]";

    // ── Query keys ───────────────────────────────────────────────

    const dayKeys = {
      day: (date: string) => ["day", date] as const,
    } as const;

    const statsKeys = {
      average: ["average"] as const,
      cumulative: ["cumulative"] as const,
    } as const;

    // Anything that changes stored kcal also changes the average and cumulative
    // cards, so they must be invalidated together.
    function useInvalidateDayAndStats(date: string) {
      const queryClient = useQueryClient();
      return useCallback(() => {
        queryClient.invalidateQueries({ queryKey: dayKeys.day(date) });
        queryClient.invalidateQueries({ queryKey: statsKeys.average });
        queryClient.invalidateQueries({ queryKey: statsKeys.cumulative });
      }, [queryClient, date]);
    }

    // ── API client ──────────────────────────────────────────────

    const api = ky.create({ prefix: "/api" });

    const kcalClient = {
      getDay: (date: string) => api.get(`days/${date}`).json<DayData>(),
      addEntry: (data: { kcal: number; description: string; date: string; time: string }) =>
        api.post("entries", { json: data }).json<Entry>(),
      deleteEntry: (id: number) => api.delete(`entries/${id}`),
      retryMacros: (id: number) => api.post(`entries/${id}/macros`).json<Entry>(),
      setLimit: (data: { limit: number; date: string }) =>
        api.put("limits", { json: data }).json<{ date: string; limit: number }>(),
      setBurn: (data: { burn: number; date: string }) =>
        api.put("burns", { json: data }).json<{ date: string; burn: number }>(),
      setSkipped: (data: { skipped: boolean; date: string }) =>
        api.put("skip", { json: data }).json<{ date: string; skipped: boolean }>(),
      getCumulative: () =>
        api.get("cumulative").json<{ total_grams: number; days_counted: number }>(),
      getAverage: (days: number) =>
        api.get(`average/${days}`).json<{ days_requested: number; days_counted: number; average_kcal: number }>(),
    } as const;

    // ── Helpers ──────────────────────────────────────────────────

    // Whole grams read cleaner than "18.0"; a decimal only earns its place when
    // it carries information.
    // Carries its own unit so an unknown macro reads "–" rather than "–g".
    function fmtGrams(grams: number | null | undefined): string {
      if (grams == null) return "–";
      return (Number.isInteger(grams) ? String(grams) : grams.toFixed(1)) + "g";
    }

    // Poll while any estimate is still in flight. Self-limiting: every pending
    // entry ends up ok or failed, so this always settles back to no polling.
    function dayRefetchInterval(query: { state: { data?: DayData } }): number | false {
      const data = query.state.data;
      if (!data) return false;
      return data.entries.some((e) => e.macros_state === "pending") ? 1500 : false;
    }

    // toISOString() is UTC, which picks the wrong day either side of midnight.
    // Every date in this app is a *local* calendar date.
    function toDateStr(d: Date): string {
      const month = String(d.getMonth() + 1).padStart(2, "0");
      const day = String(d.getDate()).padStart(2, "0");
      return `${d.getFullYear()}-${month}-${day}`;
    }

    function todayStr(): string {
      return toDateStr(new Date());
    }

    function shiftDate(dateStr: string, days: number): string {
      const d = new Date(dateStr + "T12:00:00");
      d.setDate(d.getDate() + days);
      return toDateStr(d);
    }

    function formatDate(dateStr: string): string {
      const d = new Date(dateStr + "T12:00:00");
      const today = todayStr();
      const yesterday = shiftDate(today, -1);
      const tomorrow = shiftDate(today, 1);
      if (dateStr === today) return "TODAY";
      if (dateStr === yesterday) return "YESTERDAY";
      if (dateStr === tomorrow) return "TOMORROW";
      return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" }).toUpperCase();
    }

    // ── Burn rate helpers ────────────────────────────────────────

    const KCAL_PER_GRAM_FAT = 7.7;
    // Must match the backend: a cheat day is scored as this many kcal.
    const SKIPPED_DAY_KCAL = 4000;
    // Must match the backend bound on /api/average/{days}.
    const MAX_AVERAGE_DAYS = 3650;

    function secondsSinceMidnight(): number {
      const now = new Date();
      return now.getHours() * 3600 + now.getMinutes() * 60 + now.getSeconds();
    }

    function useLiveBurn(burnRate: number | null, consumed: number, isToday: boolean) {
      const [now, setNow] = useState(Date.now());

      useEffect(() => {
        if (!isToday || burnRate === null) return;
        const id = setInterval(() => setNow(Date.now()), 1000);
        return () => clearInterval(id);
      }, [isToday, burnRate]);

      if (burnRate === null) return null;

      const elapsed = isToday ? secondsSinceMidnight() : 86400;
      const burnedSoFar = (burnRate / 86400) * elapsed;
      const deficit = burnedSoFar - consumed;
      const grams = deficit / KCAL_PER_GRAM_FAT;

      return { burnedSoFar, deficit, grams };
    }

    // ── Error handling ────────────────────────────────────────────

    // FastAPI returns `detail` as a string for HTTPException but as an array of
    // objects for 422 validation errors. Rendering the array directly crashes React.
    function formatDetail(detail: any, fallback: string): string {
      if (typeof detail === "string") return detail;
      if (Array.isArray(detail)) {
        const parts = detail
          .map((d) => {
            const field = Array.isArray(d?.loc) ? d.loc[d.loc.length - 1] : null;
            const msg = String(d?.msg ?? "").replace(/^Value error, /, "");
            return field ? `${field}: ${msg}` : msg;
          })
          .filter(Boolean);
        if (parts.length > 0) return parts.join("; ");
      }
      return fallback;
    }

    function useErrorMessage(error: Error | null): string | null {
      const [message, setMessage] = useState<string | null>(null);

      useEffect(() => {
        if (!error) { setMessage(null); return; }
        if (error instanceof HTTPError) {
          error.response
            .json()
            .then((body: any) => setMessage(formatDetail(body?.detail, error.message)))
            .catch(() => setMessage(error.message));
        } else {
          setMessage(error.message);
        }
      }, [error]);

      return message;
    }

    function AppErrorMessage({ error, retry }: { error: Error | null; retry?: () => void }) {
      const message = useErrorMessage(error);
      if (!message) return null;

      if (retry) {
        return (
          <div className="border-2 border-destructive p-3 my-3 flex items-center justify-between">
            <span className="text-sm text-destructive">{message}</span>
            <button
              onClick={retry}
              className="text-xs border-2 border-destructive text-destructive px-2 py-1 hover:bg-destructive hover:text-white transition-colors"
            >
              RETRY
            </button>
          </div>
        );
      }

      return <p className="text-sm text-destructive mt-1">{message}</p>;
    }

    // ── Components ───────────────────────────────────────────────

    type Status = "ok" | "warning" | "danger" | "critical" | "over";

    function getStatus(total: number, limit: number | null): Status {
      if (limit === null) return "ok";
      if (total > limit) return "over";
      const remaining = limit - total;
      const pct = remaining / limit;
      if (pct <= 0.10) return "critical";
      if (pct <= 0.20) return "danger";
      if (pct <= 0.30) return "warning";
      return "ok";
    }

    const statusBarColor: Record<Status, string> = {
      ok: "bg-foreground",
      warning: "bg-yellow-500",
      danger: "bg-orange-500",
      critical: "bg-red-500",
      over: "bg-red-600",
    };

    const statusTextColor: Record<Status, string> = {
      ok: "",
      warning: "text-yellow-600",
      danger: "text-orange-500",
      critical: "text-red-500",
      over: "text-red-600",
    };

    function ProgressBar({ total, limit }: { total: number; limit: number | null }) {
      if (limit === null) return null;
      const pct = Math.min((total / limit) * 100, 100);
      const over = total > limit;
      const remaining = limit - total;
      const status = getStatus(total, limit);

      return (
        <div className="space-y-2">
          <div className="h-2 w-full bg-muted border border-foreground">
            <div
              className={`h-full transition-all duration-300 ${statusBarColor[status]}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className={`flex justify-between text-xs tracking-wider ${statusTextColor[status]}`}>
            <span>{total} KCAL CONSUMED</span>
            <span className={over ? "font-bold" : ""}>
              {over ? `⚠️ ${Math.abs(remaining)} OVER ⚠️` : `${remaining} LEFT`}
            </span>
          </div>
        </div>
      );
    }

    function InlineSetter({ label, current, placeholder, onSave, isPending, error }: {
      label: string;
      current: number | null;
      placeholder: string;
      onSave: (val: number) => Promise<unknown>;
      isPending: boolean;
      error: Error | null;
    }) {
      const [editing, setEditing] = useState(false);
      const { register, handleSubmit, reset } = useForm<{ value: string }>({
        defaultValues: { value: current?.toString() ?? "" },
      });

      // Stay open when the save fails, otherwise the error is never seen.
      const onSubmit = async ({ value }: { value: string }) => {
        const parsed = parseInt(value);
        if (!Number.isFinite(parsed) || parsed <= 0) return;
        try {
          await onSave(parsed);
          setEditing(false);
        } catch {
          /* error is rendered by AppErrorMessage below */
        }
      };

      useEffect(() => {
        reset({ value: current?.toString() ?? "" });
      }, [current, reset]);

      if (!editing) {
        return (
          <button
            onClick={() => setEditing(true)}
            className="text-xs tracking-wider text-muted-foreground hover:text-foreground transition-colors border-b border-dashed border-muted-foreground hover:border-foreground"
          >
            {current !== null ? `${label}: ${current} KCAL` : `SET ${label}`}
          </button>
        );
      }

      return (
        <form
          onSubmit={handleSubmit(onSubmit)}
          className="flex items-center gap-2"
        >
          <input
            type="number"
            autoFocus
            {...register("value", { required: true, min: 1 })}
            className="w-20 text-xs border-2 border-foreground px-2 py-1 bg-transparent font-mono focus:outline-none"
            placeholder={placeholder}
          />
          <span className="text-xs tracking-wider">KCAL</span>
          <button
            type="submit"
            disabled={isPending}
            className="text-xs border-2 border-foreground px-2 py-1 hover:bg-foreground hover:text-background transition-colors disabled:opacity-50"
          >
            {isPending ? "..." : "SET"}
          </button>
          <button
            type="button"
            onClick={() => setEditing(false)}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            ✕
          </button>
          <AppErrorMessage error={error} />
        </form>
      );
    }

    function LimitSetter({ currentLimit, date }: { currentLimit: number | null; date: string }) {
      const queryClient = useQueryClient();
      const mutation = useMutation({
        mutationFn: (limit: number) => kcalClient.setLimit({ limit, date }),
        onSuccess: () => queryClient.invalidateQueries({ queryKey: dayKeys.day(date) }),
      });
      return (
        <InlineSetter
          label="LIMIT"
          current={currentLimit}
          placeholder="1700"
          onSave={(v) => mutation.mutateAsync(v)}
          isPending={mutation.isPending}
          error={mutation.error}
        />
      );
    }

    function BurnSetter({ currentBurn, date }: { currentBurn: number | null; date: string }) {
      // Burn rate feeds the cumulative weight calculation, not just this day.
      const invalidate = useInvalidateDayAndStats(date);
      const mutation = useMutation({
        mutationFn: (burn: number) => kcalClient.setBurn({ burn, date }),
        onSuccess: invalidate,
      });
      return (
        <InlineSetter
          label="BURN"
          current={currentBurn}
          placeholder="2200"
          onSave={(v) => mutation.mutateAsync(v)}
          isPending={mutation.isPending}
          error={mutation.error}
        />
      );
    }

    function formatGrams(g: number): string {
      const abs = Math.abs(g);
      if (abs >= 1000) return (abs / 1000).toFixed(2) + "kg";
      return abs.toFixed(abs < 10 ? 3 : 1) + "g";
    }

    // Below this the forecast rounds to 0.000g, so it is neither a loss nor a gain.
    const GRAM_EPSILON = 0.0005;

    function gramsTrend(g: number): "losing" | "gaining" | "neutral" {
      if (g > GRAM_EPSILON) return "losing";
      if (g < -GRAM_EPSILON) return "gaining";
      return "neutral";
    }

    function ForecastRow({ label, grams }: { label: string; grams: number }) {
      const trend = gramsTrend(grams);
      const colorClass =
        trend === "losing" ? "text-emerald-600"
        : trend === "gaining" ? "text-red-500"
        : "text-muted-foreground";
      const arrow = trend === "losing" ? "↓" : trend === "gaining" ? "↑" : "";

      return (
        <div className="flex items-baseline justify-end gap-2">
          <span className="text-[10px] tracking-wider text-muted-foreground">{label}</span>
          <span className={`text-sm font-bold tabular-nums ${colorClass}`}>
            {arrow}{formatGrams(grams)}
          </span>
        </div>
      );
    }

    function LiveBurnCounter({ burnRate, consumed, limit, isToday }: { burnRate: number | null; consumed: number; limit: number | null; isToday: boolean }) {
      const burn = useLiveBurn(burnRate, consumed, isToday);
      if (!burn) return null;

      const losing = burn.grams > 0;
      const gaining = burn.grams < 0;
      const colorClass = losing ? "text-emerald-600" : gaining ? "text-red-500" : "";

      // Scenario A — you eat nothing else today. Independent of the limit.
      const stopTodayGrams = (burnRate! - consumed) / KCAL_PER_GRAM_FAT;

      // Scenario B — you eat up to the limit. You cannot un-eat what is already
      // logged, so a day already past its limit lands on the actual intake.
      const limitDayIntake = limit !== null ? Math.max(consumed, limit) : consumed;
      const limitTodayGrams = (burnRate! - limitDayIntake) / KCAL_PER_GRAM_FAT;
      const alreadyOverLimit = limit !== null && consumed > limit;

      // Future days start empty, so they use the limit itself. With no limit set
      // the only honest projection is "every day like today".
      const futureDayGrams =
        limit !== null ? (burnRate! - limit) / KCAL_PER_GRAM_FAT : stopTodayGrams;
      const baseTodayGrams = limit !== null ? limitTodayGrams : stopTodayGrams;
      const weekGrams = baseTodayGrams + futureDayGrams * 6;
      const monthGrams = baseTodayGrams + futureDayGrams * 29;

      return (
        <div className="border-2 border-foreground p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs tracking-wider text-muted-foreground">
              {isToday ? "LIVE" : "FINAL"} WEIGHT CHANGE
            </span>
            <span className="text-xs tracking-wider text-muted-foreground tabular-nums">
              {Math.round(burn.burnedSoFar)} BURNED
            </span>
          </div>

          <div className="flex items-start justify-between gap-4">
            {/* Left: live counter */}
            <div>
              <div className={`text-3xl font-bold tabular-nums tracking-tight ${colorClass}`}>
                {losing ? "↓" : gaining ? "↑" : ""} {Math.abs(burn.grams).toFixed(3)}g
              </div>
              <div className={`text-xs tracking-wider mt-1 ${colorClass || "text-muted-foreground"}`}>
                {losing ? "LOSING" : gaining ? "GAINING" : "NEUTRAL"}
              </div>
              <div className="text-[10px] tabular-nums text-muted-foreground mt-1">
                DEFICIT {burn.deficit >= 0 ? "+" : ""}{Math.round(burn.deficit)} KCAL
              </div>
            </div>

            {/* Right: forecasts. Each block states its own assumption, so the
                numbers are never a mix of two different scenarios. */}
            {isToday && (
              <div className="text-right space-y-3 border-l border-muted pl-4">
                <div className="space-y-1.5">
                  <div className="text-[10px] tracking-wider text-muted-foreground">
                    IF YOU STOP EATING NOW
                  </div>
                  <ForecastRow label="TODAY" grams={stopTodayGrams} />
                </div>

                {limit !== null ? (
                  <div className="space-y-1.5">
                    <div className="text-[10px] tracking-wider text-muted-foreground">
                      IF YOU EAT {limit} KCAL/DAY
                    </div>
                    <ForecastRow label="TODAY" grams={limitTodayGrams} />
                    <ForecastRow label="7 DAYS" grams={weekGrams} />
                    <ForecastRow label="30 DAYS" grams={monthGrams} />
                    {alreadyOverLimit && (
                      <div className="text-[10px] tracking-wider text-muted-foreground">
                        TODAY ALREADY OVER LIMIT
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-1.5">
                    <div className="text-[10px] tracking-wider text-muted-foreground">
                      IF EVERY DAY LIKE TODAY
                    </div>
                    <ForecastRow label="7 DAYS" grams={weekGrams} />
                    <ForecastRow label="30 DAYS" grams={monthGrams} />
                    <div className="text-[10px] tracking-wider text-muted-foreground">
                      SET A LIMIT FOR A REAL FORECAST
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      );
    }

    interface AddEntryFields {
      kcal: string;
      description: string;
    }

    function AddEntryForm({ date }: { date: string }) {
      const { register, handleSubmit, reset, formState: { isValid } } = useForm<AddEntryFields>({
        defaultValues: { kcal: "", description: "" },
      });
      const invalidate = useInvalidateDayAndStats(date);

      const mutation = useMutation({
        mutationFn: (data: { kcal: number; description: string; date: string; time: string }) =>
          kcalClient.addEntry(data),
        onSuccess: () => {
          invalidate();
          reset();
        },
      });

      const onSubmit = ({ kcal, description }: AddEntryFields) => {
        const now = new Date();
        const time = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
        mutation.mutate({ kcal: parseInt(kcal), description: description.trim(), date, time });
      };

      return (
        <div>
          <form onSubmit={handleSubmit(onSubmit)} className="flex gap-0">
            <input
              type="number"
              placeholder="kcal"
              {...register("kcal", { required: true, min: 1 })}
              disabled={mutation.isPending}
              className="w-20 border-2 border-foreground px-3 py-2.5 text-sm bg-transparent font-mono focus:outline-none placeholder:text-muted-foreground"
            />
            <input
              type="text"
              placeholder="description"
              {...register("description", { required: true, validate: (v) => v.trim().length > 0 })}
              disabled={mutation.isPending}
              className="flex-1 border-2 border-l-0 border-foreground px-3 py-2.5 text-sm bg-transparent font-mono focus:outline-none placeholder:text-muted-foreground"
            />
            <button
              type="submit"
              disabled={mutation.isPending || !isValid}
              className="border-2 border-l-0 border-foreground px-4 py-2.5 text-sm font-bold bg-foreground text-background hover:bg-transparent hover:text-foreground transition-colors disabled:opacity-30"
            >
              {mutation.isPending ? "..." : "+"}
            </button>
          </form>
          <AppErrorMessage error={mutation.error} />
        </div>
      );
    }

    // Macros are a best-effort estimate, so their slot in a row stays quiet: the
    // figures when we have them, a retry affordance when we don't, nothing at all
    // when estimation is switched off.
    function MacroBadge({ entry, date }: { entry: Entry; date: string }) {
      const invalidate = useInvalidateDayAndStats(date);

      const retry = useMutation({
        mutationFn: () => kcalClient.retryMacros(entry.id),
        onSuccess: invalidate,
      });

      if (entry.macros_state === "skipped") return null;

      if (entry.macros_state === "pending" || retry.isPending) {
        return (
          <span className="text-[11px] text-muted-foreground tabular-nums animate-pulse" title="Estimating macros">
            ···
          </span>
        );
      }

      if (entry.macros_state === "failed") {
        return (
          <button
            onClick={() => retry.mutate()}
            title="Macro estimate unavailable — click to retry"
            className="text-[11px] text-muted-foreground/60 hover:text-foreground transition-colors"
          >
            ↻
          </button>
        );
      }

      // An entry estimated before fat and fiber were tracked has no figure for
      // them. Rather than invent a zero, show a dash and let a click fill it in.
      const incomplete = MACROS.some((m) => entry.macros[m.key] == null);

      // Protein leads because it is the figure most often being watched; fat and
      // fiber stay muted so the line still scans at a glance.
      return (
        <span
          onClick={incomplete ? () => retry.mutate() : undefined}
          title={incomplete ? "Estimated before fat and fiber were tracked — click to re-estimate" : undefined}
          className={`text-[11px] tabular-nums text-muted-foreground ${incomplete ? "cursor-pointer hover:text-foreground" : ""}`}
        >
          {MACROS.map((m, i) => (
            <span key={m.key}>
              {i > 0 && <span className="mx-1 text-muted-foreground/40">·</span>}
              <span className={i === 0 ? "font-bold text-foreground" : ""}>
                {fmtGrams(entry.macros[m.key])}
              </span>{" "}
              {m.short}
            </span>
          ))}
          {incomplete && <span className="ml-1">↻</span>}
        </span>
      );
    }

    function MacroTotals({ data }: { data: DayData }) {
      // Nothing to say before the first entry, or when estimation is switched off.
      const tracked = data.entries.some((e) => e.macros_state !== "skipped");
      if (!tracked) return null;

      const partial = !data.macros_complete;
      return (
        <div
          className="text-xs tracking-wider mt-1 text-muted-foreground tabular-nums"
          title={partial ? "Some entries have no macro estimate yet" : undefined}
        >
          {MACROS.map((m, i) => (
            <span key={m.key}>
              {i > 0 && <span className="mx-1 text-muted-foreground/40">·</span>}
              <span className="font-bold text-foreground">{fmtGrams(data.total_macros[m.key])}</span> {m.label}
            </span>
          ))}
          {partial && "*"}
        </div>
      );
    }

    function EntryItem({ entry, date }: { entry: Entry; date: string }) {
      const invalidate = useInvalidateDayAndStats(date);
      const [expanded, setExpanded] = useState(false);

      const mutation = useMutation({
        mutationFn: () => kcalClient.deleteEntry(entry.id),
        onSuccess: invalidate,
      });

      const items = entry.macro_items;
      const canExpand = entry.macros_state === "ok" && items.length > 0;

      return (
        <div className="group py-3 border-b border-muted last:border-b-0">
          <div className="flex items-center justify-between gap-3">
            <div
              onClick={canExpand ? () => setExpanded((v) => !v) : undefined}
              className={`flex items-baseline gap-3 min-w-0 ${canExpand ? "cursor-pointer" : ""}`}
            >
              <span className="text-[11px] tabular-nums text-muted-foreground w-11 shrink-0">{entry.time}</span>
              <span className="text-sm font-bold tabular-nums w-12 text-right shrink-0">{entry.kcal}</span>
              <span className="text-sm break-words">{entry.description}</span>
              {canExpand && (
                <span className="text-[10px] text-muted-foreground shrink-0">{expanded ? "▾" : "▸"}</span>
              )}
            </div>
            <button
              onClick={() => mutation.mutate()}
              disabled={mutation.isPending}
              className="text-xs text-muted-foreground hover:text-destructive transition-colors disabled:opacity-50 shrink-0"
            >
              {mutation.isPending ? "..." : "DEL"}
            </button>
          </div>

          {/* Macros sit on their own line: three figures alongside the description
              would squeeze it onto several lines. MACRO_INDENT aligns them under it. */}
          <div className={`${MACRO_INDENT} empty:hidden`}>
            <MacroBadge entry={entry} date={date} />
          </div>

          {canExpand && expanded && (
            <div className="mt-2 ml-4 pl-3 border-l-2 border-muted text-[11px] text-muted-foreground">
              <div className="flex gap-2 pb-1 text-muted-foreground/60 tracking-wider">
                <span className="flex-1" />
                {MACROS.map((m) => (
                  <span key={m.key} className="w-12 text-right shrink-0">{m.short}</span>
                ))}
              </div>
              {items.map((item, i) => (
                <div key={i} className="flex gap-2 py-0.5">
                  <span className="flex-1 break-words">{item.name}</span>
                  {MACROS.map((m) => (
                    <span key={m.key} className="w-12 text-right tabular-nums shrink-0">
                      {fmtGrams(item[`${m.key}_g`])}
                    </span>
                  ))}
                </div>
              ))}
              <div className="flex gap-2 py-0.5 border-t border-muted mt-1 pt-1 text-foreground">
                <span className="flex-1 tracking-wider">TOTAL</span>
                {MACROS.map((m) => (
                  <span key={m.key} className="w-12 text-right tabular-nums font-bold shrink-0">
                    {fmtGrams(entry.macros[m.key])}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      );
    }

    function CumulativeWeightChange() {
      const { data } = useSuspenseQuery({
        queryKey: statsKeys.cumulative,
        queryFn: () => kcalClient.getCumulative(),
        refetchInterval: 60000,
      });

      if (data.days_counted === 0) return null;

      const losing = data.total_grams > 0;
      const gaining = data.total_grams < 0;
      const colorClass = losing ? "text-emerald-600" : gaining ? "text-red-500" : "text-muted-foreground";

      const absGrams = Math.abs(data.total_grams);
      let display: string;
      if (absGrams >= 1000) {
        display = (absGrams / 1000).toFixed(2) + "kg";
      } else {
        display = absGrams.toFixed(1) + "g";
      }

      return (
        <div className="border-2 border-dashed border-foreground px-4 py-3 flex items-center justify-between">
          <div>
            <div className="text-[10px] tracking-wider text-muted-foreground">NET WEIGHT CHANGE SINCE START</div>
            <div className="text-[10px] tracking-wider text-muted-foreground">{data.days_counted} DAYS COUNTED</div>
          </div>
          <div className={`text-2xl font-bold tabular-nums tracking-tight ${colorClass}`}>
            {losing ? "↓" : gaining ? "↑" : ""} {display}
          </div>
        </div>
      );
    }

    function AverageIntake() {
      const presets = [7, 14, 30];
      const [selectedDays, setSelectedDays] = useState(7);
      const [customInput, setCustomInput] = useState("");
      const [isCustom, setIsCustom] = useState(false);

      const { data } = useSuspenseQuery({
        queryKey: [...statsKeys.average, selectedDays],
        queryFn: () => kcalClient.getAverage(selectedDays),
        refetchInterval: 60000,
      });

      // Never hide the range controls: an empty 7-day window must still allow
      // switching to 14/30/custom, where older data may exist.

      return (
        <div className="px-4 py-3 space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-[10px] tracking-wider text-muted-foreground">AVG DAILY INTAKE</div>
            <div className="text-2xl font-bold tabular-nums tracking-tight">
              {data.days_counted > 0 ? `${Math.round(data.average_kcal)} KCAL` : "—"}
            </div>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            {presets.map((d) => (
              <button
                key={d}
                onClick={() => { setSelectedDays(d); setIsCustom(false); setCustomInput(""); }}
                className={`text-[10px] tracking-wider px-2.5 py-1 border-2 transition-colors ${
                  selectedDays === d && !isCustom
                    ? "border-foreground bg-foreground text-background"
                    : "border-foreground text-foreground hover:bg-foreground hover:text-background"
                }`}
              >
                {d}D
              </button>
            ))}

            <form
              onSubmit={(e) => {
                e.preventDefault();
                const val = parseInt(customInput);
                if (val > 0) { setSelectedDays(Math.min(val, MAX_AVERAGE_DAYS)); setIsCustom(true); }
              }}
              className="flex items-center gap-1 ml-auto"
            >
              <input
                type="number"
                value={customInput}
                onChange={(e) => setCustomInput(e.target.value)}
                placeholder="N"
                min={1}
                max={MAX_AVERAGE_DAYS}
                className="w-12 text-[10px] border-2 border-foreground px-1.5 py-1 bg-transparent font-mono focus:outline-none placeholder:text-muted-foreground text-center"
              />
              <button
                type="submit"
                className="text-[10px] tracking-wider border-2 border-foreground px-2 py-1 hover:bg-foreground hover:text-background transition-colors"
              >
                GO
              </button>
            </form>
          </div>

          <div className="text-[10px] tracking-wider text-muted-foreground">
            {data.days_counted} OF {data.days_requested} DAYS WITH DATA
          </div>
        </div>
      );
    }

    function SkipDayToggle({ skipped, date }: { skipped: boolean; date: string }) {
      const invalidate = useInvalidateDayAndStats(date);
      const mutation = useMutation({
        mutationFn: (newSkipped: boolean) => kcalClient.setSkipped({ skipped: newSkipped, date }),
        onSuccess: invalidate,
      });

      return (
        <button
          onClick={() => mutation.mutate(!skipped)}
          disabled={mutation.isPending}
          className={`text-xs tracking-wider border-2 px-3 py-1.5 transition-colors disabled:opacity-50 ${
            skipped
              ? "border-yellow-500 bg-yellow-500 text-white hover:bg-transparent hover:text-yellow-500"
              : "border-foreground text-muted-foreground hover:bg-foreground hover:text-background"
          }`}
        >
          {mutation.isPending ? "..." : skipped ? "🍕 CHEAT DAY" : "SKIP DAY"}
        </button>
      );
    }

    function DayView({ date }: { date: string }) {
      const { data } = useSuspenseQuery({
        queryKey: dayKeys.day(date),
        queryFn: () => kcalClient.getDay(date),
        refetchInterval: dayRefetchInterval,
      });

      if (data.skipped) {
        return (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-4xl font-bold tracking-tighter text-yellow-500">
                  🍕 CHEAT DAY
                </div>
                <div className="text-xs tracking-wider mt-0.5 text-yellow-500">
                  COUNTS AS {SKIPPED_DAY_KCAL} KCAL
                </div>
              </div>
              <SkipDayToggle skipped={data.skipped} date={date} />
            </div>

            {data.entries.length > 0 && (
              <div className="border-t border-muted pt-1 opacity-50">
                <div className="text-[10px] tracking-wider text-muted-foreground mb-2">
                  ENTRIES (IGNORED — DAY SCORED AS {SKIPPED_DAY_KCAL})
                </div>
                {data.entries.map((entry) => (
                  <EntryItem key={entry.id} entry={entry} date={date} />
                ))}
              </div>
            )}
          </div>
        );
      }

      return (
        <div className="space-y-6">
          {(() => {
            const status = getStatus(data.total, data.limit);
            const counterColor = statusTextColor[status];
            const isOver = status === "over";
            const isCurrentDay = date === todayStr();
            return (
              <>
                <div className="flex items-center justify-between">
                  <div>
                    <div className={`text-4xl font-bold tabular-nums tracking-tighter ${counterColor}`}>
                      {isOver && "🔥 "}{data.total}{isOver && " 🔥"}
                    </div>
                    <div className={`text-xs tracking-wider mt-0.5 ${counterColor || "text-muted-foreground"}`}>
                      {isOver ? "⚠️ OVER LIMIT" : "KCAL"}
                    </div>
                    <MacroTotals data={data} />
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <LimitSetter currentLimit={data.limit} date={date} />
                    <BurnSetter currentBurn={data.burn} date={date} />
                    <SkipDayToggle skipped={data.skipped} date={date} />
                  </div>
                </div>

                <ProgressBar total={data.total} limit={data.limit} />

                <LiveBurnCounter burnRate={data.burn} consumed={data.total} limit={data.limit} isToday={isCurrentDay} />
              </>
            );
          })()}

          <div className="border-t-2 border-foreground pt-4">
            <AddEntryForm date={date} />
          </div>

          {data.entries.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8 tracking-wider">
              NO ENTRIES YET
            </p>
          ) : (
            <div className="border-t border-muted pt-1">
              {data.entries.map((entry) => (
                <EntryItem key={entry.id} entry={entry} date={date} />
              ))}
            </div>
          )}
        </div>
      );
    }

    function ErrorFallback({ error, resetErrorBoundary }: { error: Error; resetErrorBoundary: () => void }) {
      return <AppErrorMessage error={error} retry={resetErrorBoundary} />;
    }

    function LoadingFallback() {
      return (
        <div className="flex items-center justify-center py-12 gap-2 text-muted-foreground">
          <span className="text-xs tracking-wider">LOADING</span>
          <span className="animate-pulse">■</span>
        </div>
      );
    }

    // ── App ──────────────────────────────────────────────────────

    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: 1 },
      },
    });

    function App() {
      const [date, setDate] = useState(todayStr());

      const goBack = useCallback(() => setDate((d) => shiftDate(d, -1)), []);
      const goForward = useCallback(() => setDate((d) => shiftDate(d, 1)), []);
      const goToday = useCallback(() => setDate(todayStr()), []);

      useEffect(() => {
        const handler = (e: KeyboardEvent) => {
          if (e.metaKey || e.ctrlKey || e.altKey) return;
          // Do not hijack arrow keys used for cursor movement inside a field.
          const target = e.target as HTMLElement | null;
          if (
            target &&
            (target.isContentEditable ||
              ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
          ) {
            return;
          }
          if (e.key === "ArrowLeft") goBack();
          if (e.key === "ArrowRight") goForward();
        };
        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
      }, [goBack, goForward]);

      const isToday = date === todayStr();

      return (
        <QueryClientProvider client={queryClient}>
          <div className="min-h-[100dvh] flex flex-col sm:items-center sm:justify-start p-0 sm:p-4 sm:pt-20">
            <div className="w-full max-w-md flex flex-col min-h-[100dvh] sm:min-h-0">

              {/* Header */}
              <div className="border-b-2 border-foreground sm:border-2">
                <div className="flex items-center justify-between px-4 py-3 border-b-2 border-foreground">
                  <h1 className="text-xs font-bold tracking-[0.3em]">KCAL TRACKER</h1>
                  {!isToday && (
                    <button
                      onClick={goToday}
                      className="text-xs tracking-wider text-muted-foreground hover:text-foreground transition-colors"
                    >
                      TODAY →
                    </button>
                  )}
                </div>

                {/* Date navigation */}
                <div className="flex items-center justify-between px-4 py-4">
                  <button
                    onClick={goBack}
                    className="w-10 h-10 border-2 border-foreground flex items-center justify-center text-lg font-bold hover:bg-foreground hover:text-background transition-colors select-none"
                  >
                    ←
                  </button>
                  <div className="text-center">
                    <div className="text-sm font-bold tracking-wider">{formatDate(date)}</div>
                    <div className="text-xs text-muted-foreground tracking-wider mt-0.5">{date}</div>
                  </div>
                  <button
                    onClick={goForward}
                    className="w-10 h-10 border-2 border-foreground flex items-center justify-center text-lg font-bold hover:bg-foreground hover:text-background transition-colors select-none"
                  >
                    →
                  </button>
                </div>
              </div>

              {/* Body */}
              <div className="flex-1 sm:flex-none border-b-2 border-foreground sm:border-2 sm:border-t-0 p-5">
                <ErrorBoundary FallbackComponent={ErrorFallback} resetKeys={[date]}>
                  <Suspense fallback={<LoadingFallback />}>
                    <DayView date={date} />
                  </Suspense>
                </ErrorBoundary>
              </div>

              {/* Average Intake */}
              <div className="sm:border-2 sm:border-t-0 border-b-2 border-foreground sm:border-b-2">
                <ErrorBoundary FallbackComponent={ErrorFallback}>
                  <Suspense fallback={null}>
                    <AverageIntake />
                  </Suspense>
                </ErrorBoundary>
              </div>

              {/* Cumulative */}
              <div className="sm:border-2 sm:border-t-0 border-b-2 border-foreground sm:border-b-2">
                <ErrorBoundary FallbackComponent={ErrorFallback}>
                  <Suspense fallback={null}>
                    <CumulativeWeightChange />
                  </Suspense>
                </ErrorBoundary>
              </div>

              {/* Footer */}
              <div className="text-center py-3 sm:mt-3 sm:py-0">
                <span className="text-[10px] tracking-wider text-muted-foreground">
                  ← → KEYS TO NAVIGATE
                </span>
              </div>

            </div>
          </div>
        </QueryClientProvider>
      );
    }

    createRoot(document.getElementById("root")!).render(<App />);
  </script>
</body>
</html>
"""


@app.get("/{path:path}", response_class=HTMLResponse)
async def spa(path: str):
    return HTML

@click.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host")
@click.option("--port", default=8765, type=int, help="Bind port")
@click.option("--db", default="kcal.db", show_default=True, help="Path to SQLite database")
def main(host: str, port: int | None, db: str):
    global repo, estimator
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    repo = SqliteKcalRepository(db)
    estimator = build_estimator()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
