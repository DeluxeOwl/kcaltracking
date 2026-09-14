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
from typing import Annotated, Literal

import click
import uvicorn
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import AfterValidator, BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
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

# A day can be marked to override how its own entries score it.
#   cheat    — entries are ignored and the day is scored as one fixed blowout,
#              so a known binge still weighs on the numbers.
#   excluded — the day is left out of every computation, as though it never
#              happened. For days that genuinely cannot be accounted for (eating
#              at someone else's table), where a guessed figure is worse than no
#              figure at all.
# An unmarked day is scored from its entries, which is the ordinary case.
DayMark = Literal["cheat", "excluded"]
DAY_CHEAT: DayMark = "cheat"
DAY_EXCLUDED: DayMark = "excluded"

# What one cheat day is scored as, in kcal.
CHEAT_DAY_KCAL = 4000

# Sorts below and above every real ISO date, so an unbounded date range needs no
# separate query. ISO dates compare correctly as plain strings.
_DATE_MIN = ""
_DATE_MAX = "9999-12-31"

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
    def get_day_mark(self, entry_date: str) -> DayMark | None: ...

    @abstractmethod
    def set_day_mark(self, entry_date: str, mark: DayMark | None) -> None: ...

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
        # Must run before the CREATE below, which would otherwise shadow the
        # legacy table and strand every day already marked in it.
        self._migrate_day_marks()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS day_marks ("
            "  entry_date TEXT PRIMARY KEY,"
            f"  mark TEXT NOT NULL DEFAULT '{DAY_CHEAT}'"
            ")"
        )
        self._migrate_entries()
        self._conn.commit()

    def _migrate_day_marks(self) -> None:
        """Carry a pre-day_marks database forward.

        The table began life as `skipped_days`, a bare list of dates that all
        meant one thing: a cheat day. Renaming it keeps those rows, and the
        `mark` column's default records exactly what they always meant.
        """
        tables = {
            row[0]
            for row in self._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if "skipped_days" in tables and "day_marks" not in tables:
            log.info("Migrating skipped_days to day_marks")
            self._conn.execute("ALTER TABLE skipped_days RENAME TO day_marks")
            tables.add("day_marks")

        if "day_marks" not in tables:
            return
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(day_marks)")}
        if "mark" not in columns:
            self._conn.execute(
                f"ALTER TABLE day_marks ADD COLUMN mark TEXT NOT NULL DEFAULT '{DAY_CHEAT}'"
            )

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

    def get_day_mark(self, entry_date: str) -> DayMark | None:
        row = self._conn.execute(
            "SELECT mark FROM day_marks WHERE entry_date = ?",
            (entry_date,),
        ).fetchone()
        return row[0] if row else None

    def set_day_mark(self, entry_date: str, mark: DayMark | None) -> None:
        """Mark a day, or clear its mark with None to track it normally again."""
        with self._lock:
            if mark is None:
                self._conn.execute(
                    "DELETE FROM day_marks WHERE entry_date = ?",
                    (entry_date,),
                )
            else:
                self._conn.execute(
                    "INSERT INTO day_marks (entry_date, mark) VALUES (?, ?) "
                    "ON CONFLICT(entry_date) DO UPDATE SET mark = excluded.mark",
                    (entry_date, mark),
                )
            self._conn.commit()

    def _day_marks(self, start: str = _DATE_MIN, end: str = _DATE_MAX) -> dict[str, str]:
        """Every marked day in [start, end), keyed by date."""
        rows = self._conn.execute(
            "SELECT entry_date, mark FROM day_marks WHERE entry_date >= ? AND entry_date < ?",
            (start, end),
        ).fetchall()
        return dict(rows)

    def average_intake(self, days: int) -> dict:
        """Compute average daily kcal intake over the last N days (excluding today).

        Cheat days are scored as CHEAT_DAY_KCAL. Excluded days are left out of
        both the sum and the day count, so they move the average neither way.
        """
        from datetime import timedelta
        today = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        marks = self._day_marks(start_date, today)
        cheat_dates = {d for d, mark in marks.items() if mark == DAY_CHEAT}
        excluded_dates = {d for d, mark in marks.items() if mark == DAY_EXCLUDED}

        rows = self._conn.execute(
            "SELECT entry_date, SUM(kcal) FROM entries "
            "WHERE entry_date >= ? AND entry_date < ? "
            "GROUP BY entry_date ORDER BY entry_date",
            (start_date, today),
        ).fetchall()

        # A cheat day counts even with no entries of its own, so it contributes a
        # date. An excluded day only ever takes one away, entries or not.
        entry_totals = {entry_date: total for entry_date, total in rows}
        all_dates = sorted((set(entry_totals.keys()) | cheat_dates) - excluded_dates)

        day_totals = [
            {
                "date": entry_date,
                "total": CHEAT_DAY_KCAL if entry_date in cheat_dates else entry_totals[entry_date],
                "mark": marks.get(entry_date),
            }
            for entry_date in all_dates
        ]

        counted = len(day_totals)
        avg = round(sum(d["total"] for d in day_totals) / counted, 1) if counted > 0 else 0

        return {
            "days_requested": days,
            "days_counted": counted,
            "days_excluded": len(excluded_dates),
            "average_kcal": avg,
            "days": day_totals,
        }

    def cumulative_weight_change(self) -> dict:
        """Compute cumulative weight change across all completed days.

        Cheat days are scored as CHEAT_DAY_KCAL. Excluded days contribute no
        deficit and no surplus: the running total simply steps over them.
        """
        KCAL_PER_GRAM_FAT = 7.7

        # Get all dates that have entries
        rows = self._conn.execute(
            "SELECT entry_date, SUM(kcal) FROM entries GROUP BY entry_date ORDER BY entry_date"
        ).fetchall()

        # Get all burn rates (sorted by date)
        burn_rows = self._conn.execute(
            "SELECT entry_date, burn_kcal FROM daily_burns ORDER BY entry_date"
        ).fetchall()

        marks = self._day_marks()
        cheat_dates = {d for d, mark in marks.items() if mark == DAY_CHEAT}
        excluded_dates = {d for d, mark in marks.items() if mark == DAY_EXCLUDED}

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

        # A cheat day counts even with no entries of its own, so it contributes a
        # date. An excluded day only ever takes one away, entries or not.
        entry_totals = {entry_date: consumed for entry_date, consumed in rows}
        all_dates = sorted((set(entry_totals.keys()) | cheat_dates) - excluded_dates)

        total_grams = 0.0
        day_details = []

        for entry_date in all_dates:
            if entry_date >= today:
                # Skip today and future days (not yet complete)
                continue

            if entry_date in cheat_dates:
                consumed = CHEAT_DAY_KCAL
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
                "mark": marks.get(entry_date),
            })

        return {
            "total_grams": round(total_grams, 3),
            "days_counted": len(day_details),
            "days_excluded": len([d for d in excluded_dates if d < today]),
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
  with it. Protein supplies 4 kcal per gram and fat 9 kcal per gram
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


# ── Quick kcal estimation ─────────────────────────────────────────────

KCAL_ESTIMATE_INSTRUCTIONS = """\
You estimate the nutrition of a food or meal for a calorie tracking app.

Break the description into its distinct food items. For each item give:
- its weight in grams, and
- its nutrition PER 100 GRAMS: kcal, protein, fat and fiber.

Do NOT multiply or add anything up yourself. The app multiplies each item's
per-100g figures by its weight and sums the items. Your job is only to supply
accurate weights and per-100g densities.

Rules:
- Echo each item's quantity from the description exactly, e.g. "955g red
  cabbage", "177g sausage".
- If the user gives per-100g or per-serving nutrition values for an item, use
  those exact values. Do not substitute your own.
- Use realistic reference densities for common foods, e.g. raw red cabbage
  ~25 kcal/100g, boiled lentils ~116 kcal/100g.
- Assume ordinary supermarket products and typical serving sizes when the
  description is vague. Never ask for clarification.
- Only plant foods contain fiber; meat, fish, eggs and dairy have none.
- Give a one-line note listing the assumptions (which products / reference
  values you used).
"""


class QuickEstimateItem(BaseModel):
    """One food item with its weight and per-100g nutrition."""

    name: str = Field(description="The food including its quantity, echoed from the description")
    grams: float = Field(ge=0, description="Weight of this item in grams")
    kcal_per_100g: float = Field(ge=0, description="Energy per 100g in kcal")
    protein_per_100g: float = Field(ge=0, description="Grams of protein per 100g")
    fat_per_100g: float = Field(ge=0, description="Grams of fat per 100g")
    fiber_per_100g: float = Field(ge=0, description="Grams of dietary fiber per 100g")


class KcalEstimateResult(BaseModel):
    """A per-item nutrition estimate the app totals up itself."""

    items: list[QuickEstimateItem]
    note: str = Field(description="One-line explanation of the assumptions behind the figures")


class KcalEstimator:
    """Estimates kcal and macros for a free-text food description via one LLM call.

    The model supplies per-item weights and per-100g densities only; the totals
    are computed here so arithmetic is exact rather than hallucinated. Standalone
    and stateless: it answers a question and stores nothing.
    """

    def __init__(self, api_key: str, model_name: str = MACRO_MODEL) -> None:
        model = OpenRouterModel(model_name, provider=OpenRouterProvider(api_key=api_key))
        # Give the model room to reason: it has to pick realistic per-100g
        # densities and honour any values the user supplies, which is worth the
        # extra latency for a one-off, on-demand estimate.
        settings = OpenRouterModelSettings(openrouter_reasoning={"effort": "high"})
        self._agent = Agent(
            model,
            output_type=KcalEstimateResult,
            instructions=KCAL_ESTIMATE_INSTRUCTIONS,
            model_settings=settings,
            retries=0,
        )

    async def estimate(self, description: str) -> dict:
        result = await asyncio.wait_for(
            self._agent.run(f"Food: {description}"),
            timeout=MACRO_TIMEOUT_S,
        )
        return self._compute(result.output)

    @staticmethod
    def _compute(out: "KcalEstimateResult") -> dict:
        """Multiply per-100g figures by weight and sum — in code, not the LLM."""
        items: list[dict] = []
        total_kcal = 0.0
        totals = {name: 0.0 for name in MACRO_NAMES}
        for it in out.items:
            if not it.name.strip():
                continue
            factor = max(0.0, it.grams) / 100
            kcal = max(0.0, it.kcal_per_100g) * factor
            macros = {
                name: max(0.0, getattr(it, f"{name}_per_100g")) * factor
                for name in MACRO_NAMES
            }
            total_kcal += kcal
            for name in MACRO_NAMES:
                totals[name] += macros[name]
            items.append({
                "name": it.name.strip(),
                "grams": round(max(0.0, it.grams), 1),
                "kcal": round(kcal),
                **{f"{name}_g": round(macros[name], 1) for name in MACRO_NAMES},
            })
        return {
            "kcal": round(total_kcal),
            "macros": {name: round(totals[name], 1) for name in MACRO_NAMES},
            "items": items,
            "note": out.note.strip(),
        }


estimator: MacroEstimator | None = None
kcal_estimator: KcalEstimator | None = None


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


def build_kcal_estimator() -> KcalEstimator | None:
    """Build the quick-estimate helper, or return None without an API key."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        return KcalEstimator(api_key)
    except Exception:
        log.exception("Could not build kcal estimator — quick estimate disabled")
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

class SetDayMarkRequest(BaseModel):
    # null clears the mark and returns the day to ordinary tracking.
    mark: DayMark | None = None
    date: DateStr

class EstimateRequest(BaseModel):
    description: DescriptionStr

class EstimateItem(BaseModel):
    name: str
    grams: float
    kcal: int
    protein_g: float
    fat_g: float
    fiber_g: float

class EstimateResponse(BaseModel):
    kcal: int
    macros: dict[str, float] = {}
    items: list[EstimateItem] = []
    note: str

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
    # "cheat", "excluded", or null for an ordinary day scored from its entries.
    mark: DayMark | None
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
    mark = repo.get_day_mark(day)
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
        mark=mark,
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


@api.put("/day-mark", status_code=200)
async def set_day_mark(body: SetDayMarkRequest):
    repo.set_day_mark(body.date, body.mark)
    return {"date": body.date, "mark": body.mark}


@api.post("/estimate", response_model=EstimateResponse)
async def estimate_kcal(body: EstimateRequest):
    """Answer "how many kcal is this?" without saving anything."""
    if kcal_estimator is None:
        raise HTTPException(status_code=503, detail="Estimation is not configured")
    try:
        result = await kcal_estimator.estimate(body.description)
    except Exception as exc:
        log.warning("Quick kcal estimate failed for %r: %s", body.description, exc)
        raise HTTPException(status_code=502, detail="Could not estimate calories — try again")
    return EstimateResponse(**result)


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
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
  <meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)" />
  <meta name="theme-color" content="#080808" media="(prefers-color-scheme: dark)" />
  <title>KCAL</title>

  <!-- Resolve the theme before first paint so there is no flash of the wrong
       one. Everything downstream reads html[data-theme]. -->
  <script>
    (function () {
      var KEY = "kcal.theme";
      var root = document.documentElement;

      function preferred() {
        var stored = null;
        try { stored = localStorage.getItem(KEY); } catch (e) { /* private mode */ }
        if (stored === "light" || stored === "dark") return stored;
        return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light";
      }

      root.dataset.theme = preferred();

      // Colours cross-fade on a theme switch, but only for the switch itself —
      // a permanent global transition would smear every other state change.
      window.__kcalSetTheme = function (next) {
        root.classList.add("theme-transition");
        root.dataset.theme = next;
        try { localStorage.setItem(KEY, next); } catch (e) { /* private mode */ }
        clearTimeout(window.__kcalThemeTimer);
        window.__kcalThemeTimer = setTimeout(function () {
          root.classList.remove("theme-transition");
        }, 240);
      };
    })();
  </script>

  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />

  <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4.3.3/dist/index.global.js"></script>

  <style type="text/tailwindcss">
    /* ════════════════════════════════════════════════════════════════
       DESIGN TOKENS
       Every literal — colour, radius, size, shadow, duration — is declared
       once here. Components never hardcode a value; they name a token. A
       theme is then nothing but a different set of values for the same names.
       ════════════════════════════════════════════════════════════════ */

    :root {
      /* ── Typeface ─────────────────────────────────────────────── */
      --font-family-ui: "Inter", ui-sans-serif, system-ui, -apple-system,
        "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      --font-family-mono: ui-monospace, "SF Mono", "JetBrains Mono",
        "IBM Plex Mono", Menlo, monospace;

      /* ── Type scale ───────────────────────────────────────────── */
      --fs-micro: 0.625rem;   /* eyebrow labels */
      --fs-mini: 0.6875rem;   /* dense metadata */
      --fs-stat: 1.75rem;     /* card figures */
      --fs-hero: 2.75rem;     /* the day's total */

      --lh-micro: 1.4;
      --lh-mini: 1.45;
      --lh-stat: 1.1;
      --lh-hero: 1;

      --ls-label: 0.08em;     /* uppercase eyebrows */
      --ls-title: 0.14em;     /* the app title */
      --ls-figure: -0.025em;  /* big numerals pull tighter */
      --ls-heading: -0.015em; /* headings track in slightly */

      /* ── Radii ────────────────────────────────────────────────── */
      --r-xs: 4px;
      --r-sm: 6px;
      --r-md: 8px;
      --r-lg: 12px;
      --r-xl: 16px;
      /* The card step — a custom rung between lg and xl. */
      --r-card: 10px;
      --r-full: 9999px;

      /* ── Metrics ──────────────────────────────────────────────── */
      --app-width: 30rem;
      --gutter: 1rem;
      --card-pad: 1.25rem;
      --control-h-sm: 1.75rem;
      --control-h-md: 2rem;
      --control-h-lg: 2.25rem;
      --bar-h: 0.375rem;
      --ring-w: 2px;      /* focus ring, drawn as an offset outline */
      /* Aligns an entry's sub-lines under its description:
         time (2.75rem) + gap (0.75rem) + kcal (3rem) + gap (0.75rem). */
      --macro-indent: 7.25rem;

      /* ── Motion ───────────────────────────────────────────────── */
      --ease-out-expo: cubic-bezier(0.22, 1, 0.36, 1);
      --ease-standard: cubic-bezier(0.4, 0, 0.2, 1);
      --dur-fast: 150ms;
      --dur-med: 200ms;
      --dur-slow: 300ms;

      /* ── Elevation ────────────────────────────────────────────── */
      --z-toggle: 30;
      --z-overlay: 50;
    }

    /* ── Light theme ────────────────────────────────────────────── */
    :root,
    :root[data-theme="light"] {
      color-scheme: light;

      /* Nine-step neutral ramp; depth is tonal, so a surface is "raised"
         by moving a rung, never by casting a shadow. */
      --canvas: #ffffff;      /* surface-1000 */
      --surface: #ececec;     /* surface-500  — card fill */
      --raised: #f4f4f4;      /* surface-700  — panels, fields */
      --sunken: #e4e4e4;      /* surface-400  — tracks, wells */
      --hover: #d4d4d4;       /* surface-200  — the single hover fill */
      --active: #c9c9c9;

      --fg: #191919;
      --fg-strong: #000000;
      --fg-muted: #626262;
      --fg-subtle: #7a7a7a;
      --on-accent: #ffffff;

      /* Dividers sit a rung away from the card fill so they stay visible
         without becoming a drawn border. */
      --line: #dcdcdc;
      --line-strong: #c9c9c9;

      /* The one chromatic value in the palette — identical in both modes. */
      --accent: #ff6b01;
      --accent-hover: #f05f00;
      --accent-active: #d95500;
      --accent-soft: rgba(255, 107, 1, 0.14);

      --positive: #15803d;
      --positive-soft: #eaf6ee;
      --caution: #a16207;
      --caution-soft: #fbf6e2;
      --warn: #b45309;
      --warn-soft: #f9efe4;
      --danger: #b91c1c;
      --danger-soft: #f7e9e9;

      --overlay: rgba(0, 0, 0, 0.32);
      --scrim-blur: 4px;

      --elev-xs: 0 1px 2px rgba(0, 0, 0, 0.04);
      --elev-sm: 0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04);
      --elev-md: 0 4px 12px rgba(0, 0, 0, 0.08), 0 2px 4px rgba(0, 0, 0, 0.04);
      --elev-lg: 0 4px 12px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    /* ── Dark theme ─────────────────────────────────────────────── */
    :root[data-theme="dark"] {
      color-scheme: dark;

      /* The same ramp inverted — the two themes are strict mirrors. */
      --canvas: #080808;
      --surface: #212121;
      --raised: #171717;
      --sunken: #262626;
      --hover: #303030;
      --active: #3a3a3a;

      --fg: #e6e6e6;
      --fg-strong: #ffffff;
      --fg-muted: #9d9d9d;
      --fg-subtle: #858585;
      --on-accent: #ffffff;

      --line: #303030;
      --line-strong: #3a3a3a;

      --accent: #ff6b01;
      --accent-hover: #f05f00;
      --accent-active: #d95500;
      --accent-soft: rgba(255, 107, 1, 0.18);

      --positive: #4ade80;
      --positive-soft: #10291a;
      --caution: #facc15;
      --caution-soft: #2b240d;
      --warn: #fb923c;
      --warn-soft: #2e1d0d;
      --danger: #f87171;
      --danger-soft: #301313;

      --overlay: rgba(0, 0, 0, 0.62);
      --scrim-blur: 4px;

      --elev-xs: 0 1px 2px rgba(0, 0, 0, 0.3);
      --elev-sm: 0 1px 3px rgba(0, 0, 0, 0.4), 0 1px 2px rgba(0, 0, 0, 0.3);
      --elev-md: 0 4px 12px rgba(0, 0, 0, 0.5), 0 2px 4px rgba(0, 0, 0, 0.3);
      --elev-lg: 0 4px 12px rgba(0, 0, 0, 0.55), 0 2px 4px rgba(0, 0, 0, 0.35);
    }

    /* ════════════════════════════════════════════════════════════════
       TOKENS → TAILWIND
       `inline` keeps the var() reference in the generated utility, so a
       theme swap repaints without regenerating any CSS.
       ════════════════════════════════════════════════════════════════ */

    @theme inline {
      --font-sans: var(--font-family-ui);
      --font-mono: var(--font-family-mono);

      --color-canvas: var(--canvas);
      --color-surface: var(--surface);
      --color-raised: var(--raised);
      --color-sunken: var(--sunken);
      --color-hover: var(--hover);
      --color-active: var(--active);

      --color-fg: var(--fg);
      --color-fg-strong: var(--fg-strong);
      --color-fg-muted: var(--fg-muted);
      --color-fg-subtle: var(--fg-subtle);
      --color-on-accent: var(--on-accent);

      --color-line: var(--line);
      --color-line-strong: var(--line-strong);

      --color-accent: var(--accent);
      --color-accent-hover: var(--accent-hover);
      --color-accent-active: var(--accent-active);
      --color-accent-soft: var(--accent-soft);

      --color-positive: var(--positive);
      --color-positive-soft: var(--positive-soft);
      --color-caution: var(--caution);
      --color-caution-soft: var(--caution-soft);
      --color-warn: var(--warn);
      --color-warn-soft: var(--warn-soft);
      --color-danger: var(--danger);
      --color-danger-soft: var(--danger-soft);

      --radius-xs: var(--r-xs);
      --radius-sm: var(--r-sm);
      --radius-md: var(--r-md);
      --radius-lg: var(--r-lg);
      --radius-xl: var(--r-xl);
      --radius-card: var(--r-card);

      --shadow-xs: var(--elev-xs);
      --shadow-sm: var(--elev-sm);
      --shadow-md: var(--elev-md);
      --shadow-lg: var(--elev-lg);

      --text-micro: var(--fs-micro);
      --text-micro--line-height: var(--lh-micro);
      --text-mini: var(--fs-mini);
      --text-mini--line-height: var(--lh-mini);
      --text-stat: var(--fs-stat);
      --text-stat--line-height: var(--lh-stat);
      --text-hero: var(--fs-hero);
      --text-hero--line-height: var(--lh-hero);

      --tracking-label: var(--ls-label);
      --tracking-title: var(--ls-title);
      --tracking-figure: var(--ls-figure);
      --tracking-heading: var(--ls-heading);

      --ease-swift: var(--ease-out-expo);
      --ease-std: var(--ease-standard);
    }

    /* ════════════════════════════════════════════════════════════════
       BASE
       ════════════════════════════════════════════════════════════════ */

    @layer base {
      * {
        border-color: var(--line);
      }

      html {
        -webkit-text-size-adjust: 100%;
        -webkit-tap-highlight-color: transparent;
      }

      body {
        background: var(--canvas);
        color: var(--fg);
        font-family: var(--font-family-ui);
        font-feature-settings: "cv02", "cv03", "cv04", "cv11";
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
      }

      ::selection {
        background: var(--accent-soft);
        color: var(--accent);
      }

      /* Headings carry hierarchy through size and colour, not weight. */
      h1, h2, h3, h4 {
        font-weight: 500;
        letter-spacing: var(--ls-heading);
      }

      /* One focus treatment everywhere: an accent ring, never removed. */
      :focus-visible {
        outline: var(--ring-w) solid var(--accent);
        outline-offset: 3px;
      }

      /* Native spinners make a kcal field look like a form, not a figure. */
      input[type="number"]::-webkit-inner-spin-button,
      input[type="number"]::-webkit-outer-spin-button {
        -webkit-appearance: none;
        margin: 0;
      }
      input[type="number"] {
        -moz-appearance: textfield;
      }

      * {
        scrollbar-width: thin;
        scrollbar-color: var(--line-strong) transparent;
      }
      ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
      }
      ::-webkit-scrollbar-track {
        background: transparent;
      }
      ::-webkit-scrollbar-thumb {
        background: var(--line-strong);
        border: 3px solid transparent;
        border-radius: var(--r-full);
        background-clip: content-box;
      }
      ::-webkit-scrollbar-thumb:hover {
        background: var(--fg-subtle);
        border: 3px solid transparent;
        background-clip: content-box;
      }
    }

    /* ════════════════════════════════════════════════════════════════
       COMPONENT PRIMITIVES
       Plain CSS over tokens, so every recurring shape — card, button,
       field, pill — is defined once and stays consistent across themes.
       ════════════════════════════════════════════════════════════════ */

    @layer components {
      .theme-transition,
      .theme-transition *,
      .theme-transition *::before,
      .theme-transition *::after {
        transition:
          background-color var(--dur-med) var(--ease-standard),
          border-color var(--dur-med) var(--ease-standard),
          color var(--dur-med) var(--ease-standard),
          box-shadow var(--dur-med) var(--ease-standard) !important;
      }

      /* Depth is tonal: the card reads as a card because it is a different
         rung on the surface ramp, not because it casts a shadow. */
      .card {
        background: var(--surface);
        border: 0;
        border-radius: var(--r-card);
        box-shadow: none;
        transition: background-color var(--dur-med) var(--ease-standard);
      }

      .panel {
        background: var(--raised);
        border: 0;
        border-radius: var(--r-md);
      }

      .eyebrow {
        font-size: var(--fs-micro);
        line-height: var(--lh-micro);
        letter-spacing: var(--ls-label);
        text-transform: uppercase;
        font-weight: 500;
        color: var(--fg-subtle);
      }

      .figure {
        font-variant-numeric: tabular-nums;
        letter-spacing: var(--ls-figure);
        font-weight: 500;
      }

      /* ── Buttons ────────────────────────────────────────────── */
      .btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        flex-shrink: 0;
        border: 0;
        border-radius: var(--r-full);
        font-family: inherit;
        font-weight: 500;
        white-space: nowrap;
        cursor: pointer;
        user-select: none;
        transition:
          background-color var(--dur-fast) var(--ease-standard),
          border-color var(--dur-fast) var(--ease-standard),
          color var(--dur-fast) var(--ease-standard),
          opacity var(--dur-fast) var(--ease-standard);
      }
      .btn:disabled {
        opacity: 0.45;
        pointer-events: none;
      }

      .btn-sm { height: var(--control-h-sm); padding-inline: 0.7rem; font-size: var(--fs-mini); }
      .btn-md { height: var(--control-h-md); padding-inline: 0.9rem; font-size: 0.75rem; }
      .btn-lg { height: var(--control-h-lg); padding-inline: 1.15rem; font-size: 0.8125rem; }
      .btn-icon { padding-inline: 0; aspect-ratio: 1 / 1; }

      .btn-primary { background: var(--accent); color: var(--on-accent); }
      .btn-primary:hover { background: var(--accent-hover); }
      .btn-primary:not(:disabled):active { background: var(--accent-active); }

      /* Secondary fills step along the ramp on hover — no lift, no border. */
      .btn-outline { background: var(--sunken); color: var(--fg); }
      .btn-outline:hover { background: var(--hover); }

      .btn-ghost { color: var(--fg-muted); }
      .btn-ghost:hover { background: var(--hover); color: var(--fg-strong); }

      .btn-soft { background: var(--sunken); color: var(--fg); }
      .btn-soft:hover { background: var(--hover); }

      .btn-danger { background: var(--danger-soft); color: var(--danger); }
      .btn-danger:hover { background: var(--danger); color: var(--on-accent); }

      /* ── Fields ─────────────────────────────────────────────── */
      .field {
        background: var(--canvas);
        border: 1px solid var(--line);
        border-radius: var(--r-md);
        color: var(--fg);
        font-family: inherit;
        font-size: 0.8125rem;
        padding: 0.45rem 0.65rem;
        transition:
          background-color var(--dur-fast) var(--ease-standard),
          border-color var(--dur-fast) var(--ease-standard),
          box-shadow var(--dur-fast) var(--ease-standard);
      }
      .field::placeholder { color: var(--fg-subtle); }
      .field:focus {
        outline: var(--ring-w) solid var(--accent);
        outline-offset: -1px;
        border-color: var(--accent);
      }
      .field:disabled { opacity: 0.55; cursor: not-allowed; }
      .field-sm { font-size: var(--fs-mini); padding: 0.25rem 0.45rem; }

      /* ── Composer ─────────────────────────────────── */
      /* Adding an entry is one thought, so it reads as one control: a single
         recessed pill holding both fields and the submit, rather than three
         bordered boxes sitting next to each other. */
      .composer {
        display: flex;
        align-items: center;
        gap: 0;
        padding: 0.25rem;
        background: var(--canvas);
        border: 0;
        border-radius: var(--r-full);
        transition: background-color var(--dur-fast) var(--ease-standard);
      }
      .composer:focus-within {
        outline: var(--ring-w) solid var(--accent);
        outline-offset: 2px;
      }
      .composer-input {
        height: var(--control-h-lg);
        min-width: 0;
        padding-inline: 0.85rem;
        background: transparent;
        border: 0;
        color: var(--fg);
        font-family: inherit;
        font-size: 0.8125rem;
      }
      .composer-input:focus { outline: none; }
      .composer-input::placeholder { color: var(--fg-subtle); }
      .composer-input:disabled { opacity: 0.55; cursor: not-allowed; }
      .composer-rule {
        flex: none;
        width: 1px;
        height: 1.15rem;
        background: var(--line-strong);
      }

      /* ── Segmented control ──────────────────────────────────── */
      .seg {
        display: inline-flex;
        gap: 2px;
        padding: 2px;
        background: var(--sunken);
        border: 0;
        border-radius: var(--r-full);
      }
      .seg-item {
        border-radius: var(--r-full);
        padding: 0.2rem 0.55rem;
        font-size: var(--fs-mini);
        font-weight: 500;
        color: var(--fg-muted);
        cursor: pointer;
        transition:
          background-color var(--dur-fast) var(--ease-standard),
          color var(--dur-fast) var(--ease-standard);
      }
      .seg-item:hover { color: var(--fg-strong); }
      .seg-item[data-on="true"] {
        background: var(--canvas);
        color: var(--fg-strong);
        box-shadow: none;
      }

      /* ── Entry row ──────────────────────────────────────────── */
      .row {
        border-radius: var(--r-md);
        /* fallthrough: the only hover feedback is a fill step */
        transition: background-color var(--dur-fast) var(--ease-standard);
      }
      .row:hover { background: var(--hover); }

      /* A hover-revealed control is unreachable on touch, so it only hides
         where a real pointer exists. */
      .row-action { opacity: 1; }
      @media (hover: hover) and (pointer: fine) {
        .row-action { opacity: 0; }
        .row:hover .row-action,
        .row-action:focus-visible { opacity: 1; }
      }

      /* ── Modal ──────────────────────────────────────────────── */
      .scrim {
        background: var(--overlay);
        backdrop-filter: blur(var(--scrim-blur));
        -webkit-backdrop-filter: blur(var(--scrim-blur));
        animation: scrim-in var(--dur-med) var(--ease-standard);
      }
      .dialog {
        background: var(--canvas);
        border: 1px solid var(--line);
        border-radius: var(--r-lg);
        box-shadow: var(--elev-lg);
        animation: dialog-in var(--dur-slow) var(--ease-out-expo);
      }

      kbd {
        display: inline-block;
        min-width: 1.25rem;
        padding: 0.05rem 0.3rem;
        border: 0;
        border-radius: var(--r-xs);
        background: var(--sunken);
        color: var(--fg-muted);
        font-family: inherit;
        font-size: var(--fs-micro);
        text-align: center;
      }
    }

    @keyframes scrim-in {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    @keyframes dialog-in {
      from { opacity: 0; transform: translateY(8px) scale(0.985); }
      to { opacity: 1; transform: none; }
    }

    @media (prefers-reduced-motion: reduce) {
      *,
      *::before,
      *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
    }
  </style>

  <script type="importmap">
  {
    "imports": {
      "react": "https://esm.sh/react@19.2.8",
      "react/jsx-runtime": "https://esm.sh/react@19.2.8/jsx-runtime",
      "react/jsx-dev-runtime": "https://esm.sh/react@19.2.8/jsx-dev-runtime",
      "react-dom": "https://esm.sh/react-dom@19.2.8?deps=react@19.2.8",
      "react-dom/client": "https://esm.sh/react-dom@19.2.8/client?deps=react@19.2.8",
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
    import { Suspense, useState, useEffect, useCallback, useRef } from "react";
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

    // ── Types ────────────────────────────────────────────────────

    type MacrosState = "pending" | "ok" | "failed" | "skipped";

    // How a day is scored, overriding its own entries. "cheat" scores it as one
    // fixed blowout; "excluded" drops it from every computation. Null is an
    // ordinary day, scored from what it contains.
    type DayMark = "cheat" | "excluded";

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
      mark: DayMark | null;
      entries: Entry[];
    }

    // Display order and short labels. "Fat" and "Fib" are spelled out rather than
    // both reduced to "F", which would be ambiguous.
    const MACROS = [
      { key: "protein", short: "P", label: "Protein" },
      { key: "fat", short: "Fat", label: "Fat" },
      { key: "fiber", short: "Fib", label: "Fiber" },
    ] as const;

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
      setDayMark: (data: { mark: DayMark | null; date: string }) =>
        api.put("day-mark", { json: data }).json<{ date: string; mark: DayMark | null }>(),
      getCumulative: () =>
        api.get("cumulative").json<{ total_grams: number; days_counted: number; days_excluded: number }>(),
      getAverage: (days: number) =>
        api.get(`average/${days}`).json<{ days_requested: number; days_counted: number; days_excluded: number; average_kcal: number }>(),
      estimate: (description: string) =>
        api.post("estimate", { json: { description } }).json<{
          kcal: number;
          macros: Macros;
          items: { name: string; grams: number; kcal: number; protein_g: number; fat_g: number; fiber_g: number }[];
          note: string;
        }>(),
    } as const;

    // ── Icons ────────────────────────────────────────────────────

    // One shared frame keeps every glyph on the same optical weight and size.
    function Icon({ path, size = 16, className = "" }: { path: string; size?: number; className?: string }) {
      return (
        <svg
          viewBox="0 0 24 24"
          width={size}
          height={size}
          fill="none"
          stroke="currentColor"
          strokeWidth={1.75}
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
          className={className}
        >
          <path d={path} />
        </svg>
      );
    }

    const ICONS = {
      chevronLeft: "M15 18l-6-6 6-6",
      chevronRight: "M9 6l6 6-6 6",
      chevronDown: "M6 9l6 6 6-6",
      plus: "M12 5v14M5 12h14",
      close: "M18 6L6 18M6 6l12 12",
      check: "M20 6L9 17l-5-5",
      retry: "M20 11a8 8 0 1 0-2.3 5.6M20 5v6h-6",
      sun: "M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10zM12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4",
      moon: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z",
      sparkles: "M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3zM19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15z",
      trash: "M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2",
      arrowDown: "M12 5v14M6 13l6 6 6-6",
      arrowUp: "M12 19V5M6 11l6-6 6 6",
      flame: "M12 22a7 7 0 0 0 7-7c0-5-4-6-4-10-3 1-4 3.5-4 5.5C10 9 9 8 9 6c-1.5 1.5-2 4-2 6a7 7 0 0 0 5 10z",
      ban: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM5.6 5.6l12.8 12.8",
      pizza: "M12 21L3 6c5.5-3 12.5-3 18 0l-9 15zM10 10h.01M13.5 14h.01",
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
      if (dateStr === today) return "Today";
      if (dateStr === shiftDate(today, -1)) return "Yesterday";
      if (dateStr === shiftDate(today, 1)) return "Tomorrow";
      return d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
    }

    // Under a relative label ("Today") the calendar date is what's missing, so
    // spell it out. Under a date that already reads as one, only the exact ISO
    // form adds anything.
    function dateSubtitle(dateStr: string): string {
      const today = todayStr();
      const isRelative =
        dateStr === today ||
        dateStr === shiftDate(today, -1) ||
        dateStr === shiftDate(today, 1);
      if (!isRelative) return dateStr;
      return new Date(dateStr + "T12:00:00").toLocaleDateString("en-US", {
        weekday: "long",
        month: "long",
        day: "numeric",
      });
    }

    // ── Burn rate helpers ────────────────────────────────────────

    const KCAL_PER_GRAM_FAT = 7.7;
    // Must match the backend: a cheat day is scored as this many kcal.
    const CHEAT_DAY_KCAL = 4000;
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
          <div className="my-3 flex items-center justify-between gap-3 rounded-md bg-danger-soft px-3 py-2.5">
            <span className="text-xs text-danger">{message}</span>
            <button onClick={retry} className="btn btn-sm btn-danger">
              <Icon path={ICONS.retry} size={12} />
              Retry
            </button>
          </div>
        );
      }

      return <p className="mt-1.5 text-mini text-danger">{message}</p>;
    }

    // ── Theme ────────────────────────────────────────────────────

    type Theme = "light" | "dark";

    // The pre-paint script in <head> owns the initial resolution and the write
    // to localStorage, so React never has to guess and never double-applies.
    function useTheme(): [Theme, (next: Theme) => void] {
      const [theme, setTheme] = useState<Theme>(
        () => (document.documentElement.dataset.theme as Theme) || "light",
      );
      const apply = useCallback((next: Theme) => {
        window.__kcalSetTheme(next);
        setTheme(next);
      }, []);
      return [theme, apply];
    }

    function ThemeToggle() {
      const [theme, setTheme] = useTheme();
      const isDark = theme === "dark";

      return (
        <button
          onClick={() => setTheme(isDark ? "light" : "dark")}
          title={isDark ? "Switch to light" : "Switch to dark"}
          aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
          className="btn btn-outline btn-icon btn-lg absolute right-4 top-4 sm:right-6 sm:top-6"
          style={{ zIndex: "var(--z-toggle)" }}
        >
          {/* Both glyphs are mounted and cross-faded, so the swap has no flicker. */}
          <span className="relative block h-4 w-4">
            <span
              className={`absolute inset-0 transition-all duration-300 ease-swift ${
                isDark ? "scale-75 rotate-90 opacity-0" : "scale-100 rotate-0 opacity-100"
              }`}
            >
              <Icon path={ICONS.sun} />
            </span>
            <span
              className={`absolute inset-0 transition-all duration-300 ease-swift ${
                isDark ? "scale-100 rotate-0 opacity-100" : "scale-75 -rotate-90 opacity-0"
              }`}
            >
              <Icon path={ICONS.moon} />
            </span>
          </span>
        </button>
      );
    }

    // ── Status ───────────────────────────────────────────────────

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
      ok: "bg-accent",
      warning: "bg-caution",
      danger: "bg-warn",
      critical: "bg-danger",
      over: "bg-danger",
    };

    const statusTextColor: Record<Status, string> = {
      ok: "",
      warning: "text-caution",
      danger: "text-warn",
      critical: "text-danger",
      over: "text-danger",
    };

    function ProgressBar({ total, limit }: { total: number; limit: number | null }) {
      if (limit === null) return null;
      const pct = Math.min((total / limit) * 100, 100);
      const over = total > limit;
      const remaining = limit - total;
      const status = getStatus(total, limit);

      return (
        <div className="space-y-2">
          <div
            className="w-full overflow-hidden rounded-full bg-sunken"
            style={{ height: "var(--bar-h)" }}
          >
            <div
              className={`h-full rounded-full transition-all duration-500 ease-swift ${statusBarColor[status]}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="flex justify-between text-mini tabular-nums">
            <span className="text-fg-muted">
              <span className="font-medium text-fg">{total}</span> of {limit} kcal
            </span>
            <span className={over ? `font-medium ${statusTextColor[status]}` : statusTextColor[status] || "text-fg-muted"}>
              {over ? `${Math.abs(remaining)} over` : `${remaining} left`}
            </span>
          </div>
        </div>
      );
    }

    // ── Limit / burn setters ─────────────────────────────────────

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
          <button onClick={() => setEditing(true)} className="btn btn-sm btn-ghost gap-1.5">
            <span className="text-fg-subtle">{label}</span>
            {current !== null ? (
              <span className="font-medium tabular-nums text-fg">{current}</span>
            ) : (
              <span className="text-fg-subtle">— set</span>
            )}
          </button>
        );
      }

      return (
        <div className="flex flex-col items-end">
          <form onSubmit={handleSubmit(onSubmit)} className="flex items-center gap-1.5">
            <span className="eyebrow">{label}</span>
            <input
              type="number"
              autoFocus
              {...register("value", { required: true, min: 1 })}
              className="field field-sm w-16 text-center tabular-nums"
              placeholder={placeholder}
            />
            <button type="submit" disabled={isPending} className="btn btn-sm btn-primary btn-icon">
              <Icon path={ICONS.check} size={13} />
            </button>
            <button
              type="button"
              onClick={() => setEditing(false)}
              className="btn btn-sm btn-ghost btn-icon"
            >
              <Icon path={ICONS.close} size={13} />
            </button>
          </form>
          <AppErrorMessage error={error} />
        </div>
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
          label="Limit"
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
          label="Burn"
          current={currentBurn}
          placeholder="2200"
          onSave={(v) => mutation.mutateAsync(v)}
          isPending={mutation.isPending}
          error={mutation.error}
        />
      );
    }

    // ── Weight change ────────────────────────────────────────────

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

    const trendColor = {
      losing: "text-positive",
      gaining: "text-danger",
      neutral: "text-fg-muted",
    } as const;

    function TrendArrow({ trend, size = 14 }: { trend: "losing" | "gaining" | "neutral"; size?: number }) {
      if (trend === "neutral") return null;
      return (
        <Icon
          path={trend === "losing" ? ICONS.arrowDown : ICONS.arrowUp}
          size={size}
          className="inline-block shrink-0"
        />
      );
    }

    function ForecastRow({ label, grams }: { label: string; grams: number }) {
      const trend = gramsTrend(grams);
      return (
        <div className="flex items-center justify-end gap-2">
          <span className="text-micro text-fg-subtle">{label}</span>
          <span className={`inline-flex items-center gap-0.5 text-xs font-medium tabular-nums ${trendColor[trend]}`}>
            <TrendArrow trend={trend} size={12} />
            {formatGrams(grams)}
          </span>
        </div>
      );
    }

    function LiveBurnCounter({ burnRate, consumed, limit, isToday }: { burnRate: number | null; consumed: number; limit: number | null; isToday: boolean }) {
      const burn = useLiveBurn(burnRate, consumed, isToday);
      if (!burn) return null;

      const trend = gramsTrend(burn.grams);

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
        <div className="panel p-4">
          <div className="mb-3 flex items-center justify-between">
            <span className="eyebrow">{isToday ? "Live weight change" : "Final weight change"}</span>
            <span className="text-micro tabular-nums text-fg-subtle">
              {Math.round(burn.burnedSoFar)} burned
            </span>
          </div>

          <div className="flex items-start justify-between gap-4">
            {/* Left: live counter */}
            <div className="min-w-0">
              <div className={`figure flex items-center gap-1 text-stat ${trendColor[trend]}`}>
                <TrendArrow trend={trend} size={20} />
                {Math.abs(burn.grams).toFixed(3)}g
              </div>
              <div className={`mt-1 text-mini font-medium capitalize ${trendColor[trend]}`}>
                {trend}
              </div>
              <div className="mt-0.5 text-micro tabular-nums text-fg-subtle">
                Deficit {burn.deficit >= 0 ? "+" : ""}{Math.round(burn.deficit)} kcal
              </div>
            </div>

            {/* Right: forecasts. Each block states its own assumption, so the
                numbers are never a mix of two different scenarios. */}
            {isToday && (
              <div className="space-y-3 border-l border-line pl-4 text-right">
                <div className="space-y-1.5">
                  <div className="eyebrow">If you stop eating now</div>
                  <ForecastRow label="Today" grams={stopTodayGrams} />
                </div>

                {limit !== null ? (
                  <div className="space-y-1.5">
                    <div className="eyebrow">If you eat {limit} kcal/day</div>
                    <ForecastRow label="Today" grams={limitTodayGrams} />
                    <ForecastRow label="7 days" grams={weekGrams} />
                    <ForecastRow label="30 days" grams={monthGrams} />
                    {alreadyOverLimit && (
                      <div className="text-micro text-fg-subtle">Today already over limit</div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-1.5">
                    <div className="eyebrow">If every day like today</div>
                    <ForecastRow label="7 days" grams={weekGrams} />
                    <ForecastRow label="30 days" grams={monthGrams} />
                    <div className="text-micro text-fg-subtle">Set a limit for a real forecast</div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      );
    }

    // ── Add entry ────────────────────────────────────────────────

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
          <form onSubmit={handleSubmit(onSubmit)} className="composer">
            <input
              type="number"
              placeholder="kcal"
              {...register("kcal", { required: true, min: 1 })}
              disabled={mutation.isPending}
              className="composer-input w-[4.75rem] shrink-0 text-center tabular-nums"
            />
            <span aria-hidden="true" className="composer-rule" />
            <input
              type="text"
              placeholder="What did you eat?"
              {...register("description", { required: true, validate: (v) => v.trim().length > 0 })}
              disabled={mutation.isPending}
              className="composer-input min-w-0 flex-1"
            />
            <button
              type="submit"
              disabled={mutation.isPending || !isValid}
              aria-label="Add entry"
              className="btn btn-primary btn-icon btn-lg ml-1"
            >
              <Icon path={ICONS.plus} size={16} />
            </button>
          </form>
          <AppErrorMessage error={mutation.error} />
        </div>
      );
    }

    // ── Macros ───────────────────────────────────────────────────

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
          <span className="inline-flex items-center gap-1 text-mini text-fg-subtle" title="Estimating macros">
            <span className="inline-block h-1 w-1 animate-pulse rounded-full bg-current" />
            <span className="inline-block h-1 w-1 animate-pulse rounded-full bg-current [animation-delay:150ms]" />
            <span className="inline-block h-1 w-1 animate-pulse rounded-full bg-current [animation-delay:300ms]" />
          </span>
        );
      }

      if (entry.macros_state === "failed") {
        return (
          <button
            onClick={() => retry.mutate()}
            title="Macro estimate unavailable — click to retry"
            className="btn btn-sm btn-ghost -ml-1.5 gap-1 text-fg-subtle"
          >
            <Icon path={ICONS.retry} size={11} />
            Retry
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
          className={`inline-flex items-center gap-1.5 text-mini tabular-nums text-fg-subtle ${incomplete ? "cursor-pointer hover:text-fg" : ""}`}
        >
          {MACROS.map((m, i) => (
            <span key={m.key} className="inline-flex items-center gap-1">
              {i > 0 && <span className="text-line-strong">·</span>}
              <span className={i === 0 ? "font-medium text-fg-muted" : ""}>
                {fmtGrams(entry.macros[m.key])}
              </span>
              <span>{m.short}</span>
            </span>
          ))}
          {incomplete && <Icon path={ICONS.retry} size={11} />}
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
          className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-mini tabular-nums text-fg-muted"
          title={partial ? "Some entries have no macro estimate yet" : undefined}
        >
          {MACROS.map((m, i) => (
            <span key={m.key} className="inline-flex items-center gap-1">
              {i > 0 && <span className="text-line-strong">·</span>}
              <span className="font-medium text-fg">{fmtGrams(data.total_macros[m.key])}</span>
              <span>{m.label.toLowerCase()}</span>
            </span>
          ))}
          {partial && <span className="text-fg-subtle">*</span>}
        </div>
      );
    }

    // ── Entry row ────────────────────────────────────────────────

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
        <div className="row -mx-2 px-2 py-2">
          <div className="flex items-center justify-between gap-2">
            <div
              onClick={canExpand ? () => setExpanded((v) => !v) : undefined}
              className={`flex min-w-0 items-baseline gap-3 ${canExpand ? "cursor-pointer" : ""}`}
            >
              <span className="w-11 shrink-0 text-mini tabular-nums text-fg-subtle">{entry.time}</span>
              <span className="w-12 shrink-0 text-right text-sm font-medium tabular-nums text-fg">
                {entry.kcal}
              </span>
              <span className="break-words text-sm text-fg">{entry.description}</span>
              {canExpand && (
                <span className="shrink-0 self-center text-fg-subtle">
                  <Icon
                    path={ICONS.chevronDown}
                    size={12}
                    className={`transition-transform duration-200 ease-std ${expanded ? "rotate-180" : ""}`}
                  />
                </span>
              )}
            </div>
            <button
              onClick={() => mutation.mutate()}
              disabled={mutation.isPending}
              aria-label="Delete entry"
              className="btn btn-sm btn-ghost btn-icon row-action shrink-0 transition-opacity hover:text-danger"
            >
              <Icon path={ICONS.trash} size={13} />
            </button>
          </div>

          {/* Macros sit on their own line: three figures alongside the description
              would squeeze it onto several lines, and --macro-indent lines them up
              under it. */}
          <div className="empty:hidden" style={{ marginLeft: "var(--macro-indent)" }}>
            <MacroBadge entry={entry} date={date} />
          </div>

          {canExpand && expanded && (
            <div className="mt-2 ml-4 border-l border-line pl-3 text-mini text-fg-muted">
              <div className="flex gap-2 pb-1 text-fg-subtle">
                <span className="flex-1" />
                {MACROS.map((m) => (
                  <span key={m.key} className="w-12 shrink-0 text-right">{m.short}</span>
                ))}
              </div>
              {items.map((item, i) => (
                <div key={i} className="flex gap-2 py-0.5">
                  <span className="flex-1 break-words">{item.name}</span>
                  {MACROS.map((m) => (
                    <span key={m.key} className="w-12 shrink-0 text-right tabular-nums">
                      {fmtGrams(item[`${m.key}_g`])}
                    </span>
                  ))}
                </div>
              ))}
              <div className="mt-1 flex gap-2 border-t border-line pt-1 text-fg">
                <span className="flex-1 font-medium">Total</span>
                {MACROS.map((m) => (
                  <span key={m.key} className="w-12 shrink-0 text-right font-medium tabular-nums">
                    {fmtGrams(entry.macros[m.key])}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      );
    }

    // ── Stats cards ──────────────────────────────────────────────

    function CumulativeWeightChange() {
      const { data } = useSuspenseQuery({
        queryKey: statsKeys.cumulative,
        queryFn: () => kcalClient.getCumulative(),
        refetchInterval: 60000,
      });

      if (data.days_counted === 0) return null;

      const trend = gramsTrend(data.total_grams);
      const absGrams = Math.abs(data.total_grams);
      const display = absGrams >= 1000 ? (absGrams / 1000).toFixed(2) + "kg" : absGrams.toFixed(1) + "g";

      return (
        <div className="flex items-center justify-between gap-4 px-5 py-4">
          <div className="min-w-0">
            <div className="eyebrow">Net weight change</div>
            <div className="mt-1 text-mini text-fg-muted">
              {data.days_counted} days counted
              {data.days_excluded > 0 && ` · ${data.days_excluded} skipped`}
            </div>
          </div>
          <div className={`figure flex shrink-0 items-center gap-1 text-stat ${trendColor[trend]}`}>
            <TrendArrow trend={trend} size={18} />
            {display}
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
        <div className="space-y-3 px-5 py-4">
          <div className="flex items-center justify-between gap-4">
            <div className="min-w-0">
              <div className="eyebrow">Average daily intake</div>
              <div className="mt-1 text-mini text-fg-muted">
                {data.days_counted} of {data.days_requested} days with data
                {data.days_excluded > 0 && ` · ${data.days_excluded} skipped`}
              </div>
            </div>
            <div className="figure shrink-0 text-stat text-fg">
              {data.days_counted > 0 ? Math.round(data.average_kcal) : "—"}
              {data.days_counted > 0 && (
                <span className="ml-1 text-mini font-medium tracking-normal text-fg-subtle">kcal</span>
              )}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="seg">
              {presets.map((d) => (
                <button
                  key={d}
                  onClick={() => { setSelectedDays(d); setIsCustom(false); setCustomInput(""); }}
                  data-on={selectedDays === d && !isCustom}
                  className="seg-item"
                >
                  {d}d
                </button>
              ))}
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                const val = parseInt(customInput);
                if (val > 0) { setSelectedDays(Math.min(val, MAX_AVERAGE_DAYS)); setIsCustom(true); }
              }}
              className="ml-auto flex items-center gap-1.5"
            >
              <input
                type="number"
                value={customInput}
                onChange={(e) => setCustomInput(e.target.value)}
                placeholder="N"
                min={1}
                max={MAX_AVERAGE_DAYS}
                aria-label="Custom day range"
                className="field field-sm w-12 text-center tabular-nums"
              />
              <button type="submit" className="btn btn-sm btn-soft">Go</button>
            </form>
          </div>
        </div>
      );
    }

    // ── Day marks ────────────────────────────────────────────────

    // Each mark is its own toggle: pressing the active one clears it, pressing
    // the other switches. A day is never both at once.
    function DayMarkButtons({ mark, date }: { mark: DayMark | null; date: string }) {
      const invalidate = useInvalidateDayAndStats(date);
      const mutation = useMutation({
        mutationFn: (next: DayMark | null) => kcalClient.setDayMark({ mark: next, date }),
        onSuccess: invalidate,
      });

      const toggle = (target: DayMark) => mutation.mutate(mark === target ? null : target);

      // inline-flex so the pair takes its alignment from whatever holds it:
      // right, beside the limit and burn setters; left, under a marked day.
      return (
        <div className="inline-flex flex-col items-end gap-1">
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => toggle("cheat")}
              disabled={mutation.isPending}
              className={`btn btn-sm gap-1.5 ${
                mark === "cheat" ? "bg-caution-soft text-caution" : "btn-ghost"
              }`}
            >
              <Icon path={ICONS.pizza} size={12} />
              Cheat day
            </button>
            <button
              onClick={() => toggle("excluded")}
              disabled={mutation.isPending}
              className={`btn btn-sm gap-1.5 ${
                mark === "excluded" ? "bg-hover text-fg" : "btn-ghost"
              }`}
            >
              <Icon path={ICONS.ban} size={12} />
              Skip day
            </button>
          </div>
          <AppErrorMessage error={mutation.error} />
        </div>
      );
    }

    // A marked day replaces the counter entirely: its entries no longer decide
    // anything, so showing a total against a limit would only mislead.
    const MARKED_DAY_VIEW = {
      cheat: {
        headline: "Cheat day",
        icon: ICONS.pizza,
        detail: `Counts as ${CHEAT_DAY_KCAL} kcal`,
        entriesLabel: `Entries — ignored, day scored as ${CHEAT_DAY_KCAL}`,
        color: "text-caution",
        soft: "bg-caution-soft",
      },
      excluded: {
        headline: "Not counted",
        icon: ICONS.ban,
        detail: "Left out of the average and the weight change",
        entriesLabel: "Entries — kept, but not counted anywhere",
        color: "text-fg-muted",
        soft: "bg-hover",
      },
    } as const;

    // ── Day view ─────────────────────────────────────────────────

    function DayView({ date }: { date: string }) {
      const { data } = useSuspenseQuery({
        queryKey: dayKeys.day(date),
        queryFn: () => kcalClient.getDay(date),
        refetchInterval: dayRefetchInterval,
      });

      if (data.mark) {
        const view = MARKED_DAY_VIEW[data.mark];
        return (
          <div className="space-y-5">
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-center gap-3">
                <span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg ${view.soft} ${view.color}`}>
                  <Icon path={view.icon} size={18} />
                </span>
                <div>
                  <div className={`text-lg font-medium ${view.color}`}>{view.headline}</div>
                  <div className="mt-0.5 text-mini text-fg-muted">{view.detail}</div>
                </div>
              </div>
            </div>

            <DayMarkButtons mark={data.mark} date={date} />

            {data.entries.length > 0 && (
              <div className="border-t border-line pt-3 opacity-60">
                <div className="eyebrow mb-1">{view.entriesLabel}</div>
                {data.entries.map((entry) => (
                  <EntryItem key={entry.id} entry={entry} date={date} />
                ))}
              </div>
            )}
          </div>
        );
      }

      const status = getStatus(data.total, data.limit);
      const counterColor = statusTextColor[status];
      const isOver = status === "over";
      const isCurrentDay = date === todayStr();

      return (
        <div className="space-y-5">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex items-baseline gap-2">
                <span className={`figure text-hero ${counterColor || "text-fg"}`}>{data.total}</span>
                <span className="text-xs font-medium text-fg-subtle">kcal</span>
              </div>
              {isOver && (
                <div className="mt-1.5 inline-flex items-center gap-1.5 rounded-full bg-danger-soft px-2 py-0.5 text-micro font-medium uppercase tracking-label text-danger">
                  <Icon path={ICONS.flame} size={11} />
                  Over limit
                </div>
              )}
              <MacroTotals data={data} />
            </div>
            <div className="flex shrink-0 flex-col items-end gap-1">
              <LimitSetter currentLimit={data.limit} date={date} />
              <BurnSetter currentBurn={data.burn} date={date} />
            </div>
          </div>

          <ProgressBar total={data.total} limit={data.limit} />

          <LiveBurnCounter burnRate={data.burn} consumed={data.total} limit={data.limit} isToday={isCurrentDay} />

          <div className="flex justify-end">
            <DayMarkButtons mark={data.mark} date={date} />
          </div>

          <div className="border-t border-line pt-5">
            <AddEntryForm date={date} />
          </div>

          {data.entries.length === 0 ? (
            <div className="flex flex-col items-center gap-1 py-8 text-center">
              <p className="text-sm text-fg-muted">Nothing logged yet</p>
              <p className="text-mini text-fg-subtle">Add your first entry above</p>
            </div>
          ) : (
            <div className="border-t border-line pt-1">
              {data.entries.map((entry) => (
                <EntryItem key={entry.id} entry={entry} date={date} />
              ))}
            </div>
          )}
        </div>
      );
    }

    // ── Estimator ────────────────────────────────────────────────

    // The estimate contents (form + result). Lives inside the modal so it has
    // room to breathe; the breakdown table needs the width.
    function QuickEstimateBody({ onClose }: { onClose: () => void }) {
      const [input, setInput] = useState("");
      const mutation = useMutation({
        mutationFn: (description: string) => kcalClient.estimate(description),
      });

      const onSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        const description = input.trim();
        if (description) mutation.mutate(description);
      };

      return (
        <>
          <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-4">
            <div className="flex items-center gap-3">
              <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent-soft text-accent">
                <Icon path={ICONS.sparkles} size={17} />
              </span>
              <div>
                <h2 className="text-sm font-medium text-fg-strong">Kcal estimator</h2>
                <p className="text-mini text-fg-muted">Ask, don't track</p>
              </div>
            </div>
            <button onClick={onClose} aria-label="Close" className="btn btn-ghost btn-icon btn-md">
              <Icon path={ICONS.close} size={15} />
            </button>
          </div>

          <div className="space-y-4 overflow-y-auto p-5">
            <form onSubmit={onSubmit} className="space-y-2">
              <textarea
                autoFocus
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault();
                    onSubmit(e as unknown as React.FormEvent);
                  }
                }}
                placeholder="e.g. 955g red cabbage, 177g sausage, 952g cooked lentils"
                rows={3}
                disabled={mutation.isPending}
                className="field w-full resize-y"
              />
              <div className="flex items-center justify-between gap-3">
                <p className="text-micro text-fg-subtle">
                  <kbd>⌘</kbd> <kbd>↵</kbd> to ask
                </p>
                <button
                  type="submit"
                  disabled={mutation.isPending || input.trim().length === 0}
                  className="btn btn-primary btn-lg"
                >
                  {mutation.isPending ? "Thinking…" : "Ask"}
                </button>
              </div>
            </form>

            {mutation.data && !mutation.isPending && (
              <div className="panel p-4">
                <div className="flex items-baseline gap-2">
                  <span className="figure text-stat text-fg">{mutation.data.kcal}</span>
                  <span className="text-xs font-medium text-fg-subtle">kcal</span>
                </div>
                <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs tabular-nums text-fg-muted">
                  {MACROS.map((m, i) => (
                    <span key={m.key} className="inline-flex items-center gap-1">
                      {i > 0 && <span className="text-line-strong">·</span>}
                      <span className="font-medium text-fg">{fmtGrams(mutation.data.macros[m.key])}</span>
                      <span>{m.label.toLowerCase()}</span>
                    </span>
                  ))}
                </div>

                {mutation.data.items.length > 0 && (
                  <div className="mt-4 text-mini text-fg-muted">
                    <div className="flex gap-3 border-b border-line pb-1.5 text-fg-subtle">
                      <span className="flex-1">Item</span>
                      <span className="w-14 shrink-0 text-right">Kcal</span>
                      {MACROS.map((m) => (
                        <span key={m.key} className="w-12 shrink-0 text-right">{m.short}</span>
                      ))}
                    </div>
                    {mutation.data.items.map((item, i) => (
                      <div key={i} className="flex gap-3 border-b border-line py-1.5 last:border-b-0">
                        <span className="flex-1 break-words text-fg">{item.name}</span>
                        <span className="w-14 shrink-0 text-right tabular-nums">{item.kcal}</span>
                        {MACROS.map((m) => (
                          <span key={m.key} className="w-12 shrink-0 text-right tabular-nums">
                            {fmtGrams(item[`${m.key}_g`])}
                          </span>
                        ))}
                      </div>
                    ))}
                    <div className="mt-1 flex gap-3 border-t border-line-strong py-1.5 text-fg">
                      <span className="flex-1 font-medium">Total</span>
                      <span className="w-14 shrink-0 text-right font-medium tabular-nums">{mutation.data.kcal}</span>
                      {MACROS.map((m) => (
                        <span key={m.key} className="w-12 shrink-0 text-right font-medium tabular-nums">
                          {fmtGrams(mutation.data.macros[m.key])}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {mutation.data.note && (
                  <div className="mt-3 text-mini text-fg-subtle">{mutation.data.note}</div>
                )}
              </div>
            )}

            <AppErrorMessage error={mutation.error} />
          </div>
        </>
      );
    }

    // Standalone "how many kcal is this?" tool. A trigger button opens a wide
    // centered modal so the breakdown table has room. Never touches the tracked
    // day — type a food, ask, read the answer. Nothing is saved.
    function QuickEstimate() {
      const [open, setOpen] = useState(false);

      // Esc closes; remounting the body on each open clears the previous answer.
      // The page behind is frozen so a scroll gesture cannot drift it.
      useEffect(() => {
        if (!open) return;
        const h = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
        window.addEventListener("keydown", h);
        const previous = document.body.style.overflow;
        document.body.style.overflow = "hidden";
        return () => {
          window.removeEventListener("keydown", h);
          document.body.style.overflow = previous;
        };
      }, [open]);

      return (
        <>
          <button onClick={() => setOpen(true)} className="btn btn-outline btn-lg w-full gap-2">
            <Icon path={ICONS.sparkles} size={15} className="text-accent" />
            Kcal estimator
          </button>

          {open && (
            <div
              onClick={() => setOpen(false)}
              role="dialog"
              aria-modal="true"
              className="scrim fixed inset-0 flex items-start justify-center overflow-y-auto p-3 sm:items-center sm:p-6"
              style={{ zIndex: "var(--z-overlay)" }}
            >
              <div
                onClick={(e) => e.stopPropagation()}
                className="dialog my-auto flex max-h-[90dvh] w-full max-w-2xl flex-col"
              >
                <QuickEstimateBody onClose={() => setOpen(false)} />
              </div>
            </div>
          )}
        </>
      );
    }

    // ── Fallbacks ────────────────────────────────────────────────

    function ErrorFallback({ error, resetErrorBoundary }: { error: Error; resetErrorBoundary: () => void }) {
      return <AppErrorMessage error={error} retry={resetErrorBoundary} />;
    }

    function LoadingFallback() {
      return (
        <div className="flex items-center justify-center gap-2 py-12 text-fg-subtle">
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" className="animate-spin" aria-hidden="true">
            <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.5" opacity="0.25" />
            <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
          </svg>
          <span className="text-xs">Loading</span>
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
          <div className="relative min-h-[100dvh] bg-canvas text-fg">
            <ThemeToggle />

            <div
              className="mx-auto flex w-full flex-col gap-3 px-4 pb-10 pt-5 sm:gap-3.5 sm:pb-16 sm:pt-16"
              style={{ maxWidth: "var(--app-width)" }}
            >
              {/* Title */}
              <div className="flex h-8 items-center pr-12">
                <h1 className="text-micro font-medium uppercase tracking-title text-fg-muted">
                  Kcal Tracker
                </h1>
              </div>

              {/* Date navigation */}
              <div className="card flex items-center justify-between gap-2 px-2.5 py-2.5">
                <button
                  onClick={goBack}
                  aria-label="Previous day"
                  className="btn btn-ghost btn-icon btn-lg"
                >
                  <Icon path={ICONS.chevronLeft} size={17} />
                </button>

                <div className="min-w-0 text-center">
                  <div className="truncate text-sm font-medium text-fg">{formatDate(date)}</div>
                  <div className="mt-0.5 truncate text-mini tabular-nums text-fg-subtle">{dateSubtitle(date)}</div>
                </div>

                <div className="flex items-center gap-1">
                  {!isToday && (
                    <button onClick={goToday} className="btn btn-sm btn-soft">
                      Today
                    </button>
                  )}
                  <button
                    onClick={goForward}
                    aria-label="Next day"
                    className="btn btn-ghost btn-icon btn-lg"
                  >
                    <Icon path={ICONS.chevronRight} size={17} />
                  </button>
                </div>
              </div>

              {/* Body */}
              <div className="card p-5">
                <ErrorBoundary FallbackComponent={ErrorFallback} resetKeys={[date]}>
                  <Suspense fallback={<LoadingFallback />}>
                    <DayView date={date} />
                  </Suspense>
                </ErrorBoundary>
              </div>

              {/* Average intake */}
              <div className="card overflow-hidden">
                <ErrorBoundary FallbackComponent={ErrorFallback}>
                  <Suspense fallback={null}>
                    <AverageIntake />
                  </Suspense>
                </ErrorBoundary>
              </div>

              {/* Cumulative */}
              <div className="card overflow-hidden empty:hidden">
                <ErrorBoundary FallbackComponent={ErrorFallback}>
                  <Suspense fallback={null}>
                    <CumulativeWeightChange />
                  </Suspense>
                </ErrorBoundary>
              </div>

              {/* Quick estimator: a trigger button in the flow that opens a wide
                  centered modal, so the breakdown table has room to render. */}
              <div className="mt-1">
                <QuickEstimate />
              </div>

              {/* Footer */}
              <div className="mt-1 flex items-center justify-center gap-1.5 text-micro text-fg-subtle">
                <kbd>←</kbd>
                <kbd>→</kbd>
                <span>to change day</span>
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
    global repo, estimator, kcal_estimator
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    repo = SqliteKcalRepository(db)
    estimator = build_estimator()
    kcal_estimator = build_kcal_estimator()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
