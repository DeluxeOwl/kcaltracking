#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "fastapi",
#     "pydantic",
#     "uvicorn",
# ]
# ///

import sqlite3
import uvicorn
from abc import ABC, abstractmethod
from datetime import datetime

import click

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# ── Domain ────────────────────────────────────────────────────────────

class KcalEntry:
    def __init__(self, kcal: int, description: str, entry_date: str, created_at: str | None = None) -> None:
        self.kcal = kcal
        self.description = description
        self.entry_date = entry_date
        self.created_at = created_at or datetime.now().strftime("%H:%M")
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


class SqliteKcalRepository(KcalRepository):
    def __init__(self, db_path: str = "kcal.db") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
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
        self._conn.commit()

    def add_entry(self, entry: KcalEntry) -> KcalEntry:
        cur = self._conn.execute(
            "INSERT INTO entries (kcal, description, entry_date, created_at) VALUES (?, ?, ?, ?)",
            (entry.kcal, entry.description, entry.entry_date, entry.created_at),
        )
        self._conn.commit()
        entry.id = cur.lastrowid
        return entry

    def delete_entry(self, entry_id: int) -> None:
        cur = self._conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(f"No entry with id {entry_id}")

    def list_entries(self, entry_date: str) -> list[KcalEntry]:
        rows = self._conn.execute(
            "SELECT id, kcal, description, entry_date, created_at FROM entries "
            "WHERE entry_date = ? ORDER BY id",
            (entry_date,),
        ).fetchall()
        entries: list[KcalEntry] = []
        for row_id, kcal, description, d, created_at in rows:
            e = KcalEntry(kcal, description, d, created_at)
            e.id = row_id
            entries.append(e)
        return entries

    def get_limit(self, entry_date: str) -> int | None:
        row = self._conn.execute(
            "SELECT limit_kcal FROM daily_limits "
            "WHERE entry_date <= ? ORDER BY entry_date DESC LIMIT 1",
            (entry_date,),
        ).fetchone()
        return row[0] if row else None

    def set_limit(self, entry_date: str, limit_kcal: int) -> None:
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

    def cumulative_weight_change(self) -> dict:
        """Compute cumulative weight change across all completed (non-skipped) days."""
        KCAL_PER_GRAM_FAT = 7.7

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
        total_grams = 0.0
        day_details = []

        for entry_date, consumed in rows:
            if entry_date in skipped_set:
                continue
            if entry_date >= today:
                # Skip today and future days (not yet complete)
                continue

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
            })

        return {
            "total_grams": round(total_grams, 3),
            "days_counted": len(day_details),
            "days": day_details,
        }


# ── Schemas ───────────────────────────────────────────────────────────

class AddEntryRequest(BaseModel):
    kcal: int
    description: str
    date: str
    time: str | None = None

class SetLimitRequest(BaseModel):
    limit: int
    date: str

class SetBurnRequest(BaseModel):
    burn: int
    date: str

class SetSkippedRequest(BaseModel):
    skipped: bool
    date: str

class EntryResponse(BaseModel):
    id: int
    kcal: int
    description: str
    time: str

class DayResponse(BaseModel):
    date: str
    limit: int | None
    burn: int | None
    total: int
    skipped: bool
    entries: list[EntryResponse]


# ── API ───────────────────────────────────────────────────────────────

api = APIRouter(prefix="/api")
repo: SqliteKcalRepository


@api.get("/days/{day}", response_model=DayResponse)
async def get_day(day: str):
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
        skipped=skipped,
        entries=[EntryResponse(id=e.id, kcal=e.kcal, description=e.description, time=e.created_at) for e in entries],
    )


@api.post("/entries", response_model=EntryResponse, status_code=201)
async def add_entry(body: AddEntryRequest):
    entry = KcalEntry(body.kcal, body.description, body.date, created_at=body.time)
    repo.add_entry(entry)
    return EntryResponse(id=entry.id, kcal=entry.kcal, description=entry.description, time=entry.created_at)


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

  <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>

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
      "react": "https://esm.sh/react@19",
      "react/jsx-runtime": "https://esm.sh/react@19/jsx-runtime",
      "react/jsx-dev-runtime": "https://esm.sh/react@19/jsx-dev-runtime",
      "react-dom": "https://esm.sh/react-dom@19",
      "react-dom/client": "https://esm.sh/react-dom@19/client",
      "shadcn": "https://esm.sh/shadcn-ui-bundled/standalone",
      "@tanstack/react-query": "https://esm.sh/@tanstack/react-query@5?deps=react@19",
      "react-error-boundary": "https://esm.sh/react-error-boundary?deps=react@19",
      "ky": "https://esm.sh/ky",
      "react-hook-form": "https://esm.sh/react-hook-form?deps=react@19"
    }
  }
  </script>

  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <script>
    Babel.registerPreset("tsx-auto", {
      presets: [
        [Babel.availablePresets["react"], { runtime: "automatic" }],
        [
          Babel.availablePresets["typescript"], 
          { 
            // Replace isTSX and allExtensions with this:
            ignoreExtensions: true 
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

    interface Entry {
      id: number;
      kcal: number;
      description: string;
      time: string;
    }

    interface DayData {
      date: string;
      limit: number | null;
      burn: number | null;
      total: number;
      skipped: boolean;
      entries: Entry[];
    }

    // ── Query keys ───────────────────────────────────────────────

    const dayKeys = {
      day: (date: string) => ["day", date] as const,
    } as const;

    // ── API client ──────────────────────────────────────────────

    const api = ky.create({ prefix: "/api" });

    const kcalClient = {
      getDay: (date: string) => api.get(`days/${date}`).json<DayData>(),
      addEntry: (data: { kcal: number; description: string; date: string; time: string }) =>
        api.post("entries", { json: data }).json<Entry>(),
      deleteEntry: (id: number) => api.delete(`entries/${id}`),
      setLimit: (data: { limit: number; date: string }) =>
        api.put("limits", { json: data }).json<{ date: string; limit: number }>(),
      setBurn: (data: { burn: number; date: string }) =>
        api.put("burns", { json: data }).json<{ date: string; burn: number }>(),
      setSkipped: (data: { skipped: boolean; date: string }) =>
        api.put("skip", { json: data }).json<{ date: string; skipped: boolean }>(),
      getCumulative: () =>
        api.get("cumulative").json<{ total_grams: number; days_counted: number }>(),
    } as const;

    // ── Helpers ──────────────────────────────────────────────────

    function todayStr(): string {
      return new Date().toISOString().split("T")[0];
    }

    function shiftDate(dateStr: string, days: number): string {
      const d = new Date(dateStr + "T12:00:00");
      d.setDate(d.getDate() + days);
      return d.toISOString().split("T")[0];
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

    function secondsSinceMidnight(): number {
      const now = new Date();
      return now.getHours() * 3600 + now.getMinutes() * 60 + now.getSeconds();
    }

    function useLiveBurn(burnRate: number | null, consumed: number, isToday: boolean) {
      const [now, setNow] = useState(Date.now());

      useEffect(() => {
        if (!isToday || burnRate === null) return;
        const id = setInterval(() => setNow(Date.now()), 100);
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

    function useErrorMessage(error: Error | null): string | null {
      const [message, setMessage] = useState<string | null>(null);

      useEffect(() => {
        if (!error) { setMessage(null); return; }
        if (error instanceof HTTPError) {
          error.response
            .json()
            .then((body: any) => setMessage(body?.detail ?? error.message))
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
      onSave: (val: number) => void;
      isPending: boolean;
      error: Error | null;
    }) {
      const [editing, setEditing] = useState(false);
      const { register, handleSubmit, reset } = useForm<{ value: string }>({
        defaultValues: { value: current?.toString() ?? "" },
      });

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
          onSubmit={handleSubmit(({ value }) => { onSave(parseInt(value)); setEditing(false); })}
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
          onSave={(v) => mutation.mutate(v)}
          isPending={mutation.isPending}
          error={mutation.error}
        />
      );
    }

    function BurnSetter({ currentBurn, date }: { currentBurn: number | null; date: string }) {
      const queryClient = useQueryClient();
      const mutation = useMutation({
        mutationFn: (burn: number) => kcalClient.setBurn({ burn, date }),
        onSuccess: () => queryClient.invalidateQueries({ queryKey: dayKeys.day(date) }),
      });
      return (
        <InlineSetter
          label="BURN"
          current={currentBurn}
          placeholder="2200"
          onSave={(v) => mutation.mutate(v)}
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

    function LiveBurnCounter({ burnRate, consumed, isToday }: { burnRate: number | null; consumed: number; isToday: boolean }) {
      const burn = useLiveBurn(burnRate, consumed, isToday);
      if (!burn) return null;

      const losing = burn.grams > 0;
      const gaining = burn.grams < 0;
      const colorClass = losing ? "text-emerald-600" : gaining ? "text-red-500" : "";

      // Forecast: "if you don't eat anymore" → full day deficit, repeated
      const endOfDayDeficit = burnRate! - consumed;
      const eodGrams = endOfDayDeficit / KCAL_PER_GRAM_FAT;
      const weekGrams = eodGrams * 7;
      const monthGrams = eodGrams * 30;
      const eodLosing = eodGrams > 0;

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

            {/* Right: forecast */}
            {isToday && (
              <div className="text-right space-y-1.5 border-l border-muted pl-4">
                <div className="text-[10px] tracking-wider text-muted-foreground mb-2">IF YOU STOP EATING</div>
                <div className="flex items-baseline justify-end gap-2">
                  <span className="text-[10px] tracking-wider text-muted-foreground">TODAY</span>
                  <span className={`text-sm font-bold tabular-nums ${eodLosing ? "text-emerald-600" : "text-red-500"}`}>
                    {eodLosing ? "↓" : "↑"}{formatGrams(eodGrams)}
                  </span>
                </div>
                <div className="flex items-baseline justify-end gap-2">
                  <span className="text-[10px] tracking-wider text-muted-foreground">7 DAYS</span>
                  <span className={`text-sm font-bold tabular-nums ${eodLosing ? "text-emerald-600" : "text-red-500"}`}>
                    {eodLosing ? "↓" : "↑"}{formatGrams(weekGrams)}
                  </span>
                </div>
                <div className="flex items-baseline justify-end gap-2">
                  <span className="text-[10px] tracking-wider text-muted-foreground">30 DAYS</span>
                  <span className={`text-sm font-bold tabular-nums ${eodLosing ? "text-emerald-600" : "text-red-500"}`}>
                    {eodLosing ? "↓" : "↑"}{formatGrams(monthGrams)}
                  </span>
                </div>
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
      const queryClient = useQueryClient();

      const mutation = useMutation({
        mutationFn: (data: { kcal: number; description: string; date: string }) =>
          kcalClient.addEntry(data),
        onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: dayKeys.day(date) });
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

    function EntryItem({ entry, date }: { entry: Entry; date: string }) {
      const queryClient = useQueryClient();

      const mutation = useMutation({
        mutationFn: () => kcalClient.deleteEntry(entry.id),
        onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: dayKeys.day(date) });
        },
      });

      return (
        <div className="group flex items-center justify-between py-3 border-b border-muted last:border-b-0">
          <div className="flex items-baseline gap-3">
            <span className="text-[11px] tabular-nums text-muted-foreground w-11 shrink-0">{entry.time}</span>
            <span className="text-sm font-bold tabular-nums w-12 text-right shrink-0">{entry.kcal}</span>
            <span className="text-sm">{entry.description}</span>
          </div>
          <button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            className="text-xs text-muted-foreground hover:text-destructive transition-colors disabled:opacity-50"
          >
            {mutation.isPending ? "..." : "DEL"}
          </button>
        </div>
      );
    }

    function CumulativeWeightChange() {
      const { data } = useSuspenseQuery({
        queryKey: ["cumulative"],
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
            <div className="text-[10px] tracking-wider text-muted-foreground">TOTAL BURNED SINCE START</div>
            <div className="text-[10px] tracking-wider text-muted-foreground">{data.days_counted} DAYS COUNTED</div>
          </div>
          <div className={`text-2xl font-bold tabular-nums tracking-tight ${colorClass}`}>
            {losing ? "↓" : gaining ? "↑" : ""} {display}
          </div>
        </div>
      );
    }

    function SkipDayToggle({ skipped, date }: { skipped: boolean; date: string }) {
      const queryClient = useQueryClient();
      const mutation = useMutation({
        mutationFn: (newSkipped: boolean) => kcalClient.setSkipped({ skipped: newSkipped, date }),
        onSuccess: () => queryClient.invalidateQueries({ queryKey: dayKeys.day(date) }),
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
                  NOT COUNTING
                </div>
              </div>
              <SkipDayToggle skipped={data.skipped} date={date} />
            </div>

            {data.entries.length > 0 && (
              <div className="border-t border-muted pt-1 opacity-50">
                <div className="text-[10px] tracking-wider text-muted-foreground mb-2">ENTRIES (NOT COUNTED)</div>
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
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <LimitSetter currentLimit={data.limit} date={date} />
                    <BurnSetter currentBurn={data.burn} date={date} />
                    <SkipDayToggle skipped={data.skipped} date={date} />
                  </div>
                </div>

                <ProgressBar total={data.total} limit={data.limit} />

                <LiveBurnCounter burnRate={data.burn} consumed={data.total} isToday={isCurrentDay} />
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
                <ErrorBoundary FallbackComponent={ErrorFallback}>
                  <Suspense fallback={<LoadingFallback />}>
                    <DayView date={date} />
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
    global repo
    repo = SqliteKcalRepository(db)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
