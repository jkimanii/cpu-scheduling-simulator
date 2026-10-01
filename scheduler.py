"""CPU scheduling algorithms: FCFS, SJF, SRTF (with three tie-breakers) and Round Robin.

Pure logic, no UI. Every algorithm returns a ScheduleResult holding:
  * gantt       - list of (process_index | None, start, end); None marks CPU idle time
  * per_process - DataFrame with completion / turnaround / waiting / response times
"""
from __future__ import annotations

import heapq
import math
import re
from collections import deque
from dataclasses import dataclass

import pandas as pd

EPS = 1e-9


@dataclass(frozen=True)
class Process:
    pid: str
    arrival: float
    burst: float
    priority: float
    index: int  # original row order, the last-resort (FCFS) tie-break


@dataclass
class ScheduleResult:
    name: str
    gantt: list
    per_process: pd.DataFrame


# ----------------------------------------------------------------------------
# Loading the uploaded table
# ----------------------------------------------------------------------------

_ALIASES = {
    "pid": {"process", "processes", "pid", "processid", "processno", "id", "name", "job", "processname"},
    "arrival": {"arrivaltime", "arrival", "at", "arrivaltimes", "arrivetime"},
    "burst": {"bursttime", "burst", "bt", "cpuburst", "cpubursttime", "servicetime", "executiontime"},
    "priority": {"priority", "pr", "prio", "priorities"},
}
_REQUIRED = ("arrival", "burst", "priority")


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _match_columns(columns) -> dict:
    """Map our field names -> actual column names found in the table."""
    found = {}
    for col in columns:
        key = _norm(col)
        for field, aliases in _ALIASES.items():
            if key in aliases and field not in found:
                found[field] = col
    return found


def frame_from_raw(raw: pd.DataFrame, scan_rows: int = 15) -> pd.DataFrame:
    """Given a sheet read with header=None, locate the header row (it need not be row 1)."""
    for r in range(min(scan_rows, len(raw))):
        if all(f in _match_columns(raw.iloc[r].tolist()) for f in _REQUIRED):
            df = raw.iloc[r + 1:].copy()
            df.columns = raw.iloc[r].tolist()
            return df.reset_index(drop=True)
    raise ValueError(
        "Could not find a header row with 'Arrival Time', 'Burst Time' and 'Priority' columns "
        f"in the first {scan_rows} rows of the sheet."
    )


def _numeric(df: pd.DataFrame, col: str, label: str, header_offset: int) -> pd.Series:
    values = pd.to_numeric(df[col], errors="coerce")
    bad = values[values.isna()].index
    if len(bad):
        rows = ", ".join(str(i + header_offset) for i in bad[:10])
        raise ValueError(f"'{label}' has missing or non-numeric values (table rows {rows}{'…' if len(bad) > 10 else ''}).")
    if (values == values.round()).all():
        values = values.astype(int)
    return values


def load_processes(df: pd.DataFrame) -> list[Process]:
    cols = _match_columns(df.columns)
    missing = [f for f in _REQUIRED if f not in cols]
    if missing:
        raise ValueError(f"Missing column(s): {', '.join(missing)}. Found: {', '.join(map(str, df.columns))}")

    df = df.dropna(how="all", subset=[cols[f] for f in _REQUIRED]).reset_index(drop=True)
    if df.empty:
        raise ValueError("The table has no process rows.")

    offset = 2  # 1-based rows + header line, for friendlier error messages
    arrival = _numeric(df, cols["arrival"], "Arrival Time", offset)
    burst = _numeric(df, cols["burst"], "Burst Time", offset)
    priority = _numeric(df, cols["priority"], "Priority", offset)
    if (arrival < 0).any():
        raise ValueError("Arrival Time values must be >= 0.")
    if (burst <= 0).any():
        raise ValueError("Burst Time values must be > 0.")

    if "pid" in cols:
        pids = df[cols["pid"]].fillna("").astype(str).str.strip()
        pids = [p if p else f"P{i + 1}" for i, p in enumerate(pids)]
    else:
        pids = [f"P{i + 1}" for i in range(len(df))]

    return [
        Process(pids[i], arrival.iloc[i].item(), burst.iloc[i].item(), priority.iloc[i].item(), i)
        for i in range(len(df))
    ]


def processes_frame(procs: list[Process]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Process": [p.pid for p in procs],
            "Arrival Time": [p.arrival for p in procs],
            "Burst Time": [p.burst for p in procs],
            "Priority": [p.priority for p in procs],
        }
    )


# ----------------------------------------------------------------------------
# Tie-breakers (applied after the primary key: burst for SJF, remaining time for SRTF)
# ----------------------------------------------------------------------------

TIEBREAKERS = {
    "FCFS": lambda p: (p.arrival, p.index),
    "High int = high priority": lambda p: (-p.priority, p.arrival, p.index),
    "Low int = high priority": lambda p: (p.priority, p.arrival, p.index),
}
TIEBREAK_SHORT = {
    "FCFS": "FCFS tie",
    "High int = high priority": "High-int priority tie",
    "Low int = high priority": "Low-int priority tie",
}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _add_segment(gantt: list, idx, start, end) -> None:
    if end - start <= EPS:
        return
    if gantt and gantt[-1][0] == idx and abs(gantt[-1][2] - start) <= EPS:
        gantt[-1] = (idx, gantt[-1][1], end)
    else:
        gantt.append((idx, start, end))


def _by_arrival(procs: list[Process]) -> list[Process]:
    return sorted(procs, key=lambda p: (p.arrival, p.index))


def _clean(x):
    """Round away float noise and give back ints where possible."""
    if isinstance(x, float):
        r = round(x, 9)
        return int(r) if r == int(r) else r
    return x


def _result(name: str, procs: list[Process], gantt: list, first_start: dict, completion: dict) -> ScheduleResult:
    gantt = [(i, _clean(s), _clean(e)) for i, s, e in gantt]
    rows = []
    for p in procs:
        ct = _clean(completion[p.index])
        tat = _clean(ct - p.arrival)
        rows.append(
            {
                "Process": p.pid,
                "Arrival Time": p.arrival,
                "Burst Time": p.burst,
                "Priority": p.priority,
                "Start Time": _clean(first_start[p.index]),
                "Completion Time": ct,
                "Turnaround Time": tat,
                "Waiting Time": _clean(tat - p.burst),
                "Response Time": _clean(first_start[p.index] - p.arrival),
            }
        )
    return ScheduleResult(name, gantt, pd.DataFrame(rows))


# ----------------------------------------------------------------------------
# Algorithms
# ----------------------------------------------------------------------------

def fcfs(procs: list[Process]) -> ScheduleResult:
    gantt, first_start, completion = [], {}, {}
    t = 0
    for p in _by_arrival(procs):
        if t < p.arrival:
            _add_segment(gantt, None, t, p.arrival)
            t = p.arrival
        first_start[p.index] = t
        _add_segment(gantt, p.index, t, t + p.burst)
        t += p.burst
        completion[p.index] = t
    return _result("FCFS", procs, gantt, first_start, completion)


def sjf(procs: list[Process], tiebreak: str = "FCFS") -> ScheduleResult:
    """Non-preemptive Shortest Job First."""
    key = TIEBREAKERS[tiebreak]
    pending = _by_arrival(procs)
    by_idx = {p.index: p for p in procs}
    gantt, first_start, completion = [], {}, {}
    heap, i, t, n = [], 0, 0, len(pending)

    while i < n or heap:
        while i < n and pending[i].arrival <= t:
            p = pending[i]
            heapq.heappush(heap, ((p.burst, *key(p)), p.index))
            i += 1
        if not heap:
            _add_segment(gantt, None, t, pending[i].arrival)
            t = pending[i].arrival
            continue
        _, idx = heapq.heappop(heap)
        p = by_idx[idx]
        first_start[idx] = t
        _add_segment(gantt, idx, t, t + p.burst)
        t += p.burst
        completion[idx] = t

    return _result(f"SJF ({TIEBREAK_SHORT[tiebreak]})", procs, gantt, first_start, completion)


def srtf(procs: list[Process], tiebreak: str = "FCFS") -> ScheduleResult:
    """Preemptive Shortest Remaining Time First (event driven: decisions happen at arrivals/completions)."""
    key = TIEBREAKERS[tiebreak]
    pending = _by_arrival(procs)
    by_idx = {p.index: p for p in procs}
    remaining = {p.index: p.burst for p in procs}
    gantt, first_start, completion = [], {}, {}
    heap, i, t, n = [], 0, 0, len(pending)

    while i < n or heap:
        while i < n and pending[i].arrival <= t:
            p = pending[i]
            heapq.heappush(heap, ((remaining[p.index], *key(p)), p.index))
            i += 1
        if not heap:
            _add_segment(gantt, None, t, pending[i].arrival)
            t = pending[i].arrival
            continue
        _, idx = heapq.heappop(heap)
        first_start.setdefault(idx, t)
        next_arrival = pending[i].arrival if i < n else math.inf
        run = min(remaining[idx], next_arrival - t)
        _add_segment(gantt, idx, t, t + run)
        t += run
        remaining[idx] -= run
        if remaining[idx] <= EPS:
            completion[idx] = t
        else:
            heapq.heappush(heap, ((remaining[idx], *key(by_idx[idx])), idx))

    return _result(f"SRTF ({TIEBREAK_SHORT[tiebreak]})", procs, gantt, first_start, completion)


def round_robin(procs: list[Process], quantum: float) -> ScheduleResult:
    """Round Robin. Processes arriving during a time slice join the queue before the preempted process."""
    if quantum <= 0:
        raise ValueError("Time quantum must be > 0.")
    pending = _by_arrival(procs)
    remaining = {p.index: p.burst for p in procs}
    gantt, first_start, completion = [], {}, {}
    queue, i, t, n = deque(), 0, 0, len(pending)

    while i < n or queue:
        while i < n and pending[i].arrival <= t:
            queue.append(pending[i].index)
            i += 1
        if not queue:
            _add_segment(gantt, None, t, pending[i].arrival)
            t = pending[i].arrival
            continue
        idx = queue.popleft()
        first_start.setdefault(idx, t)
        run = min(quantum, remaining[idx])
        _add_segment(gantt, idx, t, t + run)
        t += run
        remaining[idx] -= run
        while i < n and pending[i].arrival <= t:
            queue.append(pending[i].index)
            i += 1
        if remaining[idx] <= EPS:
            completion[idx] = t
        else:
            queue.append(idx)

    q = _clean(float(quantum))
    return _result(f"Round Robin (q={q})", procs, gantt, first_start, completion)


def run_all(procs: list[Process], quantum: float) -> list[ScheduleResult]:
    results = [fcfs(procs)]
    results += [sjf(procs, tb) for tb in TIEBREAKERS]
    results += [srtf(procs, tb) for tb in TIEBREAKERS]
    results.append(round_robin(procs, quantum))
    return results


# ----------------------------------------------------------------------------
# Summary metrics
# ----------------------------------------------------------------------------

THROUGHPUT_COL = "Throughput (processes / time unit)"


def summarize(result: ScheduleResult) -> dict:
    df = result.per_process
    n = len(df)
    span = df["Completion Time"].max() - df["Arrival Time"].min()
    return {
        "Algorithm": result.name,
        "Avg Arrival Time": df["Arrival Time"].mean(),
        "Avg Completion Time": df["Completion Time"].mean(),
        "Avg Turnaround Time": df["Turnaround Time"].mean(),
        "Avg Waiting Time": df["Waiting Time"].mean(),
        "Avg Response Time": df["Response Time"].mean(),
        THROUGHPUT_COL: n / span if span > 0 else float("nan"),
        "Total Time": _clean(float(df["Completion Time"].max())),
        "Context Switches": sum(1 for idx, _, _ in result.gantt if idx is not None) - 1,
    }


def summary_table(results: list[ScheduleResult]) -> pd.DataFrame:
    return pd.DataFrame([summarize(r) for r in results])
