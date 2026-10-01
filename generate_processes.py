"""Generate a random process table (Excel) for the scheduling simulator.

Usage:
    python generate_processes.py                      # 1500 processes -> processes.xlsx
    python generate_processes.py -n 1500 --seed 42 -o my_processes.xlsx
"""
from __future__ import annotations

import argparse
import io

import numpy as np
import pandas as pd


def make_processes(
    n: int = 1500,
    max_arrival: int = 1000,
    burst_range: tuple[int, int] = (1, 20),
    priority_range: tuple[int, int] = (1, 10),
    seed: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Process": [f"P{i}" for i in range(1, n + 1)],
            "Arrival Time": rng.integers(0, max_arrival, endpoint=True, size=n),
            "Burst Time": rng.integers(burst_range[0], burst_range[1], endpoint=True, size=n),
            "Priority": rng.integers(priority_range[0], priority_range[1], endpoint=True, size=n),
        }
    )


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=1500, help="number of processes (default 1500)")
    ap.add_argument("--max-arrival", type=int, default=1000, help="arrival times are 0..MAX (default 1000)")
    ap.add_argument("--burst", type=int, nargs=2, default=(1, 20), metavar=("MIN", "MAX"))
    ap.add_argument("--priority", type=int, nargs=2, default=(1, 10), metavar=("MIN", "MAX"))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("-o", "--output", default="processes.xlsx")
    args = ap.parse_args()

    df = make_processes(args.n, args.max_arrival, tuple(args.burst), tuple(args.priority), args.seed)
    with open(args.output, "wb") as f:
        f.write(to_excel_bytes({"Processes": df}))
    print(f"Wrote {len(df)} processes to {args.output}")


if __name__ == "__main__":
    main()
