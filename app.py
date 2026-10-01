from __future__ import annotations

import io

import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st

from generate_processes import make_processes, to_excel_bytes
from scheduler import (
    THROUGHPUT_COL,
    ScheduleResult,
    frame_from_raw,
    load_processes,
    processes_frame,
    run_all,
    summary_table,
)

st.set_page_config(page_title="CPU Scheduling Simulator", page_icon="⏱️", layout="wide")

PALETTE = pc.qualitative.Dark24 + pc.qualitative.Light24 + pc.qualitative.Alphabet
IDLE_COLOR = "#bdbdbd"
MAX_LABELLED_BARS = 60  # draw PID text inside bars only when this few are visible per row


# ----------------------------------------------------------------------------
# Data loading / simulation (cached)
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def sheet_names(data: bytes, filename: str) -> list[str]:
    if filename.lower().endswith(".csv"):
        return []
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


@st.cache_data(show_spinner=False)
def read_raw(data: bytes, filename: str, sheet: str | None) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(data), header=None)
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet or 0, header=None)


@st.cache_data(show_spinner="Running the schedulers…")
def simulate(data: bytes, filename: str, sheet: str | None, quantum: float):
    procs = load_processes(frame_from_raw(read_raw(data, filename, sheet)))
    results = run_all(procs, quantum)
    return processes_frame(procs), results, summary_table(results)


@st.cache_data(show_spinner=False)
def sample_workbook(seed: int) -> bytes:
    return to_excel_bytes({"Processes": make_processes(1500, seed=seed)})


# ----------------------------------------------------------------------------
# Gantt chart
# ----------------------------------------------------------------------------

def gantt_figure(results: list[ScheduleResult], pids: list[str], window: tuple[float, float]) -> go.Figure:
    lo, hi = window
    fig = go.Figure()
    boundaries: set = set()
    labelled = True

    for r in results:
        visible = [(idx, s, e) for idx, s, e in r.gantt if e > lo and s < hi]
        show_text = len(visible) <= MAX_LABELLED_BARS
        labelled &= show_text
        starts = [max(s, lo) for _, s, _ in visible]
        ends = [min(e, hi) for _, _, e in visible]
        boundaries.update(starts)
        boundaries.update(ends)
        names = [pids[idx] if idx is not None else "Idle" for idx, _, _ in visible]
        fig.add_trace(
            go.Bar(
                y=[r.name] * len(visible),
                base=starts,
                x=[e - s for s, e in zip(starts, ends)],
                orientation="h",
                marker=dict(
                    color=[IDLE_COLOR if idx is None else PALETTE[idx % len(PALETTE)] for idx, _, _ in visible],
                    line=dict(color="white", width=0.6),
                ),
                text=names if show_text else None,
                textposition="inside",
                insidetextanchor="middle",
                textangle=0,
                customdata=[[n, s, e, e - s] for n, (_, s, e) in zip(names, visible)],
                hovertemplate="<b>%{customdata[0]}</b><br>Start: %{customdata[1]}<br>End: %{customdata[2]}"
                "<br>Duration: %{customdata[3]}<extra>%{y}</extra>",
                showlegend=False,
            )
        )

    xaxis = dict(title="Time", range=[lo, hi], showgrid=True, zeroline=False)
    if labelled and len(boundaries) <= 45:
        # Textbook style: tick at every segment boundary
        ticks = sorted(boundaries)
        xaxis.update(tickmode="array", tickvals=ticks, ticktext=[str(t) for t in ticks], tickangle=-60 if len(ticks) > 20 else 0)

    fig.update_layout(
        barmode="overlay",
        bargap=0.25,
        height=140 + 70 * len(results),
        margin=dict(l=10, r=10, t=30, b=40),
        xaxis=xaxis,
        yaxis=dict(autorange="reversed"),
        uniformtext=dict(minsize=8, mode="hide"),
        dragmode="pan",
    )
    return fig


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

st.title("⏱️ CPU Scheduling Simulator")
st.caption("FCFS · SJF · SRTF (three tie-breakers) · Round Robin — upload an Excel table of processes to compare them.")

with st.sidebar:
    st.header("1. Process table")
    upload = st.file_uploader("Upload Excel / CSV", type=["xlsx", "xls", "csv"],
                              help="Needs columns: Arrival Time, Burst Time, Priority (Process / PID optional).")
    use_sample = st.toggle("Use a random 1500-process sample instead", value=False, disabled=upload is not None)
    sample_seed = st.number_input("Sample seed", min_value=0, value=42, step=1, disabled=upload is not None)
    st.download_button(
        "⬇️ Download a 1500-process sample (.xlsx)",
        data=sample_workbook(int(sample_seed)),
        file_name="processes_1500.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.header("2. Round Robin")
    quantum = st.number_input("Time quantum", min_value=1, value=4, step=1)

if upload is not None:
    data, filename = upload.getvalue(), upload.name
elif use_sample:
    data, filename = sample_workbook(int(sample_seed)), "sample.xlsx"
else:
    st.info(
        "Upload an Excel file with the columns **Arrival Time**, **Burst Time** and **Priority** "
        "(optionally **Process**) in the sidebar, or switch on the random sample.\n\n"
    )
    st.stop()

sheet = None
names = sheet_names(data, filename)
if len(names) > 1:
    sheet = st.sidebar.selectbox("Sheet", names)

try:
    proc_df, results, summary = simulate(data, filename, sheet, float(quantum))
except Exception as exc:  # show bad-input problems in the page instead of a traceback
    st.error(f"Could not read the process table: {exc}")
    st.stop()

pids = proc_df["Process"].tolist()

# --- Input overview ----------------------------------------------------------
st.subheader("Input")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Processes", f"{len(proc_df):,}")
c2.metric("Arrival range", f"{proc_df['Arrival Time'].min()} – {proc_df['Arrival Time'].max()}")
c3.metric("Burst range", f"{proc_df['Burst Time'].min()} – {proc_df['Burst Time'].max()}")
c4.metric("Priority range", f"{proc_df['Priority'].min()} – {proc_df['Priority'].max()}")
with st.expander("View uploaded processes"):
    st.dataframe(proc_df, width="stretch", height=300, hide_index=True)

# --- Comparison table --------------------------------------------------------
st.subheader("Results")
st.caption("Throughput = number of processes ÷ (last completion time − first arrival time).")
fmt = {c: "{:.2f}" for c in summary.columns if c.startswith("Avg")}
fmt[THROUGHPUT_COL] = "{:.4f}"
st.dataframe(summary.style.format(fmt), width="stretch", hide_index=True)

report = {"Summary": summary, "Processes": proc_df} | {r.name: r.per_process for r in results}
st.download_button(
    "⬇️ Download full report (.xlsx)",
    data=to_excel_bytes(report),
    file_name="scheduling_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# --- Gantt charts ------------------------------------------------------------
st.subheader("Gantt charts")
makespan = max(r.gantt[-1][2] for r in results)
integer_time = all(isinstance(e, int) for r in results for _, _, e in r.gantt)
default_hi = min(makespan, 60)
if integer_time:
    window = st.slider("Time window", 0, int(makespan), (0, int(default_hi)), step=1,
                       help="Narrow the window to read individual processes; drag/zoom inside the chart as well.")
else:
    window = st.slider("Time window", 0.0, float(makespan), (0.0, float(default_hi)))
if window[1] <= window[0]:
    st.warning("Pick a time window with a non-zero width.")
    st.stop()

tabs = st.tabs(["All algorithms"] + [r.name for r in results])
with tabs[0]:
    st.plotly_chart(gantt_figure(results, pids, window), width="stretch", key="gantt-all")

for i, (tab, r) in enumerate(zip(tabs[1:], results)):
    with tab:
        st.plotly_chart(gantt_figure([r], pids, window), width="stretch", key=f"gantt-{i}")
        s = summary.iloc[i]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg completion", f"{s['Avg Completion Time']:.2f}")
        m2.metric("Avg turnaround", f"{s['Avg Turnaround Time']:.2f}")
        m3.metric("Avg waiting", f"{s['Avg Waiting Time']:.2f}")
        m4.metric("Throughput", f"{s[THROUGHPUT_COL]:.4f}")
        st.dataframe(r.per_process, width="stretch", height=320, hide_index=True)
        st.download_button(
            f"⬇️ Download {r.name} table (.csv)",
            data=r.per_process.to_csv(index=False).encode(),
            file_name=f"{r.name}.csv",
            mime="text/csv",
            key=f"dl-{i}",
        )
