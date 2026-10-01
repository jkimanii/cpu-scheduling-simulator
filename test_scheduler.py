"""Hand-checked tests for scheduler.py.  Run:  python test_scheduler.py"""
from generate_processes import make_processes
from scheduler import Process, fcfs, load_processes, round_robin, run_all, sjf, srtf, summarize, TIEBREAKERS


def mk(rows):
    """rows: list of (arrival, burst, priority)"""
    return [Process(f"P{i + 1}", a, b, pr, i) for i, (a, b, pr) in enumerate(rows)]


def completions(result):
    return result.per_process["Completion Time"].tolist()


def order(result):
    return [idx for idx, _, _ in result.gantt if idx is not None]


# Classic Silberschatz example: P1(0,8) P2(1,4) P3(2,9) P4(3,5)
CLASSIC = mk([(0, 8, 1), (1, 4, 1), (2, 9, 1), (3, 5, 1)])


def test_fcfs():
    assert completions(fcfs(CLASSIC)) == [8, 12, 21, 26]


def test_sjf():
    # P1 0-8, then P2 8-12, P4 12-17, P3 17-26
    assert completions(sjf(CLASSIC)) == [8, 12, 26, 17]


def test_srtf():
    r = srtf(CLASSIC)
    assert completions(r) == [17, 5, 26, 10]
    assert r.per_process["Waiting Time"].mean() == 6.5
    assert r.gantt == [(0, 0, 1), (1, 1, 5), (3, 5, 10), (0, 10, 17), (2, 17, 26)]


def test_round_robin():
    # q=4: P1 0-4, P2 4-8, P3 8-12, P4 12-16, P1 16-20, P3 20-24, P4 24-25, P3 25-26
    assert completions(round_robin(CLASSIC, 4)) == [20, 8, 26, 25]


def test_rr_new_arrival_before_preempted():
    # P2 arrives exactly when P1's slice ends, so P2 runs before P1 resumes
    r = round_robin(mk([(0, 3, 1), (2, 2, 1)]), 2)
    assert r.gantt == [(0, 0, 2), (1, 2, 4), (0, 4, 5)]


def test_idle_gap():
    r = fcfs(mk([(2, 3, 1), (10, 1, 1)]))
    assert r.gantt == [(None, 0, 2), (0, 2, 5), (None, 5, 10), (1, 10, 11)]


def test_tiebreakers():
    # All arrive at 0 with equal bursts; priorities P1=2, P2=3, P3=1
    procs = mk([(0, 5, 2), (0, 5, 3), (0, 5, 1)])
    expected = {"FCFS": [0, 1, 2], "High int = high priority": [1, 0, 2], "Low int = high priority": [2, 0, 1]}
    for tb, exp in expected.items():
        assert order(sjf(procs, tb)) == exp, (tb, order(sjf(procs, tb)))
        assert order(srtf(procs, tb)) == exp, (tb, order(srtf(procs, tb)))


def test_srtf_equal_remaining_preemption_by_priority():
    # P1(0,4,pri1) runs; at t=2 P2 arrives with burst 2 == P1's remaining.
    procs = mk([(0, 4, 1), (2, 2, 5)])
    assert order(srtf(procs, "FCFS")) == [0, 1]  # FCFS: P1 keeps the CPU
    assert order(srtf(procs, "High int = high priority")) == [0, 1, 0]  # P2 (pri 5) preempts
    assert order(srtf(procs, "Low int = high priority")) == [0, 1]


def test_invariants_on_1500_random():
    procs = load_processes(make_processes(1500, seed=7))
    total_burst = sum(p.burst for p in procs)
    mean_at = sum(p.arrival for p in procs) / len(procs)
    for r in run_all(procs, 4):
        busy = {}
        prev_end = 0
        for idx, s, e in r.gantt:
            assert s == prev_end, f"{r.name}: gap/overlap at {s}"
            prev_end = e
            if idx is not None:
                assert s >= procs[idx].arrival, f"{r.name}: P{idx} runs before arrival"
                busy[idx] = busy.get(idx, 0) + (e - s)
        assert sum(busy.values()) == total_burst, r.name
        assert all(busy[p.index] == p.burst for p in procs), r.name
        s = summarize(r)
        assert abs(s["Avg Arrival Time"] - mean_at) < 1e-9
        assert (r.per_process["Turnaround Time"] >= r.per_process["Burst Time"]).all()


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\nAll {len(tests)} tests passed.")
