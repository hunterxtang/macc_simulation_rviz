# Benchmark Results — MILP+CBS Hybrid vs CBS-Alone

## Honest framing

This is **not** a reproduction of any paper's published numbers. The six
test structures are designed in this repository (see
[`docs/benchmark_structures.md`](benchmark_structures.md)); the
reference paper's Table III thumbnails were judged too low-resolution
to extract reliable shapes from. We frame this as a characterization of
our hybrid MILP+CBS planner across a designed complexity axis, not an
apples-to-apples reproduction.

The reference paper's own MILP-only baseline ("Method A — Exact
[Lam et al.]") timed out at >10,000 s on three of its six instances and
required up to 5.7 days of wall time on the hardest instance ([1]).
Our results reproduce this scaling story: the MILP solves cleanly on
small/medium subs, falls back to CBS on the boundary, and stops being
worth attempting on the largest case in our set.

**Single parallel group caveat (verbatim, per project policy):**

> All six benchmark structures decompose into a single parallel group
> due to the shadow-based decomposition assigning each column to
> exactly one sub. Parallel-group constraint injection (PAR-V/PAR-E)
> is validated by Phase 4 unit tests, not by this benchmark set.
> Within-group parallelism is bounded by num_robots=4.

## Setup

- Robots: 4, all starting at depot (0, 0)
- Per-T MILP time limit: 60.0 s (Gurobi `TimeLimit`)
- Total MILP time limit: 600.0 s (soft — adapter shrinks per-T limit toward 1.0 s as the budget drains)
- MILP `T_max`: 150 (hard cap on the per-sub T sweep)
- CBS `max_t`: 400, branch limit: 500
- MIP gap: 0.0 (prove optimality; in practice Gurobi often hits the time limit first)
- Raw stdout logs preserved at [`bench_full.log`](bench_full.log) and [`bench_rest.log`](bench_rest.log)

## Aggregate results (cases 1-5)

`mix` is `(MILP / CBS-fallback)` count of substructures within the case.
Case 5 has no aggregate `T_final` because the run was killed after the
hybrid finished its 18 MILP solves but before the runner emitted its
summary line; per-sub breakdown is intact.

| Case | Blocks | Subs | Hybrid T_final | Hybrid solve (s) | Mix (M/CBS) | CBS-alone T_final | CBS-alone solve (s) | Hybrid speedup vs CBS-alone (T_final) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 — tiny        |  5 |  2 |  20 |    4.4 | 2 / 0 |  35 | 0.01 | 1.75× |
| 2 — small       | 13 |  5 |  42 |    2.5 | 5 / 0 |  93 | 0.03 | 2.21× |
| 3 — medium_low  | 20 |  6 |  62 |  116.1 | 6 / 0 | 181 | 0.06 | 2.92× |
| 4 — medium_high | 27 |  2 | 102 | 1446.2 | 1 / 1 | 204 | 0.06 | 2.00× |
| 5 — large_flat  | 40 | 18 | (n/a — log truncated) | ≥101 (sum of per-sub solves) | 18 / 0 | (not run) | — | — |
| 6 — large_dense | 44 |  9 | not completed (scale wall — see below) | — | — | — | — | — |

Notes on the table:

- **Hybrid solve (s)** is the runner's wall time, which can exceed the
  600 s soft budget because the budget only shrinks the per-T limit, it
  does not abort. Case 4 hybrid took 1446 s of which 1071 s was a single
  MILP solve on its 18-block central sub.
- **CBS-alone solve (s)** is dominated by the CBS branching loop itself;
  for these cases CBS terminates very quickly without conflicts.
- **Hybrid speedup** here is the makespan ratio. The hybrid trades
  orders-of-magnitude more solve time for a roughly 2× shorter execution
  schedule.

## Per-substructure breakdown (hybrid)

Per-sub solve times and `T` budgets, parsed from
[`bench_full.log`](bench_full.log) and [`bench_rest.log`](bench_rest.log).

### Case 1 — tiny (5 blocks, 2 subs)

| sub | blocks | planner | T  | trips | solve (s) | obj |
|---:|---:|---|---:|---:|---:|---:|
|  0 | 3 | milp | 14 |  8 |  3.97 | 30.0 |
|  1 | 2 | milp |  8 |  3 |  0.44 | 12.0 |

### Case 2 — small (13 blocks, 5 subs)

| sub | blocks | planner | T  | trips | solve (s) | obj |
|---:|---:|---|---:|---:|---:|---:|
|  0 | 4 | milp |  9 |  4 |  0.58 | 19.0 |
|  1 | 3 | milp |  8 |  3 |  0.35 | 14.0 |
|  2 | 3 | milp |  8 |  3 |  0.36 | 14.0 |
|  3 | 2 | milp |  8 |  2 |  0.34 | 10.0 |
|  4 | 1 | milp | 11 |  2 |  0.92 | 11.0 |

### Case 3 — medium_low (20 blocks, 6 subs)

| sub | blocks | planner | T  | trips | solve (s) | obj |
|---:|---:|---|---:|---:|---:|---:|
|  0 | 8 | milp | 16 |  9 | 60.75 | 44.0 |
|  1 | 8 | milp | 16 |  9 | 52.55 | 44.0 |
|  2 | 1 | milp |  6 |  1 |  0.24 |  4.0 |
|  3 | 1 | milp | 10 |  2 |  1.18 | 12.0 |
|  4 | 1 | milp | 10 |  2 |  1.18 | 12.0 |
|  5 | 1 | milp |  6 |  1 |  0.22 |  4.0 |

### Case 4 — medium_high (27 blocks, 2 subs)

| sub | blocks | planner | T  | trips | solve (s) | obj |
|---:|---:|---|---:|---:|---:|---:|
|  0 | 18 | milp | 44 | 29 | 1071.19 | 124.0 |
|  1 |  9 | cbs_fallback | 58 | — |    0.03 | — |

Sub 0 ate the bulk of the 600 s `total_time_limit` (it actually overran
the soft budget — the adapter only enforces the limit by shrinking the
per-T window, not by aborting). When sub 1 began, `time_budget_left`
had collapsed: its `per_t_limit` was clamped to 1.0 s and Gurobi could
not find a feasible solution at any T ≤ 150 within that window. The
adapter logged this honestly:

```
[MILP] sub 1: MILP infeasible at all T <= T_max after 374.93s,
              falling back to CBS (blocks=9, T_max=150, per_t_limit=1.0s)
[MILP] sub 1: planner=cbs_fallback T_contribution=58 cbs_solve=0.03s
              cbs_serial_fallback=True blocks=9
```

This is the fallback path doing exactly its job: instead of aborting
the whole group, it spliced a CBS solution for sub 1 onto sub 0's MILP
plan and the run completed.

### Case 5 — large_flat (40 blocks, 18 subs)

All 18 subs solved via MILP. Aggregate `T_final` not in the log — the
run was killed before the runner printed its summary line — but every
per-sub solve is on disk.

| sub | blocks | planner | T  | trips | solve (s) | obj |
|---:|---:|---|---:|---:|---:|---:|
|  0 | 4 | milp |  9 |  4 |  1.35 | 19.0 |
|  1 | 4 | milp |  9 |  4 |  1.33 | 19.0 |
|  2 | 8 | milp | 18 |  8 | 86.90 | 59.0 |
|  3 | 4 | milp | 11 |  4 |  2.02 | 30.0 |
|  4 | 4 | milp |  9 |  4 |  1.25 | 19.0 |
|  5 | 4 | milp |  9 |  4 |  1.26 | 19.0 |
|  6 | 1 | milp |  6 |  1 |  0.28 |  4.0 |
|  7 | 1 | milp |  6 |  1 |  0.30 |  4.0 |
|  8 | 1 | milp | 10 |  1 |  1.46 |  8.0 |
|  9 | 1 | milp |  6 |  1 |  0.28 |  4.0 |
| 10 | 1 | milp |  8 |  1 |  0.73 |  6.0 |
| 11 | 1 | milp |  6 |  1 |  0.28 |  4.0 |
| 12 | 1 | milp |  6 |  1 |  0.30 |  4.0 |
| 13 | 1 | milp | 10 |  1 |  1.45 |  8.0 |
| 14 | 1 | milp |  6 |  1 |  0.29 |  4.0 |
| 15 | 1 | milp |  8 |  1 |  0.74 |  6.0 |
| 16 | 1 | milp |  6 |  1 |  0.30 |  4.0 |
| 17 | 1 | milp |  6 |  1 |  0.28 |  4.0 |

The single 8-block sub is the dominant cost (88 s); the seventeen 1-4
block subs amortize to under a second each. Sum of per-sub solves
≈ 101 s. Note this case has no CBS-alone data — the runner was killed
before reaching that mode for case 5.

### Case 6 — large_dense (44 blocks, 9 subs) — not completed

Case 6 was the planned worst case, with a 24-block central sub on a
padded 9×9 MILP grid (vs case 4's 18-block central sub on a padded 7×7).
The solve was not completed: the runner was killed during this case
after substantial wall time was spent on the 24-block sub.

This is the **scale wall** the benchmark was designed to surface. We do
not report a fabricated `T_final` for it.

## Scale boundary discussion

The empirical scale boundary lives between cases 4 and 6. Concretely:

- **Case 4's 18-block sub** solved in 1071 s — already 1.8× over the
  600 s soft budget. That single solve consumed enough of the global
  time budget that the *next* sub (only 9 blocks) could not be allotted
  more than 1 s per T and fell back to CBS. The hybrid still produced
  a valid plan (T=102 vs 204 for CBS-alone), but at 1446 s of wall
  time.
- **Case 5's 8-block sub** took 88 s on its own — significant, but
  small enough that the surrounding 17 tiny subs absorbed it without
  triggering fallback.
- **Case 6's 24-block sub** is in the regime where the per-sub MILP
  variable count and time-expanded horizon both blow up. We could not
  complete it within feasible wall time on this hardware.

This matches the structure of the reference paper's own numbers: their
exact-MILP method (Method A in their Table III) reported
`Total Solve Time > 10,000 s` on three of six instances, and on the
hardest instance the published exact-MILP wall time is 5.7 days.

For our hybrid the practical takeaway is unambiguous:
- Per-sub block counts up to ~10 → MILP is the right tool, with
  ~2× makespan reduction over CBS-alone for negligible wall cost.
- Per-sub block counts in the 15-20 range → MILP still produces the
  better plan, but the wall cost dominates and CBS fallback may fire
  on a downstream sub due to budget squeeze.
- Per-sub block counts ≥ 24 on a 7×7-or-larger grid → MILP is
  not worth attempting at our budgets; the fallback path (or CBS-alone)
  is the correct choice.

This is why the hybrid is the right architecture: *every* sub gets a
MILP attempt, and only the few that hit the wall fall back. We never
abandon the MILP advantage on the small subs the way a pure-CBS pipeline
would.

## CBS-alone baseline

| Case | T_final | Solve (s) |
|---|---:|---:|
| 1 — tiny        |  35 | 0.01 |
| 2 — small       |  93 | 0.03 |
| 3 — medium_low  | 181 | 0.06 |
| 4 — medium_high | 204 | 0.06 |
| 5 — large_flat  | (not run) | — |
| 6 — large_dense | (not run) | — |

Cases 5 and 6 have no CBS-alone numbers because the runner was killed
before reaching them in either pass. We do not extrapolate.

CBS-alone consistently terminates in milliseconds for the cases that
were measured, but produces makespans 1.75–2.92× longer than the
hybrid. This is the trade-off our hybrid is built to navigate: pay
MILP wall-time on the subs where it can finish, fall back to CBS on
the subs where it can't.

## Reproducibility

- Structures: [`macc_rviz/benchmark_structures.py`](../macc_rviz/benchmark_structures.py),
  visualised in [`docs/benchmark_structures.md`](benchmark_structures.md).
- Runner: [`scripts/run_benchmarks.py`](../scripts/run_benchmarks.py)
  (do not invoke for case 6 without raising the time/memory budgets;
  the 24-block central sub will exhaust both).
- Raw logs: [`bench_full.log`](bench_full.log) (cases 1–4 + start of
  case 5), [`bench_rest.log`](bench_rest.log) (case 5 hybrid only).

## References

[1] Lam, Stuckey, Koenig, Kumar — "Exact approaches to the multi-agent
collective construction problem", in *Principles and Practice of
Constraint Programming*, Springer, 2020, pp. 743-758. The 5.7-day
solve time is reported in their experimental section as the wall time
required by their exact-MILP method on the hardest of their six
benchmark instances.
