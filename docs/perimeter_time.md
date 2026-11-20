# The perimeter's time with many clusters, and the question before it

The MPS analysis builds the axon's contour by joining the kept cluster centres and refining the tour with 2-opt
**from every start**, keeping the shortest (`tools.mps_geometry.reconstruct_perimeter(all_starts=True)`; the
default since 2026-09-19). That is one 2-opt per cluster, so the cost grows roughly with the cube of the number of
clusters: seconds with the 100-200 clusters DBSCAN finds on a typical axon, minutes when a method leaves several
hundred small clusters (the Emitters layer with a small ξ, DBSCAN on a large axon). With DBSCAN and the other fast
methods the analysis runs in the window's own thread and the program does not respond meanwhile; with a slow method
(Emitters, OPTICS, voronoi_csr) the MPS analysis window runs it in the background, with the time elapsed and a
Cancel button that also stops the 2-opt between two starts (F5d, `tools/mps_background_run.py`).

## What the program does

* **Interactive (MPS analysis, its re-runs).** When the estimate below reaches one minute, a dialog appears
  *before* the perimeter is built, with the number of clusters and the estimated time:
  * **Continue (every start)** — the analysis as always;
  * **One start (faster)** — the contour refined from one start, as this program did before 2026-09-19. It is
    not a new algorithm: the export records it (`contour_2opt` = `one start`) and the tour can come out a few
    percent longer. The axoplasm panel's discard comparison still builds its contours from every start;
  * **Cancel** — nothing is computed; the previous analysis stays, marked as not current.
* **In the background** (a slow method in the MPS analysis window, F5d): the analysis runs in a worker thread and
  the question is relayed to the window's thread, which shows the same dialog; the worker waits for the answer
  (and stops waiting if the run is cancelled meanwhile).
* **Not interactive** (the batch, the command-line validators, the tests, an offscreen session, any other worker
  thread): nothing is asked and the analysis runs from every start. Its *Analysis notes* gain one line,
  `perimeter: N clusters, estimated about X min from every start (...)`.
* Below the threshold nothing changes, and nothing is written.

## Where each number comes from

| Number | Value | Origin |
|---|---|---|
| time from every start | 8.772 × 10⁻⁷ × N^2.887 s | [simulation of this project] fit (log-log least squares through the slower geometry at each N) to `docs/sims/perimeter/perimeter_time.jsonl`, written by `measure_perimeter_time.py` beside it: synthetic centres on an ellipse of 3000 × 2200 nm, radial scatter 25 and 60 nm, N = 100, 300, 600, 1000, one process, one thread, one laptop (2026-10-19). Every point lies within 0.9-1.3 × the fit |
| time from one start | 1.113 × 10⁻⁶ × N^2.470 s | [simulation of this project] the same timings |
| threshold | 60 s of estimate, i.e. 518 clusters | [ad hoc] one minute: shorter waits are not mistaken for a frozen window. 300 clusters, the other candidate, would ask before ~13 s |

Measured times (seconds; scatter 25 / 60 nm):

| N | every start | one start |
|---|---|---|
| 100 | 0.2 / 0.5 | 0.06 / 0.09 |
| 300 | 7.5 / 14.8 | 1.1 / 1.9 |
| 600 | 55 / 89 | 5.9 / 7.4 |
| 1000 | 234 / 380 | 27 / 22 |

The estimate is an order of magnitude, not a promise: it depends on the machine and on how much the contour folds.
On the simulated axon of `docs/emitters.md` (Emitters, learned ξ, about 1,040 kept clusters) the perimeter took
646 s where the fit gives about 450 s (1.4 ×).

## Code

`tools/mps_perimeter_cost.py` (the estimate, the threshold, the note and the gate, no Qt),
`tools/perimeter_cost_ui.py` (the dialog, shown only on a real screen and on the GUI thread),
`tools/mps_analysis.py` (`analyze_axon` asks just before the contour is built). Tests: `test_perimeter_cost.py`.
