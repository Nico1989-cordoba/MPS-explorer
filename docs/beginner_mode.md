# Beginner mode

A guided tour of the main window, an "Explain this" mode and a simulated example axon, for somebody opening MPS
Explorer for the first time. The cards are in Spanish or English (Help → *Idioma de las explicaciones* / *Language of
the explanations*); the rest of the interface stays in English.

## How to use it

- **The tour.** On the first start, the main window opens with its tour: one card at a time, beside the control it
  explains. The rest of the window is blurred; the control stays sharp and usable. **Next** (or →, Enter),
  **Previous** (or ←), **Skip** (or Esc). The first card has **Do not show again**, which turns beginner mode off.
- **Each card** has four parts - *what it is*, *what it is for*, *how to read it*, *an example* - and **More**, which
  unfolds the long explanation and, for a parameter, the parameter registry's own text. **See the full method** opens
  a page of this repository when the step has one.
- **Help** (the last button of the toolbar):
  - *Beginner mode* (on/off). Turning it on shows every window's tour again, starting with the main window's.
  - *Tour this window* replays the whole tour.
  - *Explain this* (**Shift+F1**): the cursor changes, and the next click on a control shows its card instead of
    running it - no button fires, no field changes. A control without a card gets a short note with its tooltip.
    Esc cancels.
  - *Open the example axon (simulated)* loads `examples/demo_axon/demo_axon_simulated.hdf5` into channel 1, so the
    tour can be followed on the program's real output.
- **"?"** in the top-right corner of each panel (files, view, clustering, measurement) replays only that panel's cards.

Beginner mode, the language and which windows' tours have been seen are kept in the settings file
(`mps_analysis_settings.json`, field `beginner_tour`). A tour opens by itself only the first time its window is
shown while beginner mode is on, and never on an offscreen Qt platform (the golden harness and the GUI tests) unless
`MPS_TOUR_AUTOSTART=1`; `MPS_TOUR_AUTOSTART=0` turns that off everywhere.

## Where the texts come from

- `tools/mps_tour_steps.py` holds every text, in both languages, one list of steps per window.
- A step about a parameter the registry documents (`tools/mps_param_registry.py`) names its registry keys. The card
  quotes the registry's default and source through placeholders (`{eps_default}`, `{paper}`, ...), read when the card
  is drawn, and **More** shows the registry's own tooltip text - so a value is never typed twice and cannot drift.
- Every example number comes from the simulated example axon and is written as a placeholder from `DEMO_FACTS`
  (`{n_locs}`, `{n_clusters}`, ...); every example says it is a simulated example. `test_beginner_tour_gui.py`
  re-measures each fact on the program's own output for that file. No number comes from real or unpublished data.
- The example axon is written by `tools/make_demo_axon.py` (see `examples/demo_axon/README.md`).

## How to add a step

1. Give the widget an `objectName` if it has none (a widget built in code: `widget.setObjectName("...")`; a toolbar
   action's button: `MPS_explorer._TOOLBAR_NAMES`).
2. Add a `_step(key, target, panel, es={...}, en={...})` to the window's list in `tools/mps_tour_steps.py`, in the
   order a beginner meets it. Fill the six fields (`title`, `what`, `why`, `read`, `example`, `more`) in both
   languages; name `registry=(...)` keys when the step is about a registered parameter, and `method="path/in/repo"`
   when a page says more.
3. Numbers: quote the registry through a placeholder, and the example axon through `DEMO_FACTS` (add the fact, and
   its check in `test_beginner_tour_gui.py`, section 7).
4. Run `venv\Scripts\python.exe test_beginner_tour_gui.py`: it fails when a target does not name exactly one visible
   widget, when a field is empty in either language, or when an example does not say it is simulated.

A new window gets its tour with `controller.register_window(window, "<id>")` and a `STEPS["<id>"]` list. The MPS
analysis window is registered as `"mps_analysis"` with an empty list: its tour waits for the science changes that
settle its rows.
