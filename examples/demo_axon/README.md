# Example axon (SIMULATED)

`demo_axon_simulated.hdf5` and its Picasso sidecar `demo_axon_simulated.yaml` are **one simulated axon, not a
measurement**. The beginner tour opens it (Help → *Simulated axon to demonstrate the analysis capabilities of MPS Explorer*) so that every example number it
quotes comes from a file anyone can regenerate.

How it was generated:

    venv\Scripts\python.exe tools\make_demo_axon.py

which calls this project's simulator (`tools.mps_simulate_axon.simulate_axon`, through
`batch_columns.write_simulated_input`) with **contour `circle`, seed `39`, axial-leak scale `0.3`** - the "viable" case
of the Z-quality demo and axon A of the UI golden - and writes it in the Picasso format the program reads: x, y and
the lateral precisions in camera pixels, z and lpz in nanometres, a pixel size of **130 nm** in the YAML. The
`photons`, `sx`, `sy` and `bg` columns are constant placeholders. The output is byte-for-byte reproducible.

12,137 localizations over 60,000 frames; about 0.5 MB.

## What it demonstrates

Because it is the "viable" case of the Z-quality demo, **both consecutive ring pairs are VIABLE under rule v2**: the
column analysis runs on it (Rings... -> Columns: lumen review... -> Apply and re-run analyses) and the *Two rings
(unrolled)...* view opens. The real axons of the April set have no viable pair (0 of 33), so this file is how those
parts of the program are shown. Nothing in it is a measurement.
