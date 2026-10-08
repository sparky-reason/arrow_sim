# The Archer's Paradox

Is it really paradox? Who knows!

Read how it's simulated here [arrow_shot_simulation_explanation.md](arrow_shot_simulation_explanation.md)

## Interactive viewer

`arrow_viewer.py` is a PyQt6/pyqtgraph front end for `arrow_shot_simulator.py`.
It integrates a whole shot on a worker thread and plays it back in two views
at once, driven by a single frame index so they cannot drift apart:

* **Side view** - x-y plane (range against height)
* **Top view** - x-z plane (range against sideways offset)

Playback covers the entire shot, not just the free flight: the backend's
launch phase (the ~22 ms draw and release stroke, solved as a flexible beam)
is prepended to the flight on one continuous time axis, so Play starts at
full draw and shows the arrow accelerating off the string before it flies.

The left panel edits only what describes *this* bow on *this* shot - draw
strength, draw length, bow cant, release azimuth, nock height offset, arrow
side offset, arrow-rest longitudinal offset, the two lateral release
disturbances, string roll, plus arrow length and spine. Everything else
(shaft diameters, mass distribution, fletching, atmosphere, integrator
settings, the target and the checkpoints) is held at the values in
`arrow_shot_simulator.example()`, and the "Fixed by the simulation" section
lists them. Defaults therefore reproduce the simulator's own example shot
exactly.

A full solve takes roughly twenty seconds (a 2 us maximum step through the
launch stroke, then the flight), which is why the default `playback speed` is
well below real time - the launch is only a few percent of the simulated
duration. The `playback speed` and `frame rate` entries are view-only and do
not affect the solve.

Run it with:

```bash
python arrow_viewer.py
```

The shaft is drawn as a genuinely bent polyline: the launch uses the beam
solver's own nodal shape, and the flight uses the same first bending mode the
integrator itself uses in `_tip_state`.

## Tests

```bash
python viewer_shot_test.py    # headless: builds a shot, checks the phase join
python viewer_gui_test.py     # offscreen Qt: panel wiring and field defaults
python viewer_window_test.py  # offscreen Qt: presses Start, waits for the solve
```
