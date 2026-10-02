# The Archer's Paradox

Is it really paradox? Who knows!

## Interactive viewer

`arrow_viewer.py` is a PyQt6/pyqtgraph front end for `arrow_flight_sim.py`.
It integrates a shot on a worker thread and plays it back in two views at
once, driven by a single frame index so they cannot drift apart:

* **Side view** - x-y plane (range against height)
* **Top view** - x-z plane (range against sideways offset)

The left panel edits the initial position, the initial 3-D velocity (either
as speed + elevation/yaw or as an explicit vx/vy/vz vector), the shaft
attitude and roll, the body rates, the target and the checkpoints. Arrow
geometry, feathers, atmosphere and integrator settings live in a collapsed
"Advanced" group. The `playback speed` entry is a pure time scaling - `1.0`
is real time, `0.1` is ten times slower. The horizontal axes of the two
plots are linked, so zooming or panning either one moves the other.

Run it with:

```bash
python arrow_viewer.py
```

The shaft is drawn as a genuinely bent polyline, using the same first
bending mode the integrator itself uses in `_tip_state`, and the rendered
tip position and tip velocity match `_tip_state` exactly.
