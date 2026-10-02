"""Interactive PyQt6 viewer for :mod:`arrow_flight_sim`.

The panel on the left edits the shot -- initial position, initial 3-D
velocity, shaft attitude, the target and the remaining model parameters --
and the two plots on the right show the resulting flight at the same time,
once from the side (x-y plane) and once from above (x-z plane).

How the two views stay in step
------------------------------
Both plots are instances of :class:`ArrowView` and are handed *the same*
:class:`Trajectory` object and *the same* frame index from one playback
timer, so they show the same instant by construction rather than by two
synchronised clocks.  Their horizontal (range) axes are additionally linked,
so panning or zooming either view moves the other.

A note on the bending shape
---------------------------
The shaft is drawn as a real polyline, not as a rigid segment: for every
frame the axial stations ``x_i`` are mapped through

    p_i = r_cm + R(q) @ [x_i - x_cm, q_y * phi(x_i), q_z * phi(x_i)]

which is exactly the centreline the integrator itself uses in
``_tip_state``.  The arrow therefore bends with the first bending mode of
the shaft rather than with a cosmetic sine wave.

Why the solve runs on a thread
------------------------------
A 20 m shot costs roughly ten seconds of CPU with the module's default
``dt`` (dense output plus 1 ms maximum step), so :func:`simulate` is handed
to a worker thread and the window stays responsive while it works.

Run it with::

    python arrow_viewer.py
"""

import sys
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.spatial.transform import Rotation

from arrow_flight_sim import Arrow, Atmosphere, Feather, Target, simulate


# Antialiasing is a per-item option, so it has to be set before any item is
# built.  A light background keeps both plots readable, as in viewer_pg.py.
pg.setConfigOptions(antialias=True, background="w", foreground="k")


# Stored integration steps kept for playback.  The solver may return tens of
# thousands of steps; more than this is invisible on screen and only costs
# time in the setData calls.
MAX_FRAMES = 2500

# Axial stations used to draw the bent centreline.  41 is smooth at any
# reasonable zoom without making the per-frame arrays large.
SHAFT_NODES = 41

# Colours, shared by both views so the two plots stay visually consistent.
C_PATH = (170, 170, 170)      # whole flight, not yet flown
C_TRAIL = (216, 132, 24)      # flown part
C_SHAFT = (25, 62, 160)       # arrow shaft
C_TIP = (200, 40, 40)         # arrow tip
C_NOCK = (60, 60, 60)         # nock end
C_VELOCITY = (25, 145, 95)    # tip velocity vector
C_TARGET = (200, 40, 40)
C_IMPACT = (150, 30, 150)
C_AXIS = (110, 110, 110)
C_CHECK = (120, 120, 200)

# ==========================================
# Parameter schema
# ==========================================
@dataclass(frozen=True)
class Field:
    """One numeric parameter shown as a labelled spin box.

    Defaults are the values from the ``__main__`` block of
    :mod:`arrow_flight_sim`, so the window opens on the same shot that the
    module's own example runs.
    """

    name: str
    label: str
    default: float
    minimum: float
    maximum: float
    step: float = 0.1
    decimals: int = 2


INITIAL_FIELDS: Sequence[Field] = (
    Field("x0", "position x0 [m]", 0.0, -100.0, 100.0, 0.5, 3),
    Field("y0", "position y0 [m]", 1.5, -5.0, 20.0, 0.05, 3),
    Field("z0", "position z0 [m]", 0.0, -20.0, 20.0, 0.05, 3),
    Field("speed", "speed [m/s]", 55.0, 0.0, 200.0, 0.5, 2),
    Field("elev", "elevation [deg]", 2.0, -90.0, 90.0, 0.1, 3),
    Field("yaw", "yaw [deg]", 0.0, -180.0, 180.0, 0.1, 3),
    Field("vx", "velocity vx [m/s]", 55.0, -200.0, 200.0, 0.5, 3),
    Field("vy", "velocity vy [m/s]", 2.0, -200.0, 200.0, 0.5, 3),
    Field("vz", "velocity vz [m/s]", 0.0, -200.0, 200.0, 0.5, 3),
    Field("shaft_elev", "shaft elevation [deg]", 2.0, -90.0, 90.0, 0.1, 3),
    Field("shaft_yaw", "shaft yaw [deg]", 0.0, -180.0, 180.0, 0.1, 3),
    Field("roll", "roll about shaft [deg]", 15.0, -180.0, 180.0, 1.0, 2),
    Field("omega_pitch", "body rate pitch [rad/s]", 0.2, -200.0, 200.0, 0.1, 3),
    Field("omega_yaw", "body rate yaw [rad/s]", -0.1, -200.0, 200.0, 0.1, 3),
    Field("spin", "body spin [rad/s]", 80.0, -5000.0, 5000.0, 5.0, 1),
)

TARGET_FIELDS: Sequence[Field] = (
    Field("target_distance", "distance x [m]", 20.0, 0.05, 500.0, 0.5, 3),
    Field("target_height", "height y [m]", 1.25, -10.0, 50.0, 0.05, 3),
    Field("target_z", "sideways z [m]", 0.0, -20.0, 20.0, 0.05, 3),
    Field("target_radius", "radius [m]", 0.25, 0.0, 5.0, 0.01, 3),
)

PLAYBACK_FIELDS: Sequence[Field] = (
    # Playback speed is a pure time scaling: 1.0 is real time, 0.1 is ten
    # times slower, 2.0 twice as fast.
    Field("speed_mult", "playback speed (x real time)", 1.0, 0.01, 5.0, 0.05, 3),
    Field("fps", "frame rate [fps]", 60.0, 5.0, 500.0, 5.0, 0),
    Field("vscale", "side view height gain (x)", 1.0, 1.0, 50.0, 1.0, 2),
)

ADVANCED_FIELDS: Sequence[Field] = (
    Field("length", "arrow length [m]", 0.75, 0.05, 2.0, 0.01, 4),
    Field("shaft_od", "shaft outer dia [m]", 0.0065, 0.0001, 0.05, 0.0001, 5),
    Field("shaft_id", "shaft inner dia [m]", 0.0045, 0.0, 0.05, 0.0001, 5),
    Field("mass", "total mass [kg]", 0.028, 0.001, 2.0, 0.001, 5),
    Field("point_mass", "point mass [kg]", 0.009, 0.0, 1.0, 0.001, 5),
    Field("point_x", "point mass at x [m]", 0.75, 0.0, 2.0, 0.01, 4),
    Field("spine", "spine", 1000.0, 1.0, 100000.0, 10.0, 1),
    Field("feather_x0", "feather start x [m]", 0.08, 0.0, 2.0, 0.005, 4),
    Field("feather_x1", "feather end x [m]", 0.19, 0.0, 2.0, 0.005, 4),
    Field("feather_h", "feather height [m]", 0.012, 0.0001, 0.1, 0.001, 5),
    Field("feather_area", "feather area each [m2]", 0.00075, 0.00001, 0.01, 0.00005, 6),
    Field("feather_count", "feather count", 3.0, 0.0, 6.0, 1.0, 0),
    Field("feather_cant", "feather cant [deg]", 2.0, -90.0, 90.0, 0.5, 2),
    Field("feather_mass", "feather mass each [kg]", 0.0005, 0.0, 0.05, 0.0001, 6),
    Field("pressure", "pressure [Pa]", 101325.0, 1000.0, 200000.0, 100.0, 1),
    Field("temperature", "temperature [K]", 288.15, 150.0, 400.0, 0.5, 2),
    Field("dt", "max solver step [s]", 0.001, 0.00002, 0.05, 0.0002, 6),
    Field("t_end", "max flight time [s]", 5.0, 0.1, 60.0, 0.5, 2),
    Field("ground_y", "ground y [m]", 0.0, -10.0, 50.0, 0.05, 3),
    Field("wind_x", "wind x [m/s]", 0.0, -50.0, 50.0, 0.5, 3),
    Field("wind_y", "wind y [m/s]", 0.0, -50.0, 50.0, 0.5, 3),
    Field("wind_z", "wind z [m/s]", 0.0, -50.0, 50.0, 0.5, 3),
)

ALL_FIELDS: dict[str, Field] = {
    f.name: f
    for group in (INITIAL_FIELDS, TARGET_FIELDS, PLAYBACK_FIELDS, ADVANCED_FIELDS)
    for f in group
}

# Checkpoints are a list rather than a fixed set of spin boxes; the default
# matches the example in arrow_flight_sim.py.
DEFAULT_CHECKPOINTS = "5, 10, 15, 20"


# ==========================================
# Small helpers
# ==========================================
def direction_from_angles(elev_deg: float, yaw_deg: float) -> np.ndarray:
    """Unit direction for an elevation above +X and a yaw towards +Z.

    World axes are +X forward, +Y up, +Z sideways, so yaw is measured from
    +X towards +Z and both angles are given in degrees.
    """
    e = np.radians(elev_deg)
    a = np.radians(yaw_deg)
    return np.array([np.cos(e) * np.cos(a), np.sin(e), np.cos(e) * np.sin(a)])


def angles_from_vector(v) -> tuple[float, float]:
    """Inverse of :func:`direction_from_angles`, returned as degrees."""
    v = np.asarray(v, dtype=float)
    return (
        float(np.degrees(np.arctan2(v[1], np.hypot(v[0], v[2])))),
        float(np.degrees(np.arctan2(v[2], v[0]))),
    )


# ==========================================
# Trajectory
# ==========================================
@dataclass
class Trajectory:
    """Everything the views need, extracted once from a :class:`FlightResult`.

    The solver state is ``[r(3), v(3), quat(4), omega(3), q_y, qd_y, q_z,
    qd_z]``.  All frame data below is computed in one vectorized pass, so
    playback only ever slices pre-computed arrays.
    """

    t: np.ndarray               # (N,)      time [s]
    cm: np.ndarray              # (N, 3)    centre of mass [m]
    tip: np.ndarray             # (N, 3)    tip position [m]
    nock: np.ndarray            # (N, 3)    nock position [m]
    shaft: np.ndarray           # (N, K, 3) bent centreline [m]
    tip_velocity: np.ndarray    # (N, 3)    tip velocity [m/s]
    speed: np.ndarray           # (N,)      CM speed [m/s]
    result: object = field(repr=False, default=None)

    @classmethod
    def from_result(cls, arrow: Arrow, result, max_frames: int = MAX_FRAMES):
        sol = result.solution
        t = np.asarray(sol.t, dtype=float)
        y = np.asarray(sol.y, dtype=float)

        # Uniformly thin the stored steps; the solver's own steps are
        # non-uniform, so index-selecting on a linspace keeps a constant
        # fraction of every moment in the flight rather than sampling evenly
        # in index space.
        if t.size > max_frames:
            idx = np.unique(np.linspace(0, t.size - 1, max_frames).astype(int))
            t = t[idx]
            y = y[:, idx]

        n = t.size
        cm = y[0:3].T
        velocity = y[3:6].T
        omega_body = y[10:13].T
        qb_y, qbd_y, qb_z, qbd_z = y[13], y[14], y[15], y[16]

        quats = y[6:10].T
        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
        rot = Rotation.from_quat(quats).as_matrix()          # (N, 3, 3)

        # Bent centreline, identical in construction to _tip_state in
        # arrow_flight_sim.py.
        nodes = np.linspace(0.0, arrow.length_m, SHAFT_NODES)
        phi = np.asarray(arrow.phi(nodes), dtype=float)
        local = np.empty((n, nodes.size, 3))
        local[:, :, 0] = nodes[None, :] - arrow.x_cm
        local[:, :, 1] = qb_y[:, None] * phi[None, :]
        local[:, :, 2] = qb_z[:, None] * phi[None, :]
        shaft = cm[:, None, :] + np.einsum("nij,nkj->nki", rot, local)

        # Tip velocity = CM velocity + omega x offset + transverse bending
        # velocity, again as in _tip_state.
        phi_tip = float(arrow.phi(arrow.length_m))
        offset = np.column_stack(
            [np.full(n, arrow.length_m - arrow.x_cm), qb_y * phi_tip, qb_z * phi_tip]
        )
        bend_rate = np.column_stack(
            [np.zeros(n), qbd_y * phi_tip, qbd_z * phi_tip]
        )
        tip_velocity = velocity + np.einsum(
            "nij,nj->ni", rot, np.cross(omega_body, offset) + bend_rate
        )

        return cls(
            t=t,
            cm=cm,
            tip=shaft[:, -1, :],
            nock=shaft[:, 0, :],
            shaft=shaft,
            tip_velocity=tip_velocity,
            speed=np.linalg.norm(velocity, axis=1),
            result=result,
        )

    @property
    def duration(self) -> float:
        return float(self.t[-1]) if self.t.size else 0.0

    def frame_at(self, t_query: float) -> int:
        """Index of the last stored frame at or before ``t_query``."""
        if self.t.size == 0:
            return 0
        i = int(np.searchsorted(self.t, t_query, side="right") - 1)
        return int(np.clip(i, 0, self.t.size - 1))


# ==========================================
# Simulation worker
# ==========================================
class SimulationWorker(QtCore.QObject):
    """Runs :func:`arrow_flight_sim.simulate` on a worker thread.

    ``simulate`` validates its own inputs and raises ``ValueError`` for
    impossible arrows or initial conditions; those are surfaced through
    :attr:`failed` so the window can report them instead of dying.
    """

    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, kwargs: dict):
        super().__init__()
        self._kwargs = kwargs

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            result = simulate(**self._kwargs)
        except Exception as exc:  # noqa: BLE001 - reported in the status bar
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)


# ==========================================
# The dual view widget
# ==========================================
class ArrowView(QtWidgets.QWidget):
    """One pyqtgraph view of the flight in a chosen world plane.

    ``vertical`` selects which world coordinate is drawn on the y axis:
    1 for the side view (height, the x-y plane) and 2 for the plan view
    (sideways offset, the x-z plane).  ``horizontal`` is the x axis for both,
    so the two views are mirror images of the same shot.

    The widget owns every graphics item for its plane up front and only ever
    calls ``setData`` on them, which is what keeps playback cheap; the window
    then drives both instances from a single frame index.
    """

    def __init__(
        self,
        parent=None,
        vertical: int = 1,
        title: str = "",
        vertical_label: str = "",
        reference_label: str = "",
    ):
        super().__init__(parent)
        self.vertical = int(vertical)
        # Set by set_trajectory/set_target, both of which are called by the
        # window; initialised here so the caption helpers are safe to call at
        # any time.
        self.trajectory: Optional[Trajectory] = None
        self.target: Optional[Target] = None

        self.plot = pg.PlotWidget(background="w")
        plot_item = self.plot.getPlotItem()
        plot_item.setTitle(title)
        plot_item.setLabel("bottom", "x  -  range [m]")
        plot_item.setLabel("left", vertical_label)
        plot_item.showGrid(x=True, y=True, alpha=0.25)
        plot_item.setAspectLocked(False)
        # Keep the axes in plain metres. pyqtgraph otherwise rescales an axis
        # to an SI prefix plus a "(x...)" offset once the data leaves the
        # default range, which is confusing when the whole point of the plot
        # is to read the flight path in metres.
        for name in ("left", "bottom"):
            plot_item.getAxis(name).enableAutoSIPrefix(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)

        self.view_box = self.plot.getViewBox()

        # Ground/zero reference line.
        self.reference_line = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen(C_AXIS, width=2, style=QtCore.Qt.PenStyle.DashLine),
            label=reference_label,
            labelOpts={"position": 0.95, "color": C_AXIS, "movable": False},
        )
        self.plot.addItem(self.reference_line)

        # Full flight path, drawn faintly, plus the bright path already flown.
        self.full_path = pg.PlotCurveItem(
            pen=pg.mkPen(C_PATH, width=1), antialias=True
        )
        self.trail = pg.PlotCurveItem(
            pen=pg.mkPen(C_TRAIL, width=2), antialias=True
        )
        self.plot.addItem(self.full_path)
        self.plot.addItem(self.trail)

        # Checkpoint markers (hidden until a shot defines them).
        self.checkpoint_marks: list[pg.InfiniteLine] = []
        self.checkpoint_labels: list[pg.TextItem] = []

        # Target: a vertical line at its distance, a horizontal line at its
        # height (side view only), the bullseye itself and its caption.
        self.target_vertical = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(C_TARGET, width=1, style=QtCore.Qt.PenStyle.DotLine),
        )
        self.target_horizontal = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen(C_TARGET, width=1, style=QtCore.Qt.PenStyle.DotLine),
        )
        self.target_dot = pg.ScatterPlotItem(
            size=9,
            pen=pg.mkPen(C_TARGET, width=2),
            brush=pg.mkBrush(255, 255, 255, 40),
        )
        self.target_label = pg.TextItem(
            "target",
            color=C_TARGET,
            # Anchored to the right of its position so the caption grows
            # towards the incoming arrow instead of off the right edge.
            anchor=(1.0, 1.0),
        )
        self.plot.addItem(self.target_vertical)
        self.plot.addItem(self.target_horizontal)
        self.plot.addItem(self.target_dot)
        self.plot.addItem(self.target_label)
        self._hide_target()

        # The arrow itself: the bent centreline, plus nock and tip markers.
        self.shaft_curve = pg.PlotCurveItem(
            pen=pg.mkPen(C_SHAFT, width=3),
            antialias=True,
            connect="finite",
        )
        self.plot.addItem(self.shaft_curve)
        self.nock_dot = pg.ScatterPlotItem(
            size=7, pen=pg.mkPen(C_NOCK), brush=pg.mkBrush(*C_NOCK)
        )
        self.tip_dot = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen(C_TIP, width=2), brush=pg.mkBrush(*C_TIP)
        )
        self.plot.addItem(self.nock_dot)
        self.plot.addItem(self.tip_dot)

        # Tip velocity vector.
        self.velocity_curve = pg.PlotCurveItem(
            pen=pg.mkPen(C_VELOCITY, width=2), antialias=True
        )
        self.plot.addItem(self.velocity_curve)

        # Where the tip ended up, revealed when the shot completes.
        self.impact_dot = pg.ScatterPlotItem(
            size=12,
            symbol="t1",
            pen=pg.mkPen(C_IMPACT, width=2),
            brush=pg.mkBrush(255, 255, 255, 0),
        )
        self.impact_label = pg.TextItem("", color=C_IMPACT, anchor=(0.5, -0.2))
        self.plot.addItem(self.impact_dot)
        self.plot.addItem(self.impact_label)
        self._set_visible(self.impact_dot, False)
        self._set_visible(self.impact_label, False)

        self.set_trajectory(None)

    # ---------------------------------------------------------------
    @staticmethod
    def _set_visible(item, visible: bool) -> None:
        item.setVisible(visible)

    def _hide_target(self) -> None:
        self._set_visible(self.target_vertical, False)
        self._set_visible(self.target_horizontal, False)
        self._set_visible(self.target_dot, False)
        self._set_visible(self.target_label, False)

    def set_trajectory(self, trajectory: Optional[Trajectory]) -> None:
        """Attach a flight (or ``None`` to clear) and reset to its first frame."""
        self.trajectory = trajectory
        self._set_visible(self.impact_dot, False)
        self._set_visible(self.impact_label, False)

        if trajectory is None:
            for item in (
                self.full_path,
                self.trail,
                self.shaft_curve,
                self.velocity_curve,
            ):
                item.setData([], [])
            self.nock_dot.setData([], [])
            self.tip_dot.setData([], [])
            return

        tip = trajectory.tip
        self.full_path.setData(tip[:, 0], tip[:, self.vertical])
        self.set_frame(0)

    def set_checkpoints(self, values: Optional[Sequence[float]]) -> None:
        """Draw dotted verticals at each checkpoint distance."""
        for line in self.checkpoint_marks:
            self.plot.removeItem(line)
        for label in self.checkpoint_labels:
            self.plot.removeItem(label)
        self.checkpoint_marks = []
        self.checkpoint_labels = []

        for value in values or ():
            line = pg.InfiniteLine(
                pos=float(value),
                angle=90,
                pen=pg.mkPen(
                    C_CHECK, width=1, style=QtCore.Qt.PenStyle.DashLine
                ),
            )
            label = pg.TextItem(
                f"{float(value):g} m",
                color=C_CHECK,
                anchor=(0.5, 1.0),
            )
            self.plot.addItem(line)
            self.plot.addItem(label)
            self.checkpoint_marks.append(line)
            self.checkpoint_labels.append(label)

    def _position_checkpoint_labels(self) -> None:
        """Park each checkpoint caption just above the top of the view.

        Only needed while auto-ranging, so it is called after the window
        fits the axes rather than on every playback frame.
        """
        _, top = self.view_box.viewRange()[1]
        for label, line in zip(self.checkpoint_labels, self.checkpoint_marks):
            label.setPos(line.value(), top)

    def set_target(self, target: Optional[Target]) -> None:
        """Draw the target face in this plane, or clear it when ``None``."""
        self.target = target
        if target is None:
            self._hide_target()
            return

        self.target_vertical.setPos(target.distance_m)
        self._set_visible(self.target_vertical, True)

        # The height reference line only means something in the side view;
        # in the plan view the vertical axis is z, where a crosshair line
        # would only add clutter.
        if self.vertical == 1:
            self.target_horizontal.setPos(target.height_m)
            self._set_visible(self.target_horizontal, True)

        centre = (
            (target.distance_m, target.height_m)
            if self.vertical == 1
            else (target.distance_m, target.z_m)
        )
        self.target_dot.setData([centre[0]], [centre[1]])
        self.target_label.setText(
            f"target {target.distance_m:g} m / {centre[1]:g} m"
        )
        self._set_visible(self.target_dot, True)
        self._set_visible(self.target_label, True)
        self._position_target_label()

    def _position_target_label(self) -> None:
        """Float the target caption above the bullseye, inside the view.

        The offset is a fraction of the visible vertical span so the caption
        sits clear of both the bullseye and the ground/centre-line caption
        whichever way the axes happen to be scaled.
        """
        (x_lo, x_hi), (y_lo, y_hi) = self.view_box.viewRange()
        if self.target is None or x_hi <= x_lo or y_hi <= y_lo:
            return
        x = self.target.distance_m
        y = (
            self.target.height_m
            if self.vertical == 1
            else self.target.z_m
        )
        # Keep the caption just inside the right edge of the view.
        x = min(x, x_hi - 0.02 * (x_hi - x_lo))
        self.target_label.setPos(x, y_lo + 0.96 * (y_hi - y_lo))

    def set_reference(self, y_value: float) -> None:
        """Move the ground / zero-offset line."""
        self.reference_line.setPos(y_value)

    def show_impact(self, trajectory: Trajectory) -> None:
        """Mark where the tip finished the flight."""
        tip = trajectory.tip[-1]
        self.impact_dot.setData([tip[0]], [tip[self.vertical]])
        self.impact_label.setText("impact")
        self.impact_label.setPos(tip[0], tip[self.vertical])
        self._set_visible(self.impact_dot, True)
        self._set_visible(self.impact_label, True)

    def set_frame(self, index: int) -> None:
        """Draw the arrow at ``index``; both views get the same index."""
        trajectory = self.trajectory
        if trajectory is None:
            return

        index = int(np.clip(index, 0, trajectory.t.size - 1))
        shaft = trajectory.shaft[index]
        tip = trajectory.tip[index]
        nock = trajectory.nock[index]

        self.shaft_curve.setData(shaft[:, 0], shaft[:, self.vertical])
        self.nock_dot.setData([nock[0]], [nock[self.vertical]])
        self.tip_dot.setData([tip[0]], [tip[self.vertical]])

        # Trail: only the part of the path flown so far.
        flown = trajectory.tip[: index + 1]
        self.trail.setData(flown[:, 0], flown[:, self.vertical])

        # Velocity vector, drawn in the tip direction of this plane.
        velocity = trajectory.tip_velocity[index]
        dx = float(velocity[0])
        dy = float(velocity[self.vertical])
        norm = float(np.hypot(dx, dy))
        if norm < 1e-9:
            self.velocity_curve.setData([], [])
            return
        # 0.6 s of travel: a long enough arrow to read, short enough not to
        # swamp the plot.
        scale = 0.6 / norm
        self.velocity_curve.setData(
            [tip[0], tip[0] + dx * scale],
            [tip[self.vertical], tip[self.vertical] + dy * scale],
        )

    def set_x_range(self, x_min: float, x_max: float) -> None:
        """Pan/zoom the shared range axis (used for the follow mode)."""
        self.plot.setXRange(x_min, x_max, padding=0)


# ==========================================
# The window
# ==========================================
class ArrowViewerWindow(QtWidgets.QMainWindow):
    """Edits the shot on the left and plays it on the two views on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Arrow flight viewer")

        self._widgets: dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._group_boxes: list[QtWidgets.QGroupBox] = []
        self.trajectory: Optional[Trajectory] = None
        self.target: Optional[Target] = None
        self._arrow: Optional[Arrow] = None
        self._checkpoints: Optional[list] = None
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[SimulationWorker] = None
        self._frame = 0
        self._time = 0.0
        self._playing = False
        self._follow = False

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._sync_velocity_inputs()
        self.set_status("Ready - press Start to integrate a shot.")

    # ---------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------
    def _build_ui(self) -> None:
        panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(6, 6, 6, 6)

        panel_layout.addWidget(self._build_initial_group())
        panel_layout.addWidget(self._build_target_group())
        panel_layout.addWidget(self._build_playback_group())
        panel_layout.addWidget(self._build_advanced_group())
        panel_layout.addWidget(self._build_buttons())
        panel_layout.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(310)
        scroll.setMaximumWidth(420)

        # The two views share one horizontal (range) axis, so zooming or
        # panning one of them moves the other.
        self.side_view = ArrowView(
            vertical=1,
            title="Side view  -  x-y plane",
            vertical_label="y  -  height [m]",
            reference_label="ground",
        )
        self.top_view = ArrowView(
            vertical=2,
            title="Top view  -  x-z plane",
            vertical_label="z  -  sideways [m]",
            reference_label="centre line",
        )
        self.side_view.plot.setXLink(self.top_view.plot)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.side_view)
        splitter.addWidget(self.top_view)
        splitter.setSizes([400, 400])

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(splitter)

        outer = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(scroll)
        outer.addWidget(right)
        outer.setStretchFactor(0, 0)
        outer.setStretchFactor(1, 1)
        outer.setSizes([340, 1100])

        self.setCentralWidget(outer)

        self.readout = QtWidgets.QLabel("")
        self.readout.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.statusBar().addPermanentWidget(self.readout, 1)

        self.resize(1460, 880)

    def _spin(self, layout, fld: Field, suffix: str = "") -> QtWidgets.QDoubleSpinBox:
        """Add one labelled spin box for ``fld`` and remember it by name."""
        box = QtWidgets.QDoubleSpinBox()
        box.setDecimals(fld.decimals)
        box.setRange(fld.minimum, fld.maximum)
        box.setSingleStep(fld.step)
        box.setValue(fld.default)
        if suffix:
            box.setSuffix(suffix)
        layout.addRow(fld.label, box)
        self._widgets[fld.name] = box
        return box

    # ---------------------------------------------------------------
    # Panel sections
    # ---------------------------------------------------------------
    def _group(self, title: str) -> tuple:
        """Create a collapsible group box holding a form; return (box, form)."""
        box = QtWidgets.QGroupBox(title)
        box.setCheckable(True)
        box.setChecked(True)
        form = QtWidgets.QFormLayout(box)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._group_boxes.append(box)
        return box, form

    def _build_initial_group(self) -> QtWidgets.QWidget:
        box, form = self._group("Initial conditions")

        # World axes: +X forward, +Y up, +Z sideways.
        hint = QtWidgets.QLabel("+X range   +Y up   +Z sideways")
        hint.setStyleSheet("color: #666666;")
        form.addRow(hint)

        for fld in INITIAL_FIELDS:
            box_widget = self._spin(form, fld)
            # Angles are degrees; keep that visible in the widget itself.
            if fld.name in ("elev", "yaw", "shaft_elev", "shaft_yaw"):
                box_widget.setSuffix(" deg")

        # The velocity can be given either way round; the mode box decides
        # which triple of fields is actually read.
        self.velocity_mode = QtWidgets.QComboBox()
        self.velocity_mode.addItems(["speed + elevation/yaw", "3-D velocity vector"])
        self.velocity_mode.currentIndexChanged.connect(self._sync_velocity_inputs)
        form.addRow("velocity entry", self.velocity_mode)

        self.follow_direction = QtWidgets.QCheckBox("shaft points along velocity")
        self.follow_direction.setChecked(True)
        self.follow_direction.toggled.connect(self._sync_velocity_inputs)
        form.addRow(self.follow_direction)

        return box

    def _build_target_group(self) -> QtWidgets.QWidget:
        box, form = self._group("Target")
        for fld in TARGET_FIELDS:
            self._spin(form, fld)

        self.stop_at_target = QtWidgets.QCheckBox("stop at target distance")
        self.stop_at_target.setChecked(True)
        self.stop_at_target.setToolTip(
            "Integration ends when the tip crosses the target distance.\n"
            "Unticked, the target is still drawn but the arrow flies on."
        )
        form.addRow(self.stop_at_target)

        self.checkpoint_edit = QtWidgets.QLineEdit(DEFAULT_CHECKPOINTS)
        self.checkpoint_edit.setToolTip(
            "Comma separated distances [m] drawn as dotted verticals."
        )
        form.addRow("checkpoints [m]", self.checkpoint_edit)
        return box

    def _build_playback_group(self) -> QtWidgets.QWidget:
        box, form = self._group("Visualization")
        for fld in PLAYBACK_FIELDS:
            spin = self._spin(form, fld)
            if fld.name in ("speed_mult", "vscale"):
                spin.setSuffix(" x")
        return box

    def _build_advanced_group(self) -> QtWidgets.QWidget:
        box, form = self._group("Advanced  (arrow, feathers, air, integrator)")
        for fld in ADVANCED_FIELDS:
            self._spin(form, fld)
        note = QtWidgets.QLabel(
            "A smaller max solver step is more accurate but takes longer to run."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666666;")
        form.addRow(note)
        return box

    def _build_buttons(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self.start)
        layout.addWidget(self.start_button)

        row = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("Pause")
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setEnabled(False)
        row.addWidget(self.play_button)

        restart = QtWidgets.QPushButton("Restart")
        restart.clicked.connect(self.restart)
        self.restart_button = restart
        self.restart_button.setEnabled(False)
        row.addWidget(restart)
        layout.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        fit = QtWidgets.QPushButton("Fit view")
        fit.clicked.connect(self._fit_views)
        row2.addWidget(fit)

        self.follow_check = QtWidgets.QCheckBox("follow")
        self.follow_check.setToolTip(
            "Keep the arrow inside the window by scrolling the range axis."
        )
        self.follow_check.toggled.connect(self._set_follow)
        row2.addWidget(self.follow_check)
        layout.addLayout(row2)

        self.reset_button = QtWidgets.QPushButton("Restore defaults")
        self.reset_button.clicked.connect(self._restore_defaults)
        layout.addWidget(self.reset_button)
        return box

    # Reading the panel
    # ---------------------------------------------------------------
    def _value(self, name: str) -> float:
        return float(self._widgets[name].value())

    def _sync_velocity_inputs(self) -> None:
        """Grey out whichever velocity entries the current mode does not use."""
        by_angles = self.velocity_mode.currentIndex() == 0
        for name, active in (
            ("speed", by_angles),
            ("elev", by_angles),
            ("yaw", by_angles),
            ("vx", not by_angles),
            ("vy", not by_angles),
            ("vz", not by_angles),
            # The shaft attitude is only editable when it is not slaved to
            # the velocity direction.
            ("shaft_elev", not self.follow_direction.isChecked()),
            ("shaft_yaw", not self.follow_direction.isChecked()),
        ):
            self._widgets[name].setEnabled(active)

        if self.follow_direction.isChecked():
            elev, yaw = angles_from_vector(self._initial_velocity())
            self._widgets["shaft_elev"].setValue(elev)
            self._widgets["shaft_yaw"].setValue(yaw)

    def _initial_velocity(self) -> np.ndarray:
        if self.velocity_mode.currentIndex() == 0:
            speed = self._value("speed")
            direction = direction_from_angles(
                self._value("elev"), self._value("yaw")
            )
            return speed * direction
        return np.array(
            [self._value("vx"), self._value("vy"), self._value("vz")]
        )

    def _initial_direction(self) -> np.ndarray:
        if self.follow_direction.isChecked():
            velocity = self._initial_velocity()
            return velocity / np.linalg.norm(velocity)
        return direction_from_angles(
            self._value("shaft_elev"), self._value("shaft_yaw")
        )

    def _make_target(self) -> Target:
        return Target(
            distance_m=self._value("target_distance"),
            height_m=self._value("target_height"),
            z_m=self._value("target_z"),
            radius_m=self._value("target_radius"),
        )

    def _make_checkpoints(self) -> tuple[Optional[list], list]:
        """Parse the checkpoint field into distances and names."""
        text = self.checkpoint_edit.text().strip()
        if not text:
            return None, []

        values: list[float] = []
        names: list[str] = []
        for chunk in text.replace(";", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                value = float(chunk)
            except ValueError:
                self.set_status(
                    f"Ignoring unreadable checkpoint entry {chunk!r}.", error=True
                )
                continue
            values.append(value)
            names.append(f"{value:g} m")

        if not values:
            return None, []
        return values, names

    def _make_arrow(self) -> Arrow:
        return Arrow(
            length_m=self._value("length"),
            shaft_outer_d_m=self._value("shaft_od"),
            shaft_inner_d_m=self._value("shaft_id"),
            total_mass_kg=self._value("mass"),
            point_mass_kg=self._value("point_mass"),
            point_x_m=self._value("point_x"),
            spine=self._value("spine"),
            feathers=Feather(
                x_start_m=self._value("feather_x0"),
                x_end_m=self._value("feather_x1"),
                height_m=self._value("feather_h"),
                area_each_m2=self._value("feather_area"),
                count=int(round(self._value("feather_count"))),
                cant_deg=self._value("feather_cant"),
                mass_kg=self._value("feather_mass"),
            ),
        )

    # Running a shot
    # ---------------------------------------------------------------
    def start(self) -> None:
        """Integrate the shot on a worker thread, then play it back."""
        if self._thread is not None:
            return

        try:
            arrow = self._make_arrow()
        except ValueError as exc:
            # Arrow validates its own geometry and mass budget.
            self.set_status(f"Cannot start: {exc}", error=True)
            return

        velocity = self._initial_velocity()
        if np.linalg.norm(velocity) < 1e-9:
            self.set_status("Cannot start: the initial velocity is zero.", error=True)
            return

        direction = self._initial_direction()
        if np.linalg.norm(direction) < 1e-9:
            self.set_status(
                "Cannot start: the shaft direction is zero.", error=True
            )
            return

        checkpoints, checkpoint_names = self._make_checkpoints()
        self.target = self._make_target()

        kwargs = dict(
            arrow=arrow,
            atmosphere=Atmosphere(
                pressure_pa=self._value("pressure"),
                temperature_k=self._value("temperature"),
            ),
            initial_position_m=[
                self._value("x0"),
                self._value("y0"),
                self._value("z0"),
            ],
            initial_velocity_world_m_s=velocity,
            initial_direction=direction,
            initial_roll_deg=self._value("roll"),
            initial_angular_velocity_body_rad_s=[
                self._value("omega_pitch"),
                self._value("omega_yaw"),
                self._value("spin"),
            ],
            wind_world_m_s=[
                self._value("wind_x"),
                self._value("wind_y"),
                self._value("wind_z"),
            ],
            target=self.target if self.stop_at_target.isChecked() else None,
            checkpoints=checkpoints,
            checkpoint_names=checkpoint_names or None,
            ground_y_m=self._value("ground_y"),
            t_end=self._value("t_end"),
            dt=self._value("dt"),
        )

        self._arrow = arrow
        self._checkpoints = checkpoints

        self._set_busy(True)
        self.set_status("Integrating flight ...")
        self.readout.setText("")
        self._clear_views()

        self._thread = QtCore.QThread(self)
        self._worker = SimulationWorker(kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        # Tear the thread down once whichever outcome arrived.
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_stopped)
        self._thread.start()

    @QtCore.pyqtSlot(object)
    def _on_finished(self, result) -> None:
        self._build_trajectory(result)
        self._set_busy(False)

        summary = [
            f"finished at t = {result.time_s:.3f} s",
            f"end reason: {result.reason}",
        ]
        if result.target_error_m is not None:
            summary.append(
                "target error = "
                + np.array2string(
                    np.round(result.target_error_m, 4), precision=3, separator=", "
                )
                + " m"
            )
            summary.append("HIT" if result.target_hit else "MISS")
        elif self.target is not None:
            # The flight ended on the ground before the target distance, so
            # the integrator never evaluated a target error.  Report how far
            # short the arrow landed instead.
            short_by = self.target.distance_m - float(
                self.trajectory.tip[-1][0]
            ) if self.trajectory is not None else float("nan")
            if short_by > 0:
                summary.append(f"landed {short_by:.2f} m short of the target")
        self.set_status("   |   ".join(summary))
        self.restart_button.setEnabled(True)

    @QtCore.pyqtSlot(str)
    def _on_failed(self, message: str) -> None:
        self._set_busy(False)
        self.set_status(f"Simulation failed: {message}", error=True)

    @QtCore.pyqtSlot()
    def _on_thread_stopped(self) -> None:
        self._thread = None
        self._worker = None

    # Playback
    # ---------------------------------------------------------------
    def _build_trajectory(self, result) -> None:
        """Turn a finished solve into displayable frames for both views."""
        try:
            self.trajectory = Trajectory.from_result(self._arrow, result)
        except Exception as exc:  # noqa: BLE001 - reported in the status bar
            self.trajectory = None
            self.set_status(f"Could not prepare the flight: {exc}", error=True)
            return

        checkpoints = self._checkpoints
        for view in (self.side_view, self.top_view):
            view.set_trajectory(self.trajectory)
            view.set_target(self.target)
            view.set_checkpoints(checkpoints)
            view.set_reference(self._value("ground_y"))

        self.side_view.plot.enableAutoRange()
        self.top_view.plot.enableAutoRange()
        self._fit_views()

        # The shot is integrated: rewind and hand it to the playback timer.
        self._time = 0.0
        self._frame = 0
        self._apply_frame(0)

        self.timer.setInterval(self._frame_interval())
        self.play_button.setEnabled(True)
        self.play_button.setText("Pause")
        self.restart_button.setEnabled(True)
        self._playing = True
        self.timer.start()

    def _clear_views(self) -> None:
        self.trajectory = None
        self.timer.stop()
        for view in (self.side_view, self.top_view):
            view.set_trajectory(None)
            view.set_target(None)
            view.set_checkpoints(None)
        self.play_button.setEnabled(False)
        self.play_button.setText("Pause")
        self.restart_button.setEnabled(False)
        self._playing = False

    def _frame_interval(self) -> int:
        """Timer period in ms from the requested frame rate."""
        return max(1, int(round(1000.0 / max(1.0, self._value("fps")))))

    def _on_tick(self) -> None:
        """Advance the playback clock and draw the matching frame.

        Playback speed is a pure time scaling: each timer tick adds
        ``interval * speed_mult`` seconds of *simulation* time, and the frame
        index is then looked up in the trajectory's own (non-uniform) time
        array.  The same index goes to both views.
        """
        if self.trajectory is None:
            self.timer.stop()
            self._playing = False
            return

        interval_s = self.timer.interval() / 1000.0
        self._time += interval_s * self._value("speed_mult")

        duration = self.trajectory.duration
        if self._time >= duration:
            self._time = duration
            index = self.trajectory.t.size - 1
            self.timer.stop()
            self._playing = False
            self.play_button.setText("Play")
            for view in (self.side_view, self.top_view):
                view.show_impact(self.trajectory)
        else:
            index = self.trajectory.frame_at(self._time)

        self._frame = index
        self._apply_frame(index)

    def _apply_frame(self, index: int) -> None:
        """Draw one frame in both views at the same instant."""
        # One index into both views is what keeps them synchronised; there is
        # no second timer and no second clock.
        self.side_view.set_frame(index)
        self.top_view.set_frame(index)

        if self._follow:
            self._follow_arrow(index)

        if self.trajectory is not None and self.trajectory.t.size:
            tip = self.trajectory.tip[index]
            speed = float(self.trajectory.speed[index])
            t_now = float(self.trajectory.t[index])
            self.readout.setText(
                f"t = {t_now:6.3f} s   x = {tip[0]:7.2f} m   "
                f"y = {tip[1]:6.2f} m   z = {tip[2]:6.3f} m   "
                f"|v| = {speed:6.2f} m/s"
            )

    def _follow_arrow(self, index: int) -> None:
        """Keep the arrow inside the window by scrolling the shared x axis."""
        trajectory = self.trajectory
        if trajectory is None:
            return

        tip_x = float(trajectory.tip[index][0])
        (x_lo, x_hi), _ = self.side_view.plot.getViewBox().viewRange()
        span = max(1.0, x_hi - x_lo)
        # Leave room ahead of the arrow rather than centring it, so the
        # remaining flight stays visible.
        centre = tip_x + 0.15 * span
        new_lo = centre - 0.5 * span

        # Both views are set explicitly rather than relying on the X link:
        # the link reconciles overlapping viewports by pixel geometry, which
        # is fine for a hand-pan but would drift over thousands of frames.
        # Blocking the link stops each set from re-aligning the other.
        boxes = (self.side_view.plot.getViewBox(), self.top_view.plot.getViewBox())
        for box in boxes:
            box.blockLink(True)
        try:
            for view in (self.side_view, self.top_view):
                view.set_x_range(new_lo, new_lo + span)
        finally:
            for box in boxes:
                box.blockLink(False)

    # Controls
    # ---------------------------------------------------------------
    def toggle_playback(self) -> None:
        if self.trajectory is None:
            return

        if self._playing:
            self._playing = False
            self.timer.stop()
            self.play_button.setText("Play")
            return

        # Restarting from the end replays instead of doing nothing.
        if self._time >= self.trajectory.duration:
            self._time = 0.0
            self._apply_frame(0)
        self._playing = True
        self.play_button.setText("Pause")
        self.timer.setInterval(self._frame_interval())
        self.timer.start()

    def restart(self) -> None:
        """Replay the current shot from the beginning."""
        if self.trajectory is None:
            return
        self._time = 0.0
        self._frame = 0
        self._apply_frame(0)
        self._playing = True
        self.play_button.setText("Pause")
        self.timer.setInterval(self._frame_interval())
        self.timer.start()

    def _fit_views(self) -> None:
        self.side_view.plot.enableAutoRange()
        self.top_view.plot.enableAutoRange()
        # enableAutoRange only sets a flag; the range settles on the next
        # update, so let Qt apply it before parking the captions.
        QtWidgets.QApplication.processEvents()
        for view in (self.side_view, self.top_view):
            view._position_checkpoint_labels()
            view._position_target_label()

    def _set_follow(self, enabled: bool) -> None:
        self._follow = bool(enabled)
        if self._follow and self.trajectory is not None:
            self._follow_arrow(self._frame)

    def _restore_defaults(self) -> None:
        for name, fld in ALL_FIELDS.items():
            self._widgets[name].setValue(fld.default)
        self.checkpoint_edit.setText(DEFAULT_CHECKPOINTS)
        self.velocity_mode.setCurrentIndex(0)
        self.stop_at_target.setChecked(True)
        self.follow_check.setChecked(False)
        for box in self._group_boxes:
            if box.title().startswith("Advanced"):
                box.setChecked(False)
        self._sync_velocity_inputs()
        self.set_status("Defaults restored - press Start to run the example shot.")

    def _set_busy(self, busy: bool) -> None:
        """Grey the Start button out and show a wait cursor during a solve."""
        self.start_button.setEnabled(not busy)
        self.start_button.setText("Solving ..." if busy else "Start")
        if busy:
            QtWidgets.QApplication.setOverrideCursor(
                QtCore.Qt.CursorShape.WaitCursor
            )
        else:
            QtWidgets.QApplication.restoreOverrideCursor()

    def set_status(self, message: str, error: bool = False) -> None:
        self.statusBar().showMessage(message)
        if error:
            self.statusBar().setStyleSheet("color: #b00020;")
        else:
            self.statusBar().setStyleSheet("")

    def closeEvent(self, event) -> None:
        # Let any in-flight solve finish before the thread object dies.
        if self._thread is not None:
            self.timer.stop()
            self._thread.quit()
            self._thread.wait(5000)
        super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv)

    window = ArrowViewerWindow()
    window.show()
    return app.exec() if owns_app else 0


if __name__ == "__main__":
    raise SystemExit(main())
