"""Alternative pyqtgraph viewer for a simulated arrow shot.

This is a drop-in sibling of :mod:`viewer`: it consumes the same
:class:`~simulation.SimulationResult` and offers the same two views (global frame
and CoM vibration frame) with the same playback controls, but renders through
pyqtgraph instead of matplotlib.

Why it is smoother: every frame only calls ``setData`` on pre-created graphics
items (no artist is created, styled or re-laid-out during playback), the shaft
is drawn as a filled band between two curves rather than a 2N-vertex polygon,
and the timer runs at display rate. There is deliberately **no frame
interpolation** -- smoothness comes from raising the simulation's logging rate
(see :func:`run`), not from blending between logged frames.

The viewer creates a ``QApplication`` for you (Qt refuses to build widgets
without one, and the matplotlib viewer never needed it), so no Qt boilerplate is
required at the call site.

The top plot keeps pyqtgraph's own mouse handling: wheel to zoom in/out about the
cursor, left-drag to pan, right-click for the axis menu. ``Reset view`` (or
``PgViewer.reset_view()``) returns it to the harmonized default.

Usage::

    PgViewer(result).show()

or, to run a finely logged shot and view it::

    run()
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from simulation import SimConfig, Simulation, SimulationResult

# Numerical guards, kept at module level because they are hygiene, not style knobs.
NORM_EPS = 1e-6
VEL_EPS = 1e-4

# Antialiasing is a per-curve option, and pyqtgraph defaults it to whatever the
# global config says. Turn it on before any item is built so every curve, fill
# and marker we create is smooth. 'background' is pyqtgraph's dark default; a
# light theme is what we want for plots, and 'foreground' controls the axis /
# label text that has to stay readable on it.
pg.setConfigOptions(antialias=True, background="w", foreground="k")


# ==========================================
# Small graphics helpers
# ==========================================
def arrow_polygon(x, y, dx, dy, length, width, head_length, head_width):
    """Return the vertices of a filled arrow from ``(x, y)`` along ``(dx, dy)``.

    The arrow is drawn in data units so that it keeps its proportions when the
    window is resized, matching the ``scale_units="xy"`` arrows of the
    matplotlib viewer. A zero-length vector yields a degenerate (single point)
    polygon, which renders as nothing -- the caller's job to hide the item.
    """
    mag = float(np.hypot(dx, dy))
    if mag <= VEL_EPS or length <= 0.0:
        return np.zeros((0, 2))

    u_x, u_y = dx / mag, dy / mag
    n_x, n_y = -u_y, u_x

    head_length = min(head_length, length)
    shaft_end = length - head_length
    w_shaft = width / 2.0
    w_head = head_width / 2.0

    def pt(along, across):
        return np.array([x + along * u_x + across * n_x, y + along * u_y + across * n_y])

    return np.array(
        [
            pt(0.0, w_shaft),
            pt(shaft_end, w_shaft),
            pt(shaft_end, w_head),
            pt(length, 0.0),
            pt(shaft_end, -w_head),
            pt(shaft_end, -w_shaft),
            pt(0.0, -w_shaft),
        ]
    )


class AntialiasedFillBetween(pg.FillBetweenItem):
    """A FillBetweenItem whose fill edge is antialiased.

    ``FillBetweenItem`` paints through plain QPainterPath, which Qt renders with
    hard (aliased) edges unless a render hint is set. The shaft is a filled band
    whose whole point is to look smooth, so we turn antialiasing on for it.
    """

    def paint(self, painter, *args):
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        super().paint(painter, *args)


class DataCircle:
    """A filled circle drawn in data units.

    pyqtgraph's scatter symbols default to being sized in *pixels*, which would
    make the bow rest change apparent size whenever the window is resized. With
    ``pxMode=False`` the size is in data units instead, so the radius stays
    honest at every scale and the disc is a real filled circle.

    (Drawing the ring as a filled curve with ``fillLevel=0`` does not work: the
    fill is then measured against the y=0 data line, so a circle sitting away
    from y=0 gets a rectangular skirt down to the axis -- visible as the bow
    rest "melting" toward the zero line in the vibration plot.)
    """

    def __init__(self, plot, radius, pen="k", brush=(150, 150, 150, 90), z=5):
        self._radius = float(radius)

        # size is the diameter when pxMode is False.
        self.scatter = pg.ScatterPlotItem(
            pxMode=False,
            size=2.0 * self._radius,
            symbol="o",
            pen=pg.mkPen(pen, width=2),
            brush=None if brush is None else pg.mkBrush(*brush),
            antialias=True,
        )
        self.scatter.setZValue(z)
        plot.addItem(self.scatter)
        self.curve = self.scatter  # kept for the set_visible()/isVisible() callers
        self.set_center(0.0, 0.0)

    def set_center(self, cx, cy) -> None:
        self.scatter.setData(x=[float(cx)], y=[float(cy)])

    def set_visible(self, visible: bool) -> None:
        self.scatter.setVisible(visible)

    def is_visible(self) -> bool:
        return self.scatter.isVisible()


# ==========================================
# Application bootstrap
# ==========================================
# True once *this module* had to create the QApplication. If the host app
# created it, we must not start a nested event loop in show().
_APP_CREATED_HERE = False


def _ensure_app() -> QtWidgets.QApplication:
    """Return the process-wide QApplication, creating it if there isn't one.

    ``pg.mkQApp()`` returns the existing instance when one is already alive, so
    this is safe to call from a notebook or a larger Qt app that has already
    built its own.
    """
    global _APP_CREATED_HERE
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = pg.mkQApp()
        _APP_CREATED_HERE = True
    return app


# ==========================================
# Viewer
# ==========================================
class PgViewer:
    """Renders a :class:`SimulationResult` with pyqtgraph and a playback bar.

    Usage::

        PgViewer(result).show()
    """

    # --- Plot sizing ---
    # As in the matplotlib viewer, both plots keep a true 1:1 data scale and are
    # harmonized by giving them the SAME data aspect ratio r = y_span / x_span,
    # so their on-screen boxes always come out the same shape. r is the largest
    # value that still fills the available area and still honours each plot's
    # minimum span (the data extent, and the +/-0.06 m vibration window).
    MAIN_PADDING = 0.05  # m, x/y padding around the global data extent
    LOCAL_X_PADDING = 0.02  # m, padding on the local plot's chord axis
    LOCAL_Y_MIN_SPAN = 0.12  # m, the +/-0.06 m vibration window floor

    # --- Arrow styling, in data units ---
    # Heads are a fixed *fraction* of the shaft length, so an arrow always reads
    # as an arrow no matter how short it gets.
    ARROW_HEAD_FRACTION = 0.28
    # The CoM-velocity and initial-direction arrows are long, fixed-length
    # reference arrows, so a head scaled to their length looks like a blob.
    # They get their own, much shorter head.
    DIRECTION_HEAD_FRACTION = 0.14
    ARROW_HEAD_WIDTH_RATIO = 3.0  # head width / shaft width

    FIXED_ARROW_LENGTH = 0.3  # m
    FIXED_ARROW_WIDTH = 0.0024

    FORCE_SCALE = 0.0008
    FORCE_ARROW_WIDTH = 0.004

    # --- Widgets / animation ---
    SPEED_RANGE = (1, 5)
    ANIM_INTERVAL_MS = 16  # ms, ~60 fps

    # --- Colours (matched to the matplotlib viewer) ---
    SHAFT_COLOR = (60, 130, 200)
    SHAFT_ALPHA = 120
    NODE_COLOR = (0, 0, 128)
    TAIL_TRACE_COLOR = (128, 128, 128)
    COM_COLOR = (0, 0, 255)

    def __init__(self, result: SimulationResult):
        # Qt refuses to build any widget before a QApplication exists, and
        # matplotlib's plt.show() used to do that implicitly for us. Do it here
        # so the viewer works both standalone and when embedded in another app.
        _ensure_app()

        self.result = result
        cfg = result.config
        self.outer_radius = cfg.outer_radius
        self.brace_height = cfg.brace_height

        # Playback state (instance-level, deliberately not global).
        self.is_playing = True
        self.current_frame = 0
        self.n_frames = len(result.times)
        self._laying_out = False  # re-entrancy guard for apply_layout
        # True once the user has zoomed/panned the top plot by hand. After that
        # apply_layout() stops resetting its range, so a resize does not throw
        # away the zoom. Cleared by reset_view().
        self.main_view_zoomed = False

        self._build_window()
        self._build_main_plot()
        self._build_local_plot()
        self._build_widgets()
        self._connect_events()
        self.apply_layout()

    # ==========================================
    # Window / layout
    # ==========================================
    def _build_window(self) -> None:
        """Create the top-level window and the vertical plot stack."""
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle("The Archer's Paradox (pyqtgraph)")

        outer = QtWidgets.QVBoxLayout(self.win)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        self.canvas = pg.GraphicsLayoutWidget()
        # The canvas paints the space around/outside the plots too, and that
        # defaults to pyqtgraph's dark background.
        self.canvas.setBackground("w")
        outer.addWidget(self.canvas, 1)

        # Two rows of equal stretch: the plots share the window evenly, and the
        # GraphicsLayout handles the axis labels/titles, so nothing can overlap
        # the way hand-placed absolute-inch margins did.
        self.plot_main = self.canvas.addPlot(row=0, col=0)
        self.plot_local = self.canvas.addPlot(row=1, col=0)
        self.canvas.ci.layout.setRowStretchFactor(0, 1)
        self.canvas.ci.layout.setRowStretchFactor(1, 1)

        for plot in (self.plot_main, self.plot_local):
            plot.hideButtons()
            plot.getViewBox().setBackgroundColor("w")
            plot.showGrid(x=True, y=True, alpha=0.3)
            # A 1:1 data scale, and no auto-ranging: the ranges are managed by
            # apply_layout() so the plots always fill their boxes.
            # setRange() below disables auto-ranging for us.
            plot.getViewBox().setAspectLocked(True, ratio=1)
            # Panning/zooming would fight the layout-managed ranges -- except on
            # the top plot, which opts back in below for wheel zoom.
            plot.getViewBox().setMouseEnabled(x=False, y=False)

        self.plot_main.setLabel("bottom", "X Position", units="m")
        self.plot_main.setLabel("left", "Y Deflection", units="m")
        self.plot_local.setLabel("bottom", "Chord Position from CoM", units="m")
        self.plot_local.setLabel("left", "Transverse Deflection", units="m")
        self.plot_local.setTitle("Arrow Vibrations (Local CoM Frame, Tail-to-Tip)")

        # Pan + wheel zoom on the top plot, straight from pyqtgraph: its
        # ViewBox.wheelEvent already scales about the cursor, and a left-drag
        # pans. We only have to leave the mouse enabled for those axes and make
        # apply_layout() stop fighting the user once they have touched it.
        self.plot_main.getViewBox().setMouseEnabled(x=True, y=True)
        self.plot_main.getViewBox().sigRangeChangedManually.connect(
            lambda *_: setattr(self, "main_view_zoomed", True)
        )

    def _make_arrow(self, plot, color, z):
        """Create a filled, data-unit arrow item (updated via :meth:`_set_arrow`).

        ``fillLevel`` is what actually turns the brush on: without it a
        PlotDataItem only draws the outline, whatever brush it was given.
        """
        arrow = pg.PlotDataItem(
            pen=pg.mkPen(*color, width=1.0),
            brush=pg.mkBrush(*color, 220),
            fillLevel=0.0,
            connect="all",
        )
        arrow.setZValue(z)
        plot.addItem(arrow)
        return arrow

    def _set_arrow(self, arrow, x, y, dx, dy, length, width, head_fraction=None) -> None:
        """Point an arrow item from ``(x, y)`` along ``(dx, dy)``."""
        if head_fraction is None:
            head_fraction = self.ARROW_HEAD_FRACTION
        verts = arrow_polygon(
            x,
            y,
            dx,
            dy,
            length,
            width,
            length * head_fraction,
            width * self.ARROW_HEAD_WIDTH_RATIO,
        )
        if len(verts) == 0:
            arrow.setData(x=np.empty(0), y=np.empty(0))
            return
        # Repeat the first vertex so connect="all" closes the outline.
        verts = np.vstack((verts, verts[0]))
        arrow.setData(x=verts[:, 0], y=verts[:, 1])

    def _build_main_plot(self) -> None:
        """Build the global-frame plot: bow rest, shaft, traces and arrows."""
        res = self.result
        plot = self.plot_main

        self.bow_main_circle = DataCircle(plot, res.config.bow_rest_radius, "k", z=6)

        brace = pg.InfiniteLine(
            pos=-self.brace_height, angle=90, pen=pg.mkPen("r", width=1.5)
        )
        brace.setZValue(1)
        plot.addItem(brace)
        self.brace_line = brace

        # The shaft is a band between the two offset curves. FillBetweenItem
        # turns the pair into a filled polygon, so a frame update is just two
        # setData calls -- no polygon assembly in the animation path.
        self.shaft_top = pg.PlotDataItem(pen=pg.mkPen(self.SHAFT_COLOR, width=1))
        self.shaft_bot = pg.PlotDataItem(pen=pg.mkPen(self.SHAFT_COLOR, width=1))
        self.shaft_fill = AntialiasedFillBetween(
            self.shaft_top,
            self.shaft_bot,
            brush=pg.mkBrush(*self.SHAFT_COLOR, self.SHAFT_ALPHA),
        )
        self.shaft_fill.setZValue(2)
        plot.addItem(self.shaft_fill)

        self.nodes = pg.ScatterPlotItem(
            size=4, pen=None, brush=pg.mkBrush(*self.NODE_COLOR), pxMode=True
        )
        self.nodes.setZValue(4)
        plot.addItem(self.nodes)

        self.tail_trace = pg.PlotDataItem(
            pen=pg.mkPen(
                self.TAIL_TRACE_COLOR, width=2, style=QtCore.Qt.PenStyle.DotLine
            )
        )
        self.tail_trace.setZValue(1)
        plot.addItem(self.tail_trace)

        self.com_trace = pg.PlotDataItem(
            pen=pg.mkPen(*self.COM_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine)
        )
        self.com_trace.setZValue(1)
        plot.addItem(self.com_trace)

        self.com_marker = pg.ScatterPlotItem(
            size=9, pen=None, brush=pg.mkBrush(*self.COM_COLOR), pxMode=True
        )
        self.com_marker.setZValue(5)
        plot.addItem(self.com_marker)

        # Direction / force arrows, all drawn in data units.
        self.com_vel_arrow = self._make_arrow(plot, self.COM_COLOR, z=5)
        self.init_dir_arrow = self._make_arrow(plot, (0, 0, 0), z=5)
        self.tail_force_arrow = self._make_arrow(plot, (255, 0, 0), z=5)

        # No time readout on the plot: the playback bar already carries it
        # (see self.lbl_time), and an in-plot overlay would just duplicate it.

        all_x = res.positions[:, 0, :]
        all_y = res.positions[:, 1, :]
        self.main_y_center = 0.5 * (np.min(all_y) + np.max(all_y))
        self.main_y_min_span = (
            (np.max(all_y) + self.MAIN_PADDING) - (np.min(all_y) - self.MAIN_PADDING)
        )
        # The canonical x-range is stored, not just applied: with the aspect
        # locked, the ViewBox adjusts whichever axis we don't set, so apply_layout
        # re-asserts this range on every resize to stop the view drifting.
        self.main_x_range = (
            float(np.min(all_x)) - self.MAIN_PADDING,
            float(np.max(all_x)) + self.MAIN_PADDING,
        )
        plot.setXRange(*self.main_x_range, padding=0)

        cfg = res.config
        plot.setTitle(
            f"Archer's Paradox ({cfg.draw_weight_lbs} lbs draw, "
            f"{cfg.spine_value} spine, {cfg.tip_weight_grains:.0f} gr tip)"
        )

    def _build_local_plot(self) -> None:
        """Build the vibration-frame plot: the same arrow seen from its own CoM."""
        res = self.result
        plot = self.plot_local

        self.local_shaft_top = pg.PlotDataItem(pen=pg.mkPen(self.SHAFT_COLOR, width=1))
        self.local_shaft_bot = pg.PlotDataItem(pen=pg.mkPen(self.SHAFT_COLOR, width=1))
        self.local_shaft_fill = AntialiasedFillBetween(
            self.local_shaft_top,
            self.local_shaft_bot,
            brush=pg.mkBrush(*self.SHAFT_COLOR, self.SHAFT_ALPHA),
        )
        self.local_shaft_fill.setZValue(2)
        plot.addItem(self.local_shaft_fill)

        self.local_nodes = pg.ScatterPlotItem(
            size=4, pen=None, brush=pg.mkBrush(*self.NODE_COLOR), pxMode=True
        )
        self.local_nodes.setZValue(4)
        plot.addItem(self.local_nodes)

        # CoM reference cross. Intentionally unlabelled: these are frame axes,
        # not data, so they are left out to keep the focus on the arrow.
        for angle in (0, 90):
            line = pg.InfiniteLine(pos=0, angle=angle, pen=pg.mkPen((0, 0, 0, 128), width=1))
            line.setZValue(1)
            plot.addItem(line)

        self.bow_local_circle = DataCircle(plot, res.config.bow_rest_radius, "k", z=6)

        # x-axis is oriented tail -> tip (the plot is drawn rotated 180 degrees):
        # the tail lies at -com_arc_length and the tip at +(length - com_arc_length).
        # Arc lengths upper-bound the chord projections. Stored as the canonical
        # range so apply_layout can re-assert it (see _build_main_plot).
        self.local_x_range = (
            -res.com_arc_length - self.LOCAL_X_PADDING,
            (res.config.arrow_length - res.com_arc_length) + self.LOCAL_X_PADDING,
        )
        plot.setXRange(*self.local_x_range, padding=0)

    @staticmethod
    def _add_stretch(layout) -> None:
        """Insert an expanding spacer so neighbouring groups stay separated."""
        layout.addItem(
            QtWidgets.QSpacerItem(
                40, 0, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
            )
        )

    def _build_widgets(self) -> None:
        """Create the playback bar: play/pause, timeline scrub, speed."""
        bar = QtWidgets.QHBoxLayout()
        self.win.layout().addLayout(bar)

        self.btn_play = QtWidgets.QPushButton("Pause")
        self.btn_play.setFixedWidth(90)
        bar.addWidget(self.btn_play)

        # Expanding spacers split the bar into [transport | scrub | speed] groups,
        # so the controls do not sit shoulder to shoulder as the window resizes.
        self._add_stretch(bar)

        self.lbl_time = QtWidgets.QLabel("t = 0.00 ms")
        self.lbl_time.setMinimumWidth(120)
        bar.addWidget(self.lbl_time)

        self.slider_timeline = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_timeline.setRange(0, max(0, self.n_frames - 1))
        self.slider_timeline.setValue(0)
        bar.addWidget(self.slider_timeline, 1)

        self._add_stretch(bar)

        bar.addWidget(QtWidgets.QLabel("Speed"))
        self.slider_speed = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_speed.setRange(*self.SPEED_RANGE)
        self.slider_speed.setValue(1)
        self.slider_speed.setFixedWidth(140)
        bar.addWidget(self.slider_speed)

        self.lbl_speed = QtWidgets.QLabel("1x")
        self.lbl_speed.setMinimumWidth(40)
        bar.addWidget(self.lbl_speed)

        self.btn_reset = QtWidgets.QPushButton("Reset view")
        bar.addWidget(self.btn_reset)

    def _connect_events(self) -> None:
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider_timeline.valueChanged.connect(self.on_timeline_change)
        self.slider_speed.valueChanged.connect(self.on_speed_change)
        self.btn_reset.clicked.connect(self.reset_view)

        for plot in (self.plot_main, self.plot_local):
            plot.getViewBox().sigResized.connect(self.apply_layout)

        self.timer = QtCore.QTimer()
        self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self.timer.setInterval(self.ANIM_INTERVAL_MS)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start()

    # ==========================================
    # Layout / aspect harmonization
    # ==========================================
    def apply_layout(self, *args) -> None:
        """Give both plots the same data aspect ratio, filling the window.

        With a 1:1 locked aspect, a viewbox of ``w x h`` pixels shows a y-span
        of ``(h / w) * x_span``. So the ratio that exactly fills this plot is
        ``h / w``; we take the largest such ratio over both plots (so neither
        overflows its box) while also honouring each plot's minimum span.

        Both axes are always set in a single ``setRange`` call, and the x-range
        is always the canonical one from the data. A locked-aspect ViewBox
        silently adjusts whichever axis you leave alone, so setting only y (or
        only x) makes the other axis creep on every resize -- which is how the
        view would otherwise drift away from the data over a few resizes.
        """
        if self._laying_out:
            return
        self._laying_out = True
        try:
            specs = (
                (self.plot_main, self.main_x_range, self.main_y_min_span, self.main_y_center),
                (self.plot_local, self.local_x_range, self.LOCAL_Y_MIN_SPAN, 0.0),
            )

            # Largest ratio that still fits: over-fill for any plot would push
            # its data outside the box once the aspect lock is applied.
            r = 0.0
            for plot, x_range, min_span, _ in specs:
                size = plot.getViewBox().boundingRect().size()
                x_span = x_range[1] - x_range[0]
                if size.width() <= 0 or size.height() <= 0 or x_span <= 0:
                    continue
                r = max(r, size.height() / size.width(), min_span / x_span)

            if r > 0.0:
                for plot, x_range, _, y_center in specs:
                    # Once the user has zoomed or panned the top plot, leave its
                    # range entirely alone -- pyqtgraph owns it from then on.
                    if plot is self.plot_main and self.main_view_zoomed:
                        continue
                    y_span = r * (x_range[1] - x_range[0])
                    plot.getViewBox().setRange(
                        xRange=x_range,
                        yRange=(y_center - y_span / 2.0, y_center + y_span / 2.0),
                        padding=0,
                        disableAutoRange=True,
                    )
        finally:
            self._laying_out = False

    # ==========================================
    # Frame updates
    # ==========================================
    def update_frame(self, frame_idx: int) -> None:
        """Redraw both plots for the given frame of the simulation."""
        res = self.result
        positions = res.positions
        x = positions[frame_idx, 0, :]
        y = positions[frame_idx, 1, :]

        dx = np.gradient(x)
        dy = np.gradient(y)
        mag = np.hypot(dx, dy)
        mag[mag == 0] = NORM_EPS
        nx = -dy / mag
        ny = dx / mag

        self.shaft_top.setData(x=x + self.outer_radius * nx, y=y + self.outer_radius * ny)
        self.shaft_bot.setData(x=x - self.outer_radius * nx, y=y - self.outer_radius * ny)
        self.nodes.setData(x=x, y=y)

        end = frame_idx + 1
        self.tail_trace.setData(x=positions[:end, 0, 0], y=positions[:end, 1, 0])
        self.com_trace.setData(x=res.com_x[:end], y=res.com_y[:end])
        self.com_marker.setData(x=[res.com_x[frame_idx]], y=[res.com_y[frame_idx]])

        f_x = res.tail_forces[frame_idx, 0] * self.FORCE_SCALE
        f_y = res.tail_forces[frame_idx, 1] * self.FORCE_SCALE
        self._set_arrow(
            self.tail_force_arrow,
            x[0],
            y[0],
            f_x,
            f_y,
            max(float(np.hypot(f_x, f_y)), self.FORCE_ARROW_WIDTH * 2.0),
            self.FORCE_ARROW_WIDTH,
        )

        self._set_arrow(
            self.com_vel_arrow,
            res.com_x[frame_idx],
            res.com_y[frame_idx],
            res.com_vx[frame_idx],
            res.com_vy[frame_idx],
            self.FIXED_ARROW_LENGTH,
            self.FIXED_ARROW_WIDTH,
            head_fraction=self.DIRECTION_HEAD_FRACTION,
        )

        self._set_arrow(
            self.init_dir_arrow,
            res.com_x[frame_idx],
            res.com_y[frame_idx],
            res.init_dir[0],
            res.init_dir[1],
            self.FIXED_ARROW_LENGTH,
            self.FIXED_ARROW_WIDTH,
            head_fraction=self.DIRECTION_HEAD_FRACTION,
        )

        t_ms = float(res.times[frame_idx]) * 1000.0
        self.lbl_time.setText(f"t = {t_ms:.2f} ms")

        self._update_local_plot(frame_idx)

    def _update_local_plot(self, frame_idx: int) -> None:
        """Redraw the vibration frame: CoM origin, rotated 180 deg (tail left, tip right)."""
        res = self.result

        p_tail = res.positions[frame_idx, :2, 0]
        p_tip = res.positions[frame_idx, :2, -1]

        # Orientation ONLY: unit vector pointing from the tip toward the tail
        vec = p_tail - p_tip
        L_chord = np.linalg.norm(vec)

        if L_chord > NORM_EPS:
            u_x = vec / L_chord
            u_y = np.array([-u_x[1], u_x[0]])
        else:
            u_x = np.array([1.0, 0.0])
            u_y = np.array([0.0, 1.0])

        # Origin: mass-weighted center of mass of the current frame
        p_com = np.array([res.com_x[frame_idx], res.com_y[frame_idx]])
        rel_pos = res.positions[frame_idx, :2, :] - p_com[:, np.newaxis]

        # Rotate the frame 180 degrees so the tail is on the left and the tip on
        # the right. Negating BOTH basis vectors is a pure rotation (det = +1, so
        # the frame stays right-handed); flipping only one axis would mirror it.
        local_x = -(u_x @ rel_pos)
        local_y = -(u_y @ rel_pos)

        ldx = np.gradient(local_x)
        ldy = np.gradient(local_y)
        lmag = np.hypot(ldx, ldy)
        lmag[lmag == 0] = NORM_EPS
        lnx = -ldy / lmag
        lny = ldx / lmag

        self.local_shaft_top.setData(
            x=local_x + self.outer_radius * lnx,
            y=local_y + self.outer_radius * lny,
        )
        self.local_shaft_bot.setData(
            x=local_x - self.outer_radius * lnx,
            y=local_y - self.outer_radius * lny,
        )
        self.local_nodes.setData(x=local_x, y=local_y)

        # Bow rest center (global origin) expressed in the local CoM frame
        bow_rel = np.array([0.0, 0.0]) - p_com
        # Same 180-degree rotation as the shaft above -- the bow rest has to be
        # expressed in the SAME frame, or it would sit at the mirrored position.
        bow_lx = float(-(u_x @ bow_rel))
        bow_ly = float(-(u_y @ bow_rel))
        self.bow_local_circle.set_center(bow_lx, bow_ly)

        # Only show it once it actually reaches the plotted region
        (x0, x1), (y0, y1) = self.plot_local.getViewBox().viewRange()
        r = res.config.bow_rest_radius
        inside_view = (
            bow_lx + r >= x0
            and bow_lx - r <= x1
            and bow_ly + r >= y0
            and bow_ly - r <= y1
        )
        self.bow_local_circle.set_visible(inside_view)

    # ==========================================
    # Playback controls
    # ==========================================
    def reset_view(self, *args) -> None:
        """Undo any manual zoom/pan and go back to the harmonized default."""
        self.main_view_zoomed = False
        self.apply_layout()

    def on_timeline_change(self, value: int) -> None:
        self.current_frame = int(value)
        self.update_frame(self.current_frame)

    def on_speed_change(self, value: int) -> None:
        self.lbl_speed.setText(f"{int(value)}x")

    def toggle_play(self, *args) -> None:
        self.is_playing = not self.is_playing
        self.btn_play.setText("Pause" if self.is_playing else "Play")

    def _on_timer(self) -> None:
        if not self.is_playing:
            return
        step_inc = max(1, int(self.slider_speed.value()))
        self.current_frame = (self.current_frame + step_inc) % self.n_frames

        # Move the slider without re-entering on_timeline_change, which would
        # fight the frame index we just computed.
        self.slider_timeline.blockSignals(True)
        self.slider_timeline.setValue(self.current_frame)
        self.slider_timeline.blockSignals(False)

        self.update_frame(self.current_frame)

    # ==========================================
    # Entry points
    # ==========================================
    def show(self) -> None:
        """Show the window and hand control to Qt's event loop.

        If we are embedded in a host that already runs a Qt loop (a notebook, or
        another app that owns the QApplication), showing the window is all we
        should do -- calling exec() again would start a nested loop.
        """
        app = _ensure_app()
        self.win.show()
        if _APP_CREATED_HERE:
            app.exec()

    def save_snapshot(self, path: str) -> None:
        """Grab the current canvas to an image file (useful for headless runs)."""
        self.canvas.grab().save(path)
