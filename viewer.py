"""Interactive visualization of a simulated arrow shot.

This module owns everything that is presentation: the two plots, the widgets and
the resize handling. It consumes a :class:`~simulation.SimulationResult` and
holds no global state -- every piece of mutable state lives on the :class:`Viewer`
instance.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Circle

from simulation import SimulationResult

# Numerical guards, kept out of the class because they are hygiene, not style knobs.
NORM_EPS = 1e-6
VEL_EPS = 1e-4


class Viewer:
    """Renders a :class:`SimulationResult` with a scrub/playback GUI.

    Usage::

        Viewer(result).show()
    """

    # --- Plot sizing ---
    # Both plots keep a true 1:1 data scale (no distortion), so the on-screen box
    # shape is dictated by the x-span / y-span ratio. The two plots are harmonized
    # by giving them the SAME box shape and filling it completely, rather than by
    # hard-coding a padding. apply_layout() recomputes each y-range from the live
    # axes geometry on every resize, so the plots always expand into the available
    # space.
    FIGSIZE = (11, 6.0)
    MAIN_PADDING = 0.05  # m, x/y padding around the global data extent
    LOCAL_X_PADDING = 0.02  # m, padding on the local plot's chord axis
    LOCAL_Y_MIN_SPAN = 0.12  # m, the +/-0.06 m vibration window floor

    # --- Direction vector arrows ---
    FIXED_ARROW_LENGTH = 0.3  # m
    FIXED_ARROW_WIDTH = 0.0015

    # --- Tail force vector ---
    FORCE_SCALE = 0.0008
    FORCE_ARROW_WIDTH = 0.003

    # --- Layout, in absolute inches (see apply_layout) ---
    M_LEFT = 0.95  # y-label
    M_TOP = 0.45  # plot 1 title
    M_BOTTOM = 1.95  # x-label + sliders
    GAP_IN = 1.05  # between the two plots
    M_RIGHT_BASE = 2.45  # legend column
    M_RIGHT_FRAC = 0.03
    BTN_H_IN = 0.32
    SLD_H_IN = 0.28
    Y_BTN_IN = 0.16

    # --- Widgets / animation ---
    SPEED_RANGE = (0.2, 5.0)
    ANIM_INTERVAL_MS = 30

    def __init__(self, result: SimulationResult):
        self.result = result
        cfg = result.config
        self.outer_radius = cfg.outer_radius
        self.bow_rest_radius = cfg.bow_rest_radius
        self.brace_height = cfg.brace_height
        self.arrow_length = cfg.arrow_length

        # Playback state (instance-level, deliberately not global).
        self.is_playing = True
        self.current_frame = 0
        self.n_frames = len(result.times)
        self._ani = None

        # Layout is driven by absolute-inch margins (see apply_layout) so that
        # titles, axis labels and the sliders never overlap, at any window size.
        # Fractions would shrink on a short window and clip the headings.
        self.fig, (self.ax_main, self.ax_local) = plt.subplots(
            2, 1, figsize=self.FIGSIZE, gridspec_kw={"height_ratios": [1, 1]}
        )

        self._build_main_plot()
        self._build_local_plot()
        self._build_widgets()
        self._connect_events()

    # ==========================================
    # Plots
    # ==========================================
    def _build_main_plot(self) -> None:
        """Build the global-frame plot: bow rest, shaft, traces and arrows."""
        res = self.result
        ax = self.ax_main

        self.bow_circle = Circle(
            (0, 0), self.bow_rest_radius, color="black", alpha=0.7, label="Bow Rest (0,0)"
        )
        ax.add_patch(self.bow_circle)
        ax.axvline(
            x=-self.brace_height,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Brace Height (-{self.brace_height}m)",
        )

        self.arrow_shaft_poly = Polygon(
            np.empty((0, 2)), color="tab:blue", alpha=0.5, label="Arrow Shaft (True Width)"
        )
        ax.add_patch(self.arrow_shaft_poly)
        (self.arrow_nodes,) = ax.plot([], [], "o", color="navy", markersize=3, label="Arrow Nodes")

        (self.tail_trace,) = ax.plot([], [], ":", color="tab:gray", alpha=0.5, label="Tail Path")

        # CoM Marker & Trajectory in Blue
        (self.com_marker,) = ax.plot([], [], "b.", markersize=10, label="Center of Mass")
        (self.com_trace,) = ax.plot(
            [], [], "--", color="b", alpha=0.7, lw=1.5, label="CoM Trajectory"
        )

        # CoM Velocity Direction Arrow
        self.com_vel_arrow = ax.quiver(
            [0],
            [0],
            [0],
            [0],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=self.FIXED_ARROW_WIDTH,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=4.0,
            color="blue",
            alpha=0.8,
            label="CoM Velocity Direction",
        )

        # Initial Direction Arrow (Black, fixed length)
        self.init_dir_arrow = ax.quiver(
            [0],
            [0],
            [0],
            [0],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=self.FIXED_ARROW_WIDTH,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=4.0,
            color="black",
            alpha=0.8,
            label="Initial Arrow Direction",
        )

        # Tail Force Vector Quiver Arrow
        self.tail_force_arrow = ax.quiver(
            [0],
            [0],
            [0],
            [0],
            angles="xy",
            scale_units="xy",
            scale=1,
            width=self.FORCE_ARROW_WIDTH,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=4.0,
            color="red",
            alpha=0.8,
            label="Tail Force",
        )

        self.time_text = ax.text(
            0.02,
            0.92,
            "",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1),
        )

        all_x = res.positions[:, 0, :]
        all_y = res.positions[:, 1, :]
        ax.set_xlim(np.min(all_x) - self.MAIN_PADDING, np.max(all_x) + self.MAIN_PADDING)

        self.main_y_center = 0.5 * (np.min(all_y) + np.max(all_y))
        self.main_y_min_span = (
            (np.max(all_y) + self.MAIN_PADDING) - (np.min(all_y) - self.MAIN_PADDING)
        )
        ax.set_ylim(
            self.main_y_center - self.main_y_min_span / 2.0,
            self.main_y_center + self.main_y_min_span / 2.0,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_xlabel("X Position [m]")
        ax.set_ylabel("Y Deflection [m]")
        ax.set_title(self._title())
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    def _title(self) -> str:
        cfg = self.result.config
        return (
            f"Archer's Paradox ({cfg.draw_weight_lbs} lbs draw, "
            f"{cfg.spine_value} spine, {cfg.tip_weight_grains:.0f} gr tip)"
        )

    def _build_local_plot(self) -> None:
        """Build the vibration-frame plot: the same arrow seen from its own CoM."""
        res = self.result
        ax = self.ax_local

        self.arrow_local_poly = Polygon(
            np.empty((0, 2)), color="tab:blue", alpha=0.5, label="Local Shaft (True Width)"
        )
        ax.add_patch(self.arrow_local_poly)
        (self.arrow_local_nodes,) = ax.plot(
            [], [], "o", color="navy", markersize=3, label="Local Nodes"
        )

        # CoM reference cross. Intentionally unlabelled: these are frame axes, not
        # data, so they are left out of the legend to keep it focused on the arrow.
        ax.axhline(0, color="black", lw=1, alpha=0.5)
        ax.axvline(0, color="black", lw=1, alpha=0.5)

        # Bow rest in the local frame. The local transform is a rigid rotation +
        # translation (unit basis vectors), so the circle keeps its radius and only
        # its center moves. Styled and labelled to match the bow rest legend in the
        # main plot above.
        self.bow_local_circle = Circle(
            (0, 0),
            self.bow_rest_radius,
            color="black",
            alpha=0.7,
            label="Bow Rest (0,0)",
            zorder=5,
        )
        self.bow_local_circle.set_visible(False)
        ax.add_patch(self.bow_local_circle)

        # x-axis is oriented tail -> tip (the plot is drawn rotated 180 degrees):
        # the tail lies at -com_arc_length and the tip at +(length - com_arc_length).
        # Arc lengths upper-bound the chord projections.
        ax.set_xlim(
            -res.com_arc_length - self.LOCAL_X_PADDING,
            (self.arrow_length - res.com_arc_length) + self.LOCAL_X_PADDING,
        )

        ax.set_ylim(-self.LOCAL_Y_MIN_SPAN / 2.0, self.LOCAL_Y_MIN_SPAN / 2.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_xlabel("Chord Position from CoM [m]  (tail -> tip)")
        ax.set_ylabel("Transverse Deflection [m]")
        ax.set_title("Arrow Vibrations (Local CoM Reference Frame, Tail-to-Tip Orientation)")
        # No legend here: every entry would repeat what the main plot above already
        # shows, and the arrow's colours/width match it. The width comes from the
        # shared subplot margins, so dropping the legend does not resize this plot.

    def _build_widgets(self) -> None:
        """Create the widgets; their real positions are set by apply_layout()."""
        self.ax_timeline = plt.axes([0.0, 0.0, 0.0, 0.0])
        self.ax_play = plt.axes([0.0, 0.0, 0.0, 0.0])
        self.ax_speed = plt.axes([0.0, 0.0, 0.0, 0.0])

        self.slider_timeline = Slider(
            self.ax_timeline, "Time", 0, self.result.times[-1] * 1000, valinit=0, valfmt="%.1f ms"
        )
        self.btn_play = Button(self.ax_play, "Pause", hovercolor="0.97")
        self.slider_speed = Slider(
            self.ax_speed,
            "Speed",
            self.SPEED_RANGE[0],
            self.SPEED_RANGE[1],
            valinit=1.0,
            valfmt="%.1fx",
        )

    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect("resize_event", self.apply_layout)
        self.slider_timeline.on_changed(self.on_timeline_change)
        self.btn_play.on_clicked(self.toggle_play)
        self.apply_layout()

    # ==========================================
    # Resize Handling
    # ==========================================
    def apply_layout(self, event=None) -> None:
        """Lay the figure out in absolute inches so nothing overlaps or clips.

        Margins are expressed in inches rather than figure fractions: a fraction of
        a short window is far too small for a title or an axis label, which is
        what clipped plot 1's heading and pushed plot 2's x-label onto the time
        slider. Every resize recomputes them, so the headings and the widget row
        keep their clearance.
        """
        fig = self.fig
        fig_w, fig_h = fig.get_size_inches()
        if fig_w <= 0 or fig_h <= 0:
            return

        left = self.M_LEFT / fig_w
        # The legend is anchored 2% of the axes width to the right of the axes, so
        # the legend column also has to cover that offset, which grows with the window.
        m_right = self.M_RIGHT_BASE + self.M_RIGHT_FRAC * (fig_w - self.M_LEFT - self.M_RIGHT_BASE)
        right = 1.0 - m_right / fig_w
        bottom = self.M_BOTTOM / fig_h
        top = 1.0 - self.M_TOP / fig_h

        # hspace is a fraction of the average axes height, so convert the inch gap.
        axes_h_in = ((top - bottom) * fig_h) / 2.0
        hspace = self.GAP_IN / axes_h_in
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hspace)

        # --- Slider / button row, stacked upward from the bottom of the figure ---
        plot_span = right - left
        btn_h_in, sld_h_in = self.BTN_H_IN, self.SLD_H_IN
        y_btn = self.Y_BTN_IN / fig_h
        y_sld_tl = (self.Y_BTN_IN + btn_h_in + 0.22) / fig_h  # timeline sits above the button row

        self.ax_timeline.set_position(
            [left + 0.04 * plot_span, y_sld_tl, 0.92 * plot_span, sld_h_in / fig_h]
        )
        self.ax_play.set_position([left + 0.04 * plot_span, y_btn, 0.13 * plot_span, btn_h_in / fig_h])
        self.ax_speed.set_position(
            [left + 0.35 * plot_span, y_btn + 0.05 / fig_h, 0.60 * plot_span, 0.22 / fig_h]
        )

        # --- Make each plot's box fill the space it was given ---
        # With an equal aspect, a given box shape plus a fixed x-span fully
        # determines the visible y-span. Deriving the y-span from the allotted box
        # lets the plots expand with the window instead of leaving a margin.
        #
        # Both plots share ONE y-span-per-x-span ratio r, so their data aspect
        # ratios are identical and their boxes always come out the same shape
        # (harmonized). r is the largest value that still fills the allotted area
        # and still honours each plot's minimum span (the data extent, and the
        # +/-0.06 m vibration window).
        specs = (
            (self.ax_main, self.main_y_min_span, self.main_y_center),
            (self.ax_local, self.LOCAL_Y_MIN_SPAN, 0.0),
        )

        r = 0.0
        for ax, min_span, _ in specs:
            box = ax.get_position(original=True)  # full area allotted to this axes
            x_lo, x_hi = ax.get_xlim()
            if box.width <= 0 or box.height <= 0 or (x_hi - x_lo) <= 0:
                continue
            r_fill = (box.height * fig_h) / (box.width * fig_w)
            r_min = min_span / (x_hi - x_lo)
            r = max(r, r_fill, r_min)

        if r > 0:
            for ax, _, y_center in specs:
                x_lo, x_hi = ax.get_xlim()
                y_span = r * (x_hi - x_lo)
                ax.set_ylim(y_center - y_span / 2.0, y_center + y_span / 2.0)

        fig.canvas.draw_idle()

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

        top_x = x + self.outer_radius * nx
        top_y = y + self.outer_radius * ny
        bot_x = x - self.outer_radius * nx
        bot_y = y - self.outer_radius * ny

        poly_verts = np.column_stack(
            (np.concatenate([top_x, bot_x[::-1]]), np.concatenate([top_y, bot_y[::-1]]))
        )

        self.arrow_shaft_poly.set_xy(poly_verts)
        self.arrow_nodes.set_data(x, y)

        tail_x = positions[: frame_idx + 1, 0, 0]
        tail_y = positions[: frame_idx + 1, 1, 0]
        self.tail_trace.set_data(tail_x, tail_y)
        self.com_marker.set_data([res.com_x[frame_idx]], [res.com_y[frame_idx]])
        self.com_trace.set_data(res.com_x[: frame_idx + 1], res.com_y[: frame_idx + 1])

        # Update Tail Force Vector Arrow
        f_x = res.tail_forces[frame_idx, 0] * self.FORCE_SCALE
        f_y = res.tail_forces[frame_idx, 1] * self.FORCE_SCALE
        self.tail_force_arrow.set_offsets(np.c_[x[0], y[0]])
        self.tail_force_arrow.set_UVC(f_x, f_y)

        # Update CoM Velocity Direction Arrow
        vx = res.com_vx[frame_idx]
        vy = res.com_vy[frame_idx]
        v_mag = np.hypot(vx, vy)
        if v_mag > VEL_EPS:
            vel_dir_x = (vx / v_mag) * self.FIXED_ARROW_LENGTH
            vel_dir_y = (vy / v_mag) * self.FIXED_ARROW_LENGTH
        else:
            vel_dir_x, vel_dir_y = 0.0, 0.0

        self.com_vel_arrow.set_offsets(np.c_[res.com_x[frame_idx], res.com_y[frame_idx]])
        self.com_vel_arrow.set_UVC(vel_dir_x, vel_dir_y)

        # Update Initial Direction Arrow
        init_x = res.init_dir[0] * self.FIXED_ARROW_LENGTH
        init_y = res.init_dir[1] * self.FIXED_ARROW_LENGTH
        self.init_dir_arrow.set_offsets(np.c_[res.com_x[frame_idx], res.com_y[frame_idx]])
        self.init_dir_arrow.set_UVC(init_x, init_y)

        self.time_text.set_text(f"t = {res.times[frame_idx] * 1000:.2f} ms")

        self._update_local_plot(frame_idx)

    def _update_local_plot(self, frame_idx: int) -> None:
        """Redraw the vibration frame: CoM origin, rotated 180 deg (tail left, tip right)."""
        res = self.result

        # Local Plot Update (CoM origin, rotated 180 deg: tail left, tip right)
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
        # Rotate the frame 180 degrees so the tail is on the left and the tip on the
        # right. Negating BOTH basis vectors is a pure rotation (det = +1, so the
        # frame stays right-handed); flipping only one axis would mirror it.
        local_x = -(u_x @ rel_pos)
        local_y = -(u_y @ rel_pos)

        ldx = np.gradient(local_x)
        ldy = np.gradient(local_y)
        lmag = np.hypot(ldx, ldy)
        lmag[lmag == 0] = NORM_EPS
        lnx = -ldy / lmag
        lny = ldx / lmag

        ltop_x = local_x + self.outer_radius * lnx
        ltop_y = local_y + self.outer_radius * lny
        lbot_x = local_x - self.outer_radius * lnx
        lbot_y = local_y - self.outer_radius * lny

        lpoly_verts = np.column_stack(
            (
                np.concatenate([ltop_x, lbot_x[::-1]]),
                np.concatenate([ltop_y, lbot_y[::-1]]),
            )
        )

        self.arrow_local_poly.set_xy(lpoly_verts)
        self.arrow_local_nodes.set_data(local_x, local_y)

        # Bow rest center (global origin) expressed in the local CoM frame
        bow_rel = np.array([0.0, 0.0]) - p_com
        # Same 180-degree rotation as the shaft above -- the bow rest has to be
        # expressed in the SAME frame, or it would sit at the mirrored position.
        bow_lx = float(-(u_x @ bow_rel))
        bow_ly = float(-(u_y @ bow_rel))
        self.bow_local_circle.center = (bow_lx, bow_ly)

        # Only show it once it actually reaches the plotted region
        x0, x1 = self.ax_local.get_xlim()
        y0, y1 = self.ax_local.get_ylim()
        inside_view = (
            bow_lx + self.bow_rest_radius >= x0
            and bow_lx - self.bow_rest_radius <= x1
            and bow_ly + self.bow_rest_radius >= y0
            and bow_ly - self.bow_rest_radius <= y1
        )
        self.bow_local_circle.set_visible(inside_view)

        self.fig.canvas.draw_idle()

    # ==========================================
    # Playback controls
    # ==========================================
    def on_timeline_change(self, val) -> None:
        target_time = val / 1000.0
        self.current_frame = np.searchsorted(self.result.times, target_time)
        self.current_frame = min(self.current_frame, self.n_frames - 1)
        self.update_frame(self.current_frame)

    def toggle_play(self, event=None) -> None:
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text("Pause" if self.is_playing else "Play")

    def _animate(self, frame) -> None:
        if self.is_playing:
            step_inc = max(1, int(self.slider_speed.val))
            self.current_frame = (self.current_frame + step_inc) % self.n_frames

            # Move the slider without re-entering on_timeline_change, which would
            # fight the frame index we just computed.
            self.slider_timeline.eventson = False
            self.slider_timeline.set_val(self.result.times[self.current_frame] * 1000)
            self.slider_timeline.eventson = True

            self.update_frame(self.current_frame)

    def show(self) -> None:
        """Show the figure and hand control to matplotlib's event loop."""
        # Keep a reference alive: FuncAnimation is only kept alive by its callbacks.
        self._ani = FuncAnimation(
            self.fig, self._animate, interval=self.ANIM_INTERVAL_MS, cache_frame_data=False
        )
        plt.show()
