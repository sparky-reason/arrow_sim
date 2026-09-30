import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Circle

# PyElastica core imports
import elastica as ea
from elastica.rod.cosserat_rod import CosseratRod
from elastica.rigidbody import Cylinder
from elastica.boundary_conditions import FixedConstraint
from elastica.contact_forces import RodCylinderContact
from elastica.callback_functions import CallBackBaseClass
from elastica.dissipation import AnalyticalLinearDamper
from elastica.timestepper import integrate
from elastica.timestepper.symplectic_steppers import PositionVerlet


# ==========================================
# 1. Custom Aerodynamic Drag Force Class
# ==========================================
class ArrowAerodynamics(ea.NoForces):
    def __init__(self, rho_air=1.225, Cd_shaft=1.0, Cd_fletching=1.2, fletching_area=0.002):
        super().__init__()
        self.rho = rho_air  # kg/m^3
        self.Cd_shaft = Cd_shaft  # dimensionless
        self.Cd_fletching = Cd_fletching  # dimensionless
        self.A_fletching = fletching_area  # m^2

    def apply_forces(self, system, time=0.0):
        velocities = system.velocity_collection
        tangents = system.tangents
        radii = system.radius
        lengths = system.lengths
        n_elems = system.n_elems

        for i in range(n_elems):
            v_node = 0.5 * (velocities[:, i] + velocities[:, i + 1])
            v_mag = np.linalg.norm(v_node)

            if v_mag > 1e-4:  # m/s
                t = tangents[:, i]
                v_normal = v_node - np.dot(v_node, t) * t
                v_normal_mag = np.linalg.norm(v_normal)
                area = 2.0 * radii[i] * lengths[i]  # m^2
                drag_force = -0.5 * self.rho * self.Cd_shaft * area * v_normal_mag * v_normal  # N

                system.external_forces[:, i] += 0.5 * drag_force
                system.external_forces[:, i + 1] += 0.5 * drag_force

        v_tail = velocities[:, 0]
        v_tail_mag = np.linalg.norm(v_tail)
        if v_tail_mag > 1e-4:  # m/s
            fletching_drag = -0.5 * self.rho * self.Cd_fletching * self.A_fletching * v_tail_mag * v_tail  # N
            system.external_forces[:, 0] += fletching_drag


# ==========================================
# 2. Angled String Push Force (Cosine Target Decay)
# ==========================================
class StringPushForce(ea.NoForces):
    def __init__(
        self,
        f_max,
        draw_length,
        brace_height,
        c_string,
        pluck_angle,
        pluck_decay_length,
    ):
        super().__init__()
        self.f_max = f_max  # N
        self.draw_length = draw_length  # m
        self.brace_height = brace_height  # m
        self.c_string = c_string  # N*s/m (transverse string damping)
        self.pluck_angle = pluck_angle  # rad
        self.pluck_decay_length = pluck_decay_length  # m
        self.initialized = False

    def apply_forces(self, system, time=0.0):
        pos_x = system.position_collection[0, 0]  # m
        pos_y = system.position_collection[1, 0]  # m
        v_y = system.velocity_collection[1, 0]    # m/s

        if pos_x < -self.brace_height:
            # Distance traveled by nock from draw position (-draw_length)
            travel_distance = pos_x - (-self.draw_length)
            travel_distance = max(0.0, travel_distance)

            # Compute lateral target y-offset using cosine decay
            stroke_length = self.draw_length - self.brace_height
            y_0 = stroke_length * np.tan(self.pluck_angle)

            if travel_distance < self.pluck_decay_length:
                # Cosine decay from y_0 at travel=0 to 0.0 at travel=decay_length
                y_target = y_0 * np.cos((np.pi / 2.0) * (travel_distance / self.pluck_decay_length))
            else:
                y_target = 0.0

            # Main thrust force vector targeting (-brace_height, y_target)
            stroke_factor = (-self.brace_height - pos_x) / stroke_length
            stroke_factor = max(0.0, min(1.0, stroke_factor))
            f_mag = self.f_max * stroke_factor  # N

            dx = -self.brace_height - pos_x  # m
            dy = y_target - pos_y  # m
            mag_r = np.hypot(dx, dy)  # m

            if mag_r > 1e-6:
                system.external_forces[0, 0] += f_mag * (dx / mag_r)  # N
                system.external_forces[1, 0] += f_mag * (dy / mag_r)  # N

            # Stable Transverse String Damping
            system.external_forces[1, 0] += -self.c_string * v_y  # N


# ==========================================
# 3. Simulation Environment Setup
# ==========================================
class ArrowSimulation(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
    ea.Contact,
):
    pass


arrow_sim = ArrowSimulation()

# ==========================================
# 4. Physical & Geometric Setup
# ==========================================
draw_length = 0.70  # m (70 cm draw length)
brace_height = 0.18  # m (18 cm brace height)
length = 0.75  # m (75 cm arrow length)

# Tip Mass
tip_weight_grains = 100.0  # grains
GRAINS_TO_KG = 0.00006479891  # kg/grain
tip_mass_kg = tip_weight_grains * GRAINS_TO_KG  # kg

# String Mass Proxy
string_effective_mass_kg = 0.0035  # kg (3.5 grams of moving string mass)

n_elements = 30  # dimensionless
outer_radius = 0.00375  # m (3.75 mm radius)
density = 500.0  # kg/m^3

# Spine to E_modulus conversion
spine_value = 400.0  # dimensionless
deflection_m = (spine_value / 1000.0) * 0.0254  # m
span_L = 28.0 * 0.0254  # m
test_force_N = 1.94 * 4.44822  # N
I_beam = (np.pi / 4.0) * (outer_radius ** 4)  # m^4
E_modulus = (test_force_N * (span_L ** 3)) / (48.0 * deflection_m * I_beam)  # Pa

bow_rest_radius = 0.015  # m
bow_rest_height = 0.1  # m

R_eff = bow_rest_radius + outer_radius  # m
theta = np.arcsin(R_eff / draw_length)  # rad

direction = np.array([np.cos(theta), np.sin(theta), 0.0])
normal = np.array([-np.sin(theta), np.cos(theta), 0.0])
start_position = np.array([-draw_length, 0.0, 0.0])  # m

arrow = CosseratRod.straight_rod(
    n_elements=n_elements,
    start=start_position,
    direction=direction,
    normal=normal,
    base_length=length,
    base_radius=outer_radius,
    density=density,
    youngs_modulus=E_modulus,
    shear_modulus=E_modulus / 2.6,  # Pa
)


shaft_mass_kg = np.sum(arrow.mass)
print(f"--- Arrow Mass Breakdown ---")
print(f"Shaft Mass: {shaft_mass_kg * 1000.0:.2f} grams")
print(f"Tip Mass:   {tip_mass_kg * 1000.0:.2f} grams")
print(f"Arrow Mass: {(shaft_mass_kg + tip_mass_kg) * 1000.0:.2f} grams")

arrow.mass[-1] += tip_mass_kg  # kg
arrow.mass[0] += string_effective_mass_kg  # kg

arrow_sim.append(arrow)

arrow_sim.dampen(arrow).using(
    AnalyticalLinearDamper,
    damping_constant=0.01,  # kg/s
    time_step=1e-6,  # s
)

# ==========================================
# 5. Bow Rest & Contact Setup
# ==========================================
bow_rest = Cylinder(
    start=np.array([0.0, 0.0, -bow_rest_height / 2.0]),  # m
    direction=np.array([0.0, 0.0, 1.0]),
    normal=np.array([1.0, 0.0, 0.0]),
    base_length=bow_rest_height,  # m
    base_radius=bow_rest_radius,  # m
    density=1000.0,  # kg/m^3
)

arrow_sim.append(bow_rest)

arrow_sim.constrain(bow_rest).using(
    FixedConstraint,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
)

arrow_sim.detect_contact_between(arrow, bow_rest).using(
    RodCylinderContact,
    k=1e5,  # N/m
    nu=1.0,  # kg/s
)

# ==========================================
# 6. Apply Dynamic String Force & Aerodynamics
# ==========================================
draw_weight_lbs = 25.0  # lbs
LBS_TO_N = 4.44822  # N/lb
push_force_magnitude = draw_weight_lbs * LBS_TO_N  # N

# Kinematic finger release parameters
initial_pluck_angle_rad = np.radians(8)  # initial lateral angle
pluck_decay_length_m = 0.025  # m

arrow_sim.add_forcing_to(arrow).using(
    StringPushForce,
    f_max=push_force_magnitude,
    draw_length=draw_length,
    brace_height=brace_height,
    c_string=0.4,
    pluck_angle=initial_pluck_angle_rad,
    pluck_decay_length=pluck_decay_length_m,
)

arrow_sim.add_forcing_to(arrow).using(
    ArrowAerodynamics,
    rho_air=1.225,
    Cd_shaft=1.0,
    Cd_fletching=1.2,
    fletching_area=0.002,
)

# ==========================================
# 7. Diagnostics Callback (Logs Tail Force Vector)
# ==========================================
class ArrowCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        super().__init__()
        self.step_skip = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step):
        if current_step % self.step_skip == 0:
            self.callback_params["time"].append(time)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["tail_force"].append(system.external_forces[:, 0].copy())


data_tracker = {"time": [], "position": [], "tail_force": []}
arrow_sim.collect_diagnostics(arrow).using(
    ArrowCallBack, step_skip=250, callback_params=data_tracker
)

# Finalize and integrate
arrow_sim.finalize()
timestepper = PositionVerlet()

n_steps = 60_000
final_time = 0.03  # s

print(f"Simulating shot ({draw_weight_lbs} lbs draw, {spine_value} spine)...")
integrate(timestepper, arrow_sim, final_time, n_steps)
print("Simulation complete!")

# ==========================================
# 8. Compute Mass-Weighted Center of Mass (CoM) & Velocity
# ==========================================
positions = np.array(data_tracker["position"])
tail_forces = np.array(data_tracker["tail_force"])  # Shape: (n_frames, 3) in N
times = np.array(data_tracker["time"])
n_frames = len(times)

node_masses = arrow.mass.copy()
total_mass = np.sum(node_masses)

com_x = np.sum(positions[:, 0, :] * node_masses, axis=1) / total_mass
com_y = np.sum(positions[:, 1, :] * node_masses, axis=1) / total_mass

# Center of Mass Velocity (via finite differences)
dt = times[1] - times[0] if n_frames > 1 else 1.0
com_vx = np.gradient(com_x, dt)
com_vy = np.gradient(com_y, dt)

# Fixed arrow length for direction vectors
fixed_arrow_length = 0.3  # m
fixed_arrow_width = 0.0015

# Initial direction vector
init_dir = direction[:2] / np.linalg.norm(direction[:2])

# Arc-length position of the CoM measured from the tail (m).
# Used only to set sensible limits on the local (CoM) plot's x-axis.
node_arc_length = (np.arange(len(node_masses)) / (len(node_masses) - 1)) * length
com_arc_length = float(np.sum(node_arc_length * node_masses) / total_mass)

# ==========================================
# 9. Interactive Visualization & GUI
# ==========================================
# Layout is driven by absolute-inch margins (see apply_layout in section 10) so that
# titles, axis labels and the sliders never overlap, at any window size. Fractions
# would shrink on a short window and clip the headings.
fig, (ax_main, ax_local) = plt.subplots(2, 1, figsize=(11, 6.0), gridspec_kw={'height_ratios': [1, 1]})

# --- MAIN PLOT (GLOBAL FRAME) ---
bow_circle = Circle((0, 0), bow_rest_radius, color="black", alpha=0.7, label="Bow Rest (0,0)")
ax_main.add_patch(bow_circle)
ax_main.axvline(x=-brace_height, color="red", linestyle="--", alpha=0.7, label=f"Brace Height (-{brace_height}m)")

arrow_shaft_poly = Polygon(np.empty((0, 2)), color='tab:blue', alpha=0.5, label="Arrow Shaft (True Width)")
ax_main.add_patch(arrow_shaft_poly)
(arrow_nodes,) = ax_main.plot([], [], 'o', color='navy', markersize=3, label="Arrow Nodes")

(tail_trace,) = ax_main.plot([], [], ':', color='tab:gray', alpha=0.5, label="Tail Path")

# CoM Marker & Trajectory in Blue
(com_marker,) = ax_main.plot([], [], 'b.', markersize=10, label="Center of Mass")
(com_trace,) = ax_main.plot([], [], '--', color='b', alpha=0.7, lw=1.5, label="CoM Trajectory")

# CoM Velocity Direction Arrow
com_vel_arrow = ax_main.quiver(
    [0], [0], [0], [0],
    angles='xy',
    scale_units='xy',
    scale=1,
    width=fixed_arrow_width,
    headwidth=3.5,
    headlength=4.5,
    headaxislength=4.0,
    color='blue',
    alpha=0.8,
    label="CoM Velocity Direction"
)

# Initial Direction Arrow (Black, half force arrow width, fixed length)
init_dir_arrow = ax_main.quiver(
    [0], [0], [0], [0],
    angles='xy',
    scale_units='xy',
    scale=1,
    width=fixed_arrow_width,
    headwidth=3.5,
    headlength=4.5,
    headaxislength=4.0,
    color='black',
    alpha=0.8,
    label="Initial Arrow Direction"
)

# Tail Force Vector Quiver Arrow
force_scale = 0.0008
force_arrow_width = 0.003
tail_force_arrow = ax_main.quiver(
    [0], [0], [0], [0],
    angles='xy',
    scale_units='xy',
    scale=1,
    width=force_arrow_width,
    headwidth=3.5,
    headlength=4.5,
    headaxislength=4.0,
    color='red',
    alpha=0.8,
    label="Tail Force"
)

time_text = ax_main.text(
    0.02, 0.92, '', transform=ax_main.transAxes, fontsize=11, fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1)
)

all_x = positions[:, 0, :]
all_y = positions[:, 1, :]
# --- Plot sizing ---
# Both plots keep a true 1:1 data scale (no distortion), so the on-screen box shape
# is dictated by the x-span / y-span ratio. The two plots are therefore harmonized
# by giving them the SAME box shape and filling it completely, rather than by
# hard-coding a padding. apply_layout() recomputes each y-range from the live axes
# geometry on every resize, so the plots always expand into the available space.
ax_main.set_xlim(np.min(all_x) - 0.05, np.max(all_x) + 0.05)

main_y_center = 0.5 * (np.min(all_y) + np.max(all_y))
main_y_min_span = (np.max(all_y) + 0.05) - (np.min(all_y) - 0.05)
ax_main.set_ylim(main_y_center - main_y_min_span / 2.0, main_y_center + main_y_min_span / 2.0)

ax_main.set_aspect('equal', adjustable='box')
ax_main.grid(True)
ax_main.set_xlabel("X Position [m]")
ax_main.set_ylabel("Y Deflection [m]")
ax_main.set_title(f"Archer's Paradox ({draw_weight_lbs} lbs draw, {spine_value} spine, {tip_weight_grains:.0f} gr tip)")
ax_main.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

# --- LOCAL PLOT (VIBRATION FRAME) ---
arrow_local_poly = Polygon(np.empty((0, 2)), color='tab:blue', alpha=0.5, label="Local Shaft (True Width)")
ax_local.add_patch(arrow_local_poly)
(arrow_local_nodes,) = ax_local.plot([], [], 'o', color='navy', markersize=3, label="Local Nodes")

# CoM reference cross. Intentionally unlabelled: these are frame axes, not data, so
# they are left out of the legend to keep it focused on the arrow itself.
ax_local.axhline(0, color='black', lw=1, alpha=0.5)
ax_local.axvline(0, color='black', lw=1, alpha=0.5)

# Bow rest in the local frame. The local transform is a rigid rotation + translation
# (unit basis vectors), so the circle keeps its radius and only its center moves.
# Styled and labelled to match the bow rest legend in the main plot above.
bow_local_circle = Circle(
    (0, 0), bow_rest_radius, color="black", alpha=0.7, label="Bow Rest (0,0)", zorder=5
)
bow_local_circle.set_visible(False)
ax_local.add_patch(bow_local_circle)

# x-axis is oriented tip -> tail: the tip lies at -(length - com_arc_length),
# the tail at +com_arc_length. Arc lengths upper-bound the chord projections.
ax_local.set_xlim(-(length - com_arc_length) - 0.02, com_arc_length + 0.02)

# Minimum y-span: the +/-0.06 m vibration window. apply_layout() treats this as a floor
# so the window stays readable, while still letting the plot grow on a larger window.
local_y_min_span = 0.12
ax_local.set_ylim(-local_y_min_span / 2.0, local_y_min_span / 2.0)
ax_local.set_aspect('equal', adjustable='box')
ax_local.grid(True)
ax_local.set_xlabel("Chord Position from CoM [m]  (tip -> tail)")
ax_local.set_ylabel("Transverse Deflection [m]")
ax_local.set_title("Arrow Vibrations (Local CoM Reference Frame, Tip-to-Tail Orientation)")
# No legend here: every entry would repeat what the main plot above already shows, and
# the arrow's colours/width match it. The width comes from the shared subplot margins,
# so dropping the legend does not resize this plot.

# Widget Axes Placement. Created here so the widgets can be constructed, but their
# real positions are set by apply_layout() in absolute inches, which keeps them inside
# the reserved bottom band at any window size.
ax_timeline = plt.axes([0.0, 0.0, 0.0, 0.0])
ax_play = plt.axes([0.0, 0.0, 0.0, 0.0])
ax_speed = plt.axes([0.0, 0.0, 0.0, 0.0])

slider_timeline = Slider(ax_timeline, 'Time', 0, times[-1] * 1000, valinit=0, valfmt='%.1f ms')
btn_play = Button(ax_play, 'Pause', hovercolor='0.97')
slider_speed = Slider(ax_speed, 'Speed', 0.2, 5.0, valinit=1.0, valfmt='%.1fx')


# ==========================================
# 10. Resize Handling
# ==========================================
def apply_layout(event=None):
    """Lay the figure out in absolute inches so nothing overlaps or clips.

    Margins are expressed in inches rather than figure fractions: a fraction of a
    short window is far too small for a title or an axis label, which is what clipped
    plot 1's heading and pushed plot 2's x-label onto the time slider. Every resize
    recomputes them, so the headings and the widget row keep their clearance.
    """
    fig_w, fig_h = fig.get_size_inches()
    if fig_w <= 0 or fig_h <= 0:
        return

    # Inches reserved outside the plot area for labels, titles and widgets.
    m_left, m_top, m_bottom = 0.95, 0.45, 1.95   # y-label | plot 1 title | x-label + sliders
    # Gap between the two plots must clear plot 1's tick labels + x-label above plot 2's
    # title, otherwise the two collide.
    gap_in = 1.05

    # The legend is anchored 2% of the axes width to the right of the axes, so the
    # legend column also has to cover that offset, which itself grows with the window.
    m_right = 2.45 + 0.03 * (fig_w - m_left - 2.45)

    left = m_left / fig_w
    right = 1.0 - m_right / fig_w
    bottom = m_bottom / fig_h
    top = 1.0 - m_top / fig_h

    # hspace is a fraction of the average axes height, so convert the inch gap.
    axes_h_in = ((top - bottom) * fig_h) / 2.0
    hspace = gap_in / axes_h_in
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hspace)

    # --- Slider / button row, stacked upward from the bottom of the figure ---
    plot_span = right - left
    btn_h_in, sld_h_in = 0.32, 0.28
    y_btn = 0.16 / fig_h
    y_sld_tl = (0.16 + btn_h_in + 0.22) / fig_h   # timeline sits above the button row

    ax_timeline.set_position([left + 0.04 * plot_span, y_sld_tl, 0.92 * plot_span, sld_h_in / fig_h])
    ax_play.set_position([left + 0.04 * plot_span, y_btn, 0.13 * plot_span, btn_h_in / fig_h])
    ax_speed.set_position([left + 0.35 * plot_span, y_btn + 0.05 / fig_h, 0.60 * plot_span, 0.22 / fig_h])

    # --- Make each plot's box fill the space it was given ---
    # With an equal aspect, a given box shape plus a fixed x-span fully determines the
    # visible y-span. Deriving the y-span from the allotted box lets the plots expand
    # with the window instead of leaving a margin.
    #
    # Both plots share ONE y-span-per-x-span ratio r, so their data aspect ratios are
    # identical and their boxes always come out the same shape (harmonized). r is the
    # largest value that still fills the allotted area and still honours each plot's
    # minimum span (the data extent, and the +/-0.06 m vibration window).
    specs = (
        (ax_main, main_y_min_span, main_y_center),
        (ax_local, local_y_min_span, 0.0),
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


fig.canvas.mpl_connect('resize_event', apply_layout)
apply_layout()

is_playing = True
current_frame = 0


def update_frame(frame_idx):
    x = positions[frame_idx, 0, :]
    y = positions[frame_idx, 1, :]

    dx = np.gradient(x)
    dy = np.gradient(y)
    mag = np.hypot(dx, dy)
    mag[mag == 0] = 1e-6
    nx = -dy / mag
    ny = dx / mag

    top_x = x + outer_radius * nx
    top_y = y + outer_radius * ny
    bot_x = x - outer_radius * nx
    bot_y = y - outer_radius * ny

    poly_verts = np.column_stack((
        np.concatenate([top_x, bot_x[::-1]]),
        np.concatenate([top_y, bot_y[::-1]])
    ))

    arrow_shaft_poly.set_xy(poly_verts)
    arrow_nodes.set_data(x, y)

    tail_x = positions[:frame_idx + 1, 0, 0]
    tail_y = positions[:frame_idx + 1, 1, 0]
    tail_trace.set_data(tail_x, tail_y)
    com_marker.set_data([com_x[frame_idx]], [com_y[frame_idx]])
    com_trace.set_data(com_x[:frame_idx + 1], com_y[:frame_idx + 1])

    # Update Tail Force Vector Arrow
    f_x = tail_forces[frame_idx, 0] * force_scale
    f_y = tail_forces[frame_idx, 1] * force_scale
    tail_force_arrow.set_offsets(np.c_[x[0], y[0]])
    tail_force_arrow.set_UVC(f_x, f_y)

    # Update CoM Velocity Direction Arrow
    vx = com_vx[frame_idx]
    vy = com_vy[frame_idx]
    v_mag = np.hypot(vx, vy)
    if v_mag > 1e-4:
        vel_dir_x = (vx / v_mag) * fixed_arrow_length
        vel_dir_y = (vy / v_mag) * fixed_arrow_length
    else:
        vel_dir_x, vel_dir_y = 0.0, 0.0

    com_vel_arrow.set_offsets(np.c_[com_x[frame_idx], com_y[frame_idx]])
    com_vel_arrow.set_UVC(vel_dir_x, vel_dir_y)

    # Update Initial Direction Arrow
    init_x = init_dir[0] * fixed_arrow_length
    init_y = init_dir[1] * fixed_arrow_length
    init_dir_arrow.set_offsets(np.c_[com_x[frame_idx], com_y[frame_idx]])
    init_dir_arrow.set_UVC(init_x, init_y)

    time_text.set_text(f"t = {times[frame_idx] * 1000:.2f} ms")

    # Local Plot Update (CoM origin, tip-to-tail orientation)
    p_tail = positions[frame_idx, :2, 0]
    p_tip = positions[frame_idx, :2, -1]

    # Orientation ONLY: unit vector pointing from the tip toward the tail
    vec = p_tail - p_tip
    L_chord = np.linalg.norm(vec)

    if L_chord > 1e-6:
        u_x = vec / L_chord
        u_y = np.array([-u_x[1], u_x[0]])
    else:
        u_x = np.array([1.0, 0.0])
        u_y = np.array([0.0, 1.0])

    # Origin: mass-weighted center of mass of the current frame
    p_com = np.array([com_x[frame_idx], com_y[frame_idx]])

    rel_pos = positions[frame_idx, :2, :] - p_com[:, np.newaxis]
    local_x = u_x @ rel_pos
    local_y = u_y @ rel_pos

    ldx = np.gradient(local_x)
    ldy = np.gradient(local_y)
    lmag = np.hypot(ldx, ldy)
    lmag[lmag == 0] = 1e-6
    lnx = -ldy / lmag
    lny = ldx / lmag

    ltop_x = local_x + outer_radius * lnx
    ltop_y = local_y + outer_radius * lny
    lbot_x = local_x - outer_radius * lnx
    lbot_y = local_y - outer_radius * lny

    lpoly_verts = np.column_stack((
        np.concatenate([ltop_x, lbot_x[::-1]]),
        np.concatenate([ltop_y, lbot_y[::-1]])
    ))

    arrow_local_poly.set_xy(lpoly_verts)
    arrow_local_nodes.set_data(local_x, local_y)

    # Bow rest center (global origin) expressed in the local CoM frame
    bow_rel = np.array([0.0, 0.0]) - p_com
    bow_lx = float(u_x @ bow_rel)
    bow_ly = float(u_y @ bow_rel)
    bow_local_circle.center = (bow_lx, bow_ly)

    # Only show it once it actually reaches the plotted region
    x0, x1 = ax_local.get_xlim()
    y0, y1 = ax_local.get_ylim()
    inside_view = (
        bow_lx + bow_rest_radius >= x0 and bow_lx - bow_rest_radius <= x1
        and bow_ly + bow_rest_radius >= y0 and bow_ly - bow_rest_radius <= y1
    )
    bow_local_circle.set_visible(inside_view)

    fig.canvas.draw_idle()


def on_timeline_change(val):
    global current_frame
    target_time = val / 1000.0
    current_frame = np.searchsorted(times, target_time)
    current_frame = min(current_frame, n_frames - 1)
    update_frame(current_frame)


slider_timeline.on_changed(on_timeline_change)


def toggle_play(event):
    global is_playing
    is_playing = not is_playing
    btn_play.label.set_text('Pause' if is_playing else 'Play')


btn_play.on_clicked(toggle_play)


def animate(frame):
    global current_frame, is_playing
    if is_playing:
        step_inc = max(1, int(slider_speed.val))
        current_frame = (current_frame + step_inc) % n_frames

        slider_timeline.eventson = False
        slider_timeline.set_val(times[current_frame] * 1000)
        slider_timeline.eventson = True

        update_frame(current_frame)


ani = FuncAnimation(fig, animate, interval=30, cache_frame_data=False)
plt.show()