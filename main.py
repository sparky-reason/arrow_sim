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
# 2. Vector String Push Force (Targeting Brace Point)
# ==========================================
class StringPushForce(ea.NoForces):
    def __init__(self, f_max, draw_length, brace_height, c_string, pluck_velocity):
        super().__init__()
        self.f_max = f_max  # N
        self.draw_length = draw_length  # m
        self.brace_height = brace_height  # m
        self.c_string = c_string  # N*s/m (transverse string damping)
        self.pluck_v = pluck_velocity  # m/s (finger release roll speed)
        self.initialized = False

    def apply_forces(self, system, time=0.0):
        # Apply initial finger-pluck lateral impulse on first time step
        if not self.initialized:
            system.velocity_collection[1, 0] += self.pluck_v  # m/s
            self.initialized = True

        pos_x = system.position_collection[0, 0]  # m
        pos_y = system.position_collection[1, 0]  # m
        v_y = system.velocity_collection[1, 0]    # m/s

        if pos_x < -self.brace_height:
            # Main thrust force vector targeting (-brace_height, 0)
            stroke_factor = (-self.brace_height - pos_x) / (self.draw_length - self.brace_height)
            stroke_factor = max(0.0, min(1.0, stroke_factor))
            f_mag = self.f_max * stroke_factor  # N

            dx = -self.brace_height - pos_x  # m
            dy = 0.0 - pos_y  # m
            mag_r = np.hypot(dx, dy)  # m

            if mag_r > 1e-6:
                system.external_forces[0, 0] += f_mag * (dx / mag_r)  # N
                system.external_forces[1, 0] += f_mag * (dy / mag_r)  # N

            # Stable Transverse String Damping (Wave radiation/internal friction)
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

# String Mass Proxy (Added directly to nock node mass matrix)
string_effective_mass_kg = 0.0035  # kg (3.5 grams of moving string mass)

n_elements = 30  # dimensionless (number of discretization elements)
outer_radius = 0.00375  # m (3.75 mm radius / 7.5 mm shaft diameter)
density = 500.0  # kg/m^3 (carbon shaft density)

# Configurable Spine to E_modulus conversion (ATA Standard: 1.94 lbs at 28 inches)
spine_value = 400.0  # dimensionless (deflection in thousandths of an inch)
deflection_m = (spine_value / 1000.0) * 0.0254  # m
span_L = 28.0 * 0.0254  # m (28 inches span)
test_force_N = 1.94 * 4.44822  # N (1.94 lbs test weight)
I_beam = (np.pi / 4.0) * (outer_radius ** 4)  # m^4 (Area moment of inertia)
E_modulus = (test_force_N * (span_L ** 3)) / (48.0 * deflection_m * I_beam)  # Pa

bow_rest_radius = 0.015  # m (1.5 cm radius)
bow_rest_height = 0.1  # m (cylinder length)

# Tangent angle for bow contact
R_eff = bow_rest_radius + outer_radius  # m
theta = np.arcsin(R_eff / draw_length)  # rad

direction = np.array([np.cos(theta), np.sin(theta), 0.0])  # dimensionless (unit vector)
normal = np.array([-np.sin(theta), np.cos(theta), 0.0])  # dimensionless (unit vector)
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

# Apply point mass to arrow tip (last node)
arrow.mass[-1] += tip_mass_kg  # kg

# Add effective string mass directly to arrow nock (first node)
arrow.mass[0] += string_effective_mass_kg  # kg

# Calculate total mass of the arrow system
total_mass_kg = np.sum(arrow.mass)
total_mass_grams = total_mass_kg * 1000.0
print(f"--- Arrow Mass Breakdown ---")
print(f"Total Mass: {total_mass_grams:.2f} grams")


arrow_sim.append(arrow)

# Internal structural damping
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
    direction=np.array([0.0, 0.0, 1.0]),  # dimensionless (unit vector)
    normal=np.array([1.0, 0.0, 0.0]),  # dimensionless (unit vector)
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
    k=1e5,  # N/m (contact stiffness)
    nu=1.0,  # kg/s (contact damping)
)

# ==========================================
# 6. Apply Dynamic String Force & Aerodynamics
# ==========================================
draw_weight_lbs = 60.0  # lbs
LBS_TO_N = 4.44822  # N/lb
push_force_magnitude = draw_weight_lbs * LBS_TO_N  # N

arrow_sim.add_forcing_to(arrow).using(
    StringPushForce,
    f_max=push_force_magnitude,  # N
    draw_length=draw_length,      # m
    brace_height=brace_height,    # m
    c_string=0.4,                 # N*s/m (tuned recurve string damping)
    pluck_velocity=1.0            # m/s (finger release roll speed)
)

arrow_sim.add_forcing_to(arrow).using(
    ArrowAerodynamics,
    rho_air=1.225,  # kg/m^3
    Cd_shaft=1.0,  # dimensionless
    Cd_fletching=1.2,  # dimensionless
    fletching_area=0.002,  # m^2
)

# ==========================================
# 7. Diagnostics Callback
# ==========================================
class ArrowCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        super().__init__()
        self.step_skip = step_skip  # dimensionless (step counter)
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step):
        if current_step % self.step_skip == 0:
            self.callback_params["time"].append(time)  # s
            self.callback_params["position"].append(system.position_collection.copy())  # m


data_tracker = {"time": [], "position": []}
arrow_sim.collect_diagnostics(arrow).using(
    ArrowCallBack, step_skip=250, callback_params=data_tracker  # dimensionless
)

# Finalize and integrate
arrow_sim.finalize()
timestepper = PositionVerlet()

n_steps = 40_000  # dimensionless (number of integration steps)
final_time = 0.02  # s (total simulation time)

print(f"Simulating shot ({draw_weight_lbs} lbs draw, {spine_value} spine)...")
integrate(timestepper, arrow_sim, final_time, n_steps)
print("Simulation complete!")

# ==========================================
# 8. Compute Mass-Weighted Center of Mass (CoM)
# ==========================================
positions = np.array(data_tracker["position"])  # Shape: (n_frames, 3, n_nodes), values in m
times = np.array(data_tracker["time"])  # values in s
n_frames = len(times)  # dimensionless

node_masses = arrow.mass.copy()  # kg
total_mass = np.sum(node_masses)  # kg

com_x = np.sum(positions[:, 0, :] * node_masses, axis=1) / total_mass  # m
com_y = np.sum(positions[:, 1, :] * node_masses, axis=1) / total_mass  # m

# ==========================================
# 9. Interactive Visualization & GUI
# ==========================================
fig, (ax_main, ax_local) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={'height_ratios': [2, 1]})
plt.subplots_adjust(bottom=0.20, hspace=0.35)

# --- MAIN PLOT (GLOBAL FRAME) ---
bow_circle = Circle((0, 0), bow_rest_radius, color="black", alpha=0.7, label="Bow Rest (0,0)")
ax_main.add_patch(bow_circle)
ax_main.axvline(x=-brace_height, color="red", linestyle="--", alpha=0.7, label=f"Brace Height (-{brace_height}m)")

arrow_shaft_poly = Polygon(np.empty((0, 2)), color='tab:blue', alpha=0.5, label="Arrow Shaft (True Width)")
ax_main.add_patch(arrow_shaft_poly)
(arrow_nodes,) = ax_main.plot([], [], 'o', color='navy', markersize=3, label="Arrow Nodes")

(tail_trace,) = ax_main.plot([], [], ':', color='tab:gray', alpha=0.5, label="Tail Path")

(com_marker,) = ax_main.plot([], [], 'r.', markersize=10, label="Center of Mass")
(com_trace,) = ax_main.plot([], [], '--', color='tab:red', alpha=0.7, lw=1.5, label="CoM Trajectory")

time_text = ax_main.text(
    0.02, 0.92, '', transform=ax_main.transAxes, fontsize=11, fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1)
)

all_x = positions[:, 0, :]
all_y = positions[:, 1, :]
ax_main.set_xlim(np.min(all_x) - 0.05, np.max(all_x) + 0.05)  # m
ax_main.set_ylim(np.min(all_y) - 0.05, np.max(all_y) + 0.05)  # m
ax_main.set_aspect('equal')
ax_main.grid(True)
ax_main.set_xlabel("X Position [m]")
ax_main.set_ylabel("Y Deflection [m]")
ax_main.set_title(f"Archer's Paradox ({draw_weight_lbs} lbs draw, {spine_value} spine, {tip_weight_grains:.0f} gr tip)")
ax_main.legend(loc="upper right")

# --- LOCAL PLOT (VIBRATION FRAME) ---
arrow_local_poly = Polygon(np.empty((0, 2)), color='tab:orange', alpha=0.5, label="Local Shaft (True Width)")
ax_local.add_patch(arrow_local_poly)
(arrow_local_nodes,) = ax_local.plot([], [], 'o', color='#b34700', markersize=3, label="Local Nodes")

ax_local.axhline(0, color='black', lw=1, alpha=0.5)

ax_local.set_xlim(-0.02, length + 0.02)  # m
ax_local.set_ylim(-0.06, 0.06)  # m
ax_local.grid(True)
ax_local.set_xlabel("Chord Position [m]")
ax_local.set_ylabel("Transverse Deflection [m]")
ax_local.set_title("Arrow Vibrations (Local Tail-to-Tip Reference Frame)")
ax_local.legend(loc="upper right")

# Widget Axes Placement
ax_timeline = plt.axes([0.15, 0.10, 0.7, 0.02])
ax_play = plt.axes([0.15, 0.03, 0.1, 0.04])
ax_speed = plt.axes([0.40, 0.03, 0.45, 0.03])

# Widgets Initialization
slider_timeline = Slider(ax_timeline, 'Time', 0, times[-1] * 1000, valinit=0, valfmt='%.1f ms')  # ms
btn_play = Button(ax_play, 'Pause', hovercolor='0.97')
slider_speed = Slider(ax_speed, 'Speed', 0.2, 5.0, valinit=1.0, valfmt='%.1fx')  # multiplier

# State Variables
is_playing = True
current_frame = 0


def update_frame(frame_idx):
    # --- Main Plot Update ---
    x = positions[frame_idx, 0, :]  # m
    y = positions[frame_idx, 1, :]  # m

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

    tail_x = positions[:frame_idx + 1, 0, 0]  # m
    tail_y = positions[:frame_idx + 1, 1, 0]  # m
    tail_trace.set_data(tail_x, tail_y)
    com_marker.set_data([com_x[frame_idx]], [com_y[frame_idx]])  # m
    com_trace.set_data(com_x[:frame_idx + 1], com_y[:frame_idx + 1])  # m

    time_text.set_text(f"t = {times[frame_idx] * 1000:.2f} ms")  # ms

    # --- Local Plot Update ---
    p_tail = positions[frame_idx, :2, 0]
    p_tip = positions[frame_idx, :2, -1]

    vec = p_tip - p_tail
    L_chord = np.linalg.norm(vec)

    if L_chord > 1e-6:
        u_x = vec / L_chord
        u_y = np.array([-u_x[1], u_x[0]])
    else:
        u_x = np.array([1.0, 0.0])
        u_y = np.array([0.0, 1.0])

    rel_pos = positions[frame_idx, :2, :] - p_tail[:, np.newaxis]
    local_x = u_x @ rel_pos  # m
    local_y = u_y @ rel_pos  # m

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

    fig.canvas.draw_idle()


def on_timeline_change(val):
    global current_frame
    target_time = val / 1000.0  # s
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
        slider_timeline.set_val(times[current_frame] * 1000)  # ms
        slider_timeline.eventson = True

        update_frame(current_frame)


ani = FuncAnimation(fig, animate, interval=30, cache_frame_data=False)  # interval in ms
plt.show()