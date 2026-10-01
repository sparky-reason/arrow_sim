"""Physical simulation of an arrow being shot from a bow ("The Archer's Paradox").

This module owns everything that is physics: the rod/contact/forcing setup, the
time integration, and the post-processing of the logged diagnostics into a
:class:`SimulationResult`.

Every tunable lives in :class:`SimConfig`; everything else is either a
standardized conversion/spine-calibration constant or a numerical guard, and
stays a module-level constant. Nothing here is a mutable global: all state
lives on :class:`Simulation` instances or on the returned result.
"""

from dataclasses import dataclass

import numpy as np

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
# Standardized constants (not user-tunable)
# ==========================================
GRAINS_TO_KG = 0.00006479891  # kg/grain
LBS_TO_N = 4.44822  # N/lb
INCH_TO_M = 0.0254  # m/inch

# Arrow-spine calibration. The spine value is the user-facing knob; the rest of
# the test rig (three-point bending of the shaft) is standardized and therefore
# fixed. See youngs_modulus() below for the conversion.
SPINE_DEFLECTION_PER_1000 = 0.0254  # m of deflection per 1000 spine units
SPINE_TEST_SPAN = 28.0  # in, support span of the bending test
SPINE_TEST_FORCE = 1.94  # lbf, applied at mid-span
SPINE_BEAM_DIVISOR = 48.0  # three-point bending deflection coefficient
SHEAR_MODULUS_DIVISOR = 2.6  # G = E / 2.6

# Bow rest geometry/material. Fixed props of the rig, not part of the arrow.
BOW_REST_HEIGHT = 0.1  # m
BOW_REST_DENSITY = 1000.0  # kg/m^3

# Numerical guards, kept out of SimConfig because they are hygiene, not physics.
VEL_EPS = 1e-4  # m/s, below this a velocity is treated as zero
NORM_EPS = 1e-6  # generic zero-guard for norms and lengths


# ==========================================
# Configuration
# ==========================================
@dataclass
class SimConfig:
    """All free constants of the simulation.

    Defaults reproduce the original hard-coded setup, so ``SimConfig()`` runs the
    exact same shot as the pre-refactor script.
    """

    # --- Arrow geometry & material ---
    draw_length: float = 0.70  # m (70 cm draw length)
    brace_height: float = 0.18  # m (18 cm brace height)
    arrow_length: float = 0.75  # m (75 cm arrow length)
    n_elements: int = 30  # dimensionless
    outer_radius: float = 0.00375  # m (3.75 mm radius)
    shaft_density: float = 500.0  # kg/m^3

    # --- Masses ---
    tip_weight_grains: float = 100.0  # grains
    string_effective_mass: float = 0.0035  # kg (moving string mass)

    # --- Stiffness ---
    spine_value: float = 400.0  # dimensionless

    # --- Bow rest & contact ---
    bow_rest_radius: float = 0.015  # m
    contact_stiffness: float = 1e5  # N/m
    contact_damping: float = 1.0  # kg/s

    # --- String push (kinematic finger release) ---
    draw_weight_lbs: float = 25.0  # lbs
    string_damping: float = 0.0  # N*s/m (transverse string damping)
    initial_pluck_angle_deg: float = 8.0  # initial lateral angle
    pluck_decay_length: float = 0.025  # m

    # --- Aerodynamics ---
    rho_air: float = 1.225  # kg/m^3
    cd_shaft: float = 1.0  # dimensionless
    cd_fletching: float = 1.2  # dimensionless
    fletching_area: float = 0.002  # m^2

    # --- Internal damping ---
    damping_constant: float = 0.01  # kg/s
    damping_time_step: float = 1e-6  # s

    # --- Integration & logging ---
    n_steps: int = 60_000
    final_time: float = 0.03  # s
    diagnostic_step_skip: int = 250


def youngs_modulus(spine_value: float, outer_radius: float) -> float:
    """Convert a spine value into a Young's modulus [Pa].

    The spine value is defined by a standardized three-point bending test on the
    shaft: a known load is applied at mid-span of a support span, and the
    resulting mid-span deflection sets the stiffness. Inverting the beam
    deflection formula for a simply supported beam gives E directly.
    """
    deflection_m = (spine_value / 1000.0) * SPINE_DEFLECTION_PER_1000  # m
    span_L = SPINE_TEST_SPAN * INCH_TO_M  # m
    test_force_N = SPINE_TEST_FORCE * LBS_TO_N  # N
    i_beam = (np.pi / 4.0) * (outer_radius**4)  # m^4
    return (test_force_N * (span_L**3)) / (SPINE_BEAM_DIVISOR * deflection_m * i_beam)  # Pa


class ArrowAerodynamics(ea.NoForces):
    """Aerodynamic drag on the shaft and the fletching.

    Drag is applied perpendicular to the shaft axis on every element, plus a
    lumped fletching drag on the nock.
    """

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

            if v_mag > VEL_EPS:  # m/s
                t = tangents[:, i]
                v_normal = v_node - np.dot(v_node, t) * t
                v_normal_mag = np.linalg.norm(v_normal)
                area = 2.0 * radii[i] * lengths[i]  # m^2
                drag_force = -0.5 * self.rho * self.Cd_shaft * area * v_normal_mag * v_normal  # N

                system.external_forces[:, i] += 0.5 * drag_force
                system.external_forces[:, i + 1] += 0.5 * drag_force

        v_tail = velocities[:, 0]
        v_tail_mag = np.linalg.norm(v_tail)
        if v_tail_mag > VEL_EPS:  # m/s
            fletching_drag = -0.5 * self.rho * self.Cd_fletching * self.A_fletching * v_tail_mag * v_tail  # N
            system.external_forces[:, 0] += fletching_drag


class StringPushForce(ea.NoForces):
    """Angled push on the nock, modelling the string during the stroke.

    The force aims at a lateral target that decays from the plucker angle to
    zero over the first ``pluck_decay_length`` of the stroke, and a transverse
    damping term keeps the string release stable.
    """

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
        v_y = system.velocity_collection[1, 0]  # m/s

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

            if mag_r > NORM_EPS:
                system.external_forces[0, 0] += f_mag * (dx / mag_r)  # N
                system.external_forces[1, 0] += f_mag * (dy / mag_r)  # N

            # Stable Transverse String Damping
            system.external_forces[1, 0] += -self.c_string * v_y  # N


class ArrowSimulation(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
    ea.Contact,
):
    """PyElastica system collection holding the arrow, the bow rest and their forces."""


class ArrowCallBack(CallBackBaseClass):
    """Records time, node positions and the tail force vector every ``step_skip`` steps."""

    def __init__(self, step_skip: int, callback_params: dict):
        super().__init__()
        self.step_skip = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step):
        if current_step % self.step_skip == 0:
            self.callback_params["time"].append(time)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["tail_force"].append(system.external_forces[:, 0].copy())


@dataclass
class SimulationResult:
    """Everything the viewer needs to render a simulated shot.

    Attributes are post-processed arrays of shape ``(n_frames, ...)`` plus the
    few scalars the viewer needs for axis limits and titles. ``config`` is kept
    so the viewer is self-sufficient: it receives the result and nothing else.
    """

    times: np.ndarray  # (n_frames,) [s]
    positions: np.ndarray  # (n_frames, 3, n_nodes) [m]
    tail_forces: np.ndarray  # (n_frames, 3) [N]
    node_masses: np.ndarray  # (n_nodes,) [kg]
    total_mass: float  # kg
    com_x: np.ndarray  # (n_frames,) [m]
    com_y: np.ndarray  # (n_frames,) [m]
    com_vx: np.ndarray  # (n_frames,) [m/s]
    com_vy: np.ndarray  # (n_frames,) [m/s]
    com_arc_length: float  # m, arc length of the CoM measured from the tail
    init_dir: np.ndarray  # (2,) unit initial direction in the xy-plane
    config: SimConfig  # the configuration this shot was produced with


class Simulation:
    """Builds and integrates the arrow/bow system described by a :class:`SimConfig`.

    Usage::

        result = Simulation(SimConfig()).run()
    """

    def __init__(self, config: SimConfig):
        self.config = config

    def run(self) -> SimulationResult:
        cfg = self.config
        arrow_sim = ArrowSimulation()

        arrow, init_dir = self._build_arrow()
        self._apply_masses(arrow, arrow_sim)
        bow_rest = self._build_bow_rest()
        arrow_sim.append(bow_rest)
        arrow_sim.constrain(bow_rest).using(
            FixedConstraint,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
        )
        arrow_sim.detect_contact_between(arrow, bow_rest).using(
            RodCylinderContact,
            k=cfg.contact_stiffness,  # N/m
            nu=cfg.contact_damping,  # kg/s
        )
        self._apply_forcing(arrow, arrow_sim)

        data_tracker = {"time": [], "position": [], "tail_force": []}
        arrow_sim.collect_diagnostics(arrow).using(
            ArrowCallBack, step_skip=cfg.diagnostic_step_skip, callback_params=data_tracker
        )

        # Finalize and integrate
        arrow_sim.finalize()
        timestepper = PositionVerlet()

        print(f"Simulating shot ({cfg.draw_weight_lbs} lbs draw, {cfg.spine_value} spine)...")
        integrate(timestepper, arrow_sim, cfg.final_time, cfg.n_steps)
        print("Simulation complete!")

        return self._postprocess(data_tracker, arrow, init_dir)

    # --- Physical & Geometric Setup ---
    def _build_arrow(self) -> tuple[CosseratRod, np.ndarray]:
        """Create the straight Cosserat rod and return it with its xy-direction."""
        cfg = self.config

        # The string is deflected laterally by the plucker, so the nock sits on a
        # circle of radius (bow_rest_radius + outer_radius) around the rest: this
        # gives the release angle theta at which the string leaves the fingers.
        r_eff = cfg.bow_rest_radius + cfg.outer_radius  # m
        theta = np.arcsin(r_eff / cfg.draw_length)  # rad

        direction = np.array([np.cos(theta), np.sin(theta), 0.0])
        normal = np.array([-np.sin(theta), np.cos(theta), 0.0])
        start_position = np.array([-cfg.draw_length, 0.0, 0.0])  # m

        E_modulus = youngs_modulus(cfg.spine_value, cfg.outer_radius)  # Pa

        arrow = CosseratRod.straight_rod(
            n_elements=cfg.n_elements,
            start=start_position,
            direction=direction,
            normal=normal,
            base_length=cfg.arrow_length,
            base_radius=cfg.outer_radius,
            density=cfg.shaft_density,
            youngs_modulus=E_modulus,
            shear_modulus=E_modulus / SHEAR_MODULUS_DIVISOR,  # Pa
        )

        shaft_mass_kg = np.sum(arrow.mass)
        tip_mass_kg = self._tip_mass_kg()
        print(f"--- Arrow Mass Breakdown ---")
        print(f"Shaft Mass: {shaft_mass_kg * 1000.0:.2f} grams")
        print(f"Tip Mass:   {tip_mass_kg * 1000.0:.2f} grams")
        print(f"Arrow Mass: {(shaft_mass_kg + tip_mass_kg) * 1000.0:.2f} grams")

        init_dir = direction[:2] / np.linalg.norm(direction[:2])
        return arrow, init_dir

    def _tip_mass_kg(self) -> float:
        return self.config.tip_weight_grains * GRAINS_TO_KG  # kg

    def _apply_masses(self, arrow: CosseratRod, arrow_sim: ArrowSimulation) -> None:
        """Add the tip mass at the tip and the string mass proxy at the nock."""
        arrow.mass[-1] += self._tip_mass_kg()  # kg
        arrow.mass[0] += self.config.string_effective_mass  # kg

        arrow_sim.append(arrow)
        arrow_sim.dampen(arrow).using(
            AnalyticalLinearDamper,
            damping_constant=self.config.damping_constant,  # kg/s
            time_step=self.config.damping_time_step,  # s
        )

    # --- Bow Rest & Contact Setup ---
    def _build_bow_rest(self) -> Cylinder:
        cfg = self.config
        return Cylinder(
            start=np.array([0.0, 0.0, -BOW_REST_HEIGHT / 2.0]),  # m
            direction=np.array([0.0, 0.0, 1.0]),
            normal=np.array([1.0, 0.0, 0.0]),
            base_length=BOW_REST_HEIGHT,  # m
            base_radius=cfg.bow_rest_radius,  # m
            density=BOW_REST_DENSITY,  # kg/m^3
        )

    # --- Dynamic String Force & Aerodynamics ---
    def _apply_forcing(self, arrow: CosseratRod, arrow_sim: ArrowSimulation) -> None:
        cfg = self.config

        arrow_sim.add_forcing_to(arrow).using(
            StringPushForce,
            f_max=cfg.draw_weight_lbs * LBS_TO_N,  # N
            draw_length=cfg.draw_length,
            brace_height=cfg.brace_height,
            c_string=cfg.string_damping,
            pluck_angle=np.radians(cfg.initial_pluck_angle_deg),
            pluck_decay_length=cfg.pluck_decay_length,
        )

        arrow_sim.add_forcing_to(arrow).using(
            ArrowAerodynamics,
            rho_air=cfg.rho_air,
            Cd_shaft=cfg.cd_shaft,
            Cd_fletching=cfg.cd_fletching,
            fletching_area=cfg.fletching_area,
        )

    # --- Mass-Weighted Center of Mass (CoM) & Velocity ---
    def _postprocess(
        self,
        data_tracker: dict,
        arrow: CosseratRod,
        init_dir: np.ndarray,
    ) -> SimulationResult:
        cfg = self.config
        positions = np.array(data_tracker["position"])
        tail_forces = np.array(data_tracker["tail_force"])  # Shape: (n_frames, 3) in N
        times = np.array(data_tracker["time"])

        node_masses = arrow.mass.copy()
        total_mass = np.sum(node_masses)

        com_x = np.sum(positions[:, 0, :] * node_masses, axis=1) / total_mass
        com_y = np.sum(positions[:, 1, :] * node_masses, axis=1) / total_mass

        # Center of Mass Velocity (via finite differences)
        dt = times[1] - times[0] if len(times) > 1 else 1.0
        com_vx = np.gradient(com_x, dt)
        com_vy = np.gradient(com_y, dt)

        # Arc-length position of the CoM measured from the tail (m).
        # Used only to set sensible limits on the local (CoM) plot's x-axis.
        node_arc_length = (np.arange(len(node_masses)) / (len(node_masses) - 1)) * cfg.arrow_length
        com_arc_length = float(np.sum(node_arc_length * node_masses) / total_mass)

        return SimulationResult(
            times=times,
            positions=positions,
            tail_forces=tail_forces,
            node_masses=node_masses,
            total_mass=float(total_mass),
            com_x=com_x,
            com_y=com_y,
            com_vx=com_vx,
            com_vy=com_vy,
            com_arc_length=com_arc_length,
            init_dir=init_dir,
            config=cfg,
        )
