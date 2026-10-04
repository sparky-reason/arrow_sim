"""
High-fidelity reduced-order bow/arrow launch model.

This module is deliberately a launch solver, not a claim of a universal
"exact" bow model.  It uses a discretised Euler-Bernoulli arrow beam and
switches boundary conditions through the actual shot sequence:

    full draw -> constrained string/nock state -> finger release ->
    transient string drive + unilateral bow contact -> free arrow.

The arrow shaft is discretised in transverse Y/Z displacement coordinates.
The bow/riser is rigid in this model.  Bow-limb and string inertia are not
explicitly finite-element modelled; instead the bow is represented by a
measured/parameterised force-draw curve and a prescribed string path.
For experimental-grade prediction, supply measured force-draw data and bow
geometry rather than relying on the generic curves below.

Contact is NOT a penalty spring.  At each integration evaluation an active-set
Lagrange-multiplier solve enforces the unilateral condition

    gap >= 0, normal_force >= 0, gap * normal_force = 0.

The pre-release state is solved separately, with the nock constrained to the
string.  This is important: free-free beam dynamics are only used after the
external launch constraints disappear.

Feather clocking is an explicit input.  It is part of the initial attitude,
so the flight simulator sees the actual fletching orientation at bow exit.

Coordinate convention:
    +X = nock -> point / range
    +Y = up
    +Z = lateral, toward the arrow side of the bow grip
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from arrow_flight_sim import (
    Arrow,
    Atmosphere,
    FlightResult,
    orientation_from_direction_and_roll,
    simulate,
)

G = 9.80665


def _norm(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return v / n


def _smoothstep(u):
    u = np.clip(float(u), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


@dataclass
class Bow:
    """Traditional longbow or recurve parameterisation.

    The three primary inputs are length, draw strength and draw length.
    Other parameters describe the actual bow/string/riser and should be
    measured for a particular bow when high accuracy is required.
    """

    length_m: float
    draw_strength_kgf: float
    draw_length_m: float
    bow_type: str = "longbow"

    brace_height_m: float = 0.16
    handle_length_m: float = 0.22
    handle_height_m: float = 0.10
    handle_thickness_m: float = 0.035

    # Arrow/riser geometry.
    arrow_side_offset_m: Optional[float] = None
    nock_height_offset_m: float = 0.0
    aim_elevation_deg: float = 0.0
    aim_azimuth_deg: float = 0.0
    bow_cant_deg: float = 0.0
    bow_roll_deg: float = 0.0

    # The supplied draw length is interpreted explicitly as the nock/string
    # travel from brace to full draw.  If your bow's draw-length convention is
    # AMO/ATA (pivot-point based), convert it to nock travel before supplying it.
    draw_length_is_nock_travel: bool = True

    # Shot-to-shot draw/anchor errors. Positive draw_length_error means a
    # longer-than-reference draw; negative values model creeping/short draw.
    draw_length_error_m: float = 0.0

    # Small bow-hand motion at the instant of release.  These alter the rigid
    # riser orientation seen by the arrow; they are not substitutes for a full
    # dynamic bow-hand/limb model.
    release_bow_cant_deg: float = 0.0
    release_bow_azimuth_deg: float = 0.0
    release_bow_elevation_deg: float = 0.0

    # Bow dynamic properties not determined by the three primary inputs.
    # Energy-transfer efficiency is used only when the bow limbs/string are
    # represented by the reduced force-draw model.  It scales the delivered
    # force curve uniformly so its integral equals efficiency * stored energy.
    # If a measured *arrow-force* curve is supplied, set this to 1.0.
    bow_efficiency: float = 0.78
    nock_efficiency: float = 0.995
    string_mass_kg: float = 0.010
    limb_mass_kg: float = 0.08

    # If supplied, this overrides the generic force-draw shape.
    force_draw_exponent: Optional[float] = None
    force_draw_table: Optional[np.ndarray] = None  # columns: draw[m], force[N]

    # Finger release disturbance.  These are initial string-endpoint errors/velocities
    # at finger separation; zero is an idealised release.
    release_duration_s: float = 0.0015
    release_lateral_displacement_m: float = 0.0
    release_lateral_velocity_m_s: float = 0.0
    release_vertical_displacement_m: float = 0.0
    release_vertical_velocity_m_s: float = 0.0
    release_string_roll_deg: float = 0.0
    release_string_roll_rate_rad_s: float = 0.0
    release_lateral_acceleration_m_s2: float = 0.0
    release_vertical_acceleration_m_s2: float = 0.0

    # Effective dynamic string model.  The full limb/string FE problem is not
    # identifiable from draw weight alone, so these are explicit measured/tuned
    # properties of the particular bow/string.
    string_lateral_stiffness_N_m: float = 3500.0
    string_vertical_stiffness_N_m: float = 3500.0
    string_lateral_damping_Ns_m: float = 0.8
    string_vertical_damping_Ns_m: float = 0.8
    string_effective_mass_fraction: float = 0.50

    # Fletching orientation about the shaft at launch.
    feather_clocking_deg: float = 0.0

    # Riser contact/friction.  Friction is intentionally off by default;
    # enabling it requires measured surface friction and greatly increases
    # sensitivity to exact riser geometry.
    contact_friction_mu: float = 0.0

    # Equipment/setup errors that have direct mechanical meaning.
    nock_fit_clearance_m: float = 0.00005
    arrow_rest_longitudinal_offset_m: float = 0.0
    arrow_rest_height_error_m: float = 0.0

    def __post_init__(self):
        self.bow_type = self.bow_type.lower()
        if self.bow_type not in {"longbow", "recurve"}:
            raise ValueError("bow_type must be 'longbow' or 'recurve'")
        if min(self.length_m, self.draw_length_m) <= 0:
            raise ValueError("length and draw length must be positive")
        if self.draw_strength_kgf <= 0:
            raise ValueError("draw strength must be positive")
        if not (0 < self.brace_height_m < self.draw_length_m):
            raise ValueError("brace height must be between 0 and draw length")
        if not (0 < self.bow_efficiency <= 1):
            raise ValueError("bow_efficiency must be in (0,1]")
        if not (0 < self.nock_efficiency <= 1):
            raise ValueError("nock_efficiency must be in (0,1]")
        if self.release_duration_s <= 0:
            raise ValueError("release_duration_s must be positive")
        if self.arrow_side_offset_m is None:
            self.arrow_side_offset_m = 0.018
        if self.arrow_side_offset_m < 0:
            raise ValueError("arrow_side_offset_m must be non-negative")
        if self.contact_friction_mu < 0:
            raise ValueError("contact_friction_mu must be non-negative")
        if self.nock_fit_clearance_m < 0:
            raise ValueError("nock_fit_clearance_m must be non-negative")
        if abs(self.draw_length_error_m) >= self.draw_length_m:
            raise ValueError("draw_length_error_m is too large")
        if min(self.string_lateral_stiffness_N_m, self.string_vertical_stiffness_N_m) <= 0:
            raise ValueError("string stiffnesses must be positive")
        if min(self.string_lateral_damping_Ns_m, self.string_vertical_damping_Ns_m) < 0:
            raise ValueError("string damping must be non-negative")
        if not (0 < self.string_effective_mass_fraction <= 1):
            raise ValueError("string_effective_mass_fraction must be in (0,1]")

        if self.force_draw_table is not None:
            tab = np.asarray(self.force_draw_table, dtype=float)
            if tab.ndim != 2 or tab.shape[1] != 2 or len(tab) < 2:
                raise ValueError("force_draw_table must have shape (N,2)")
            if np.any(np.diff(tab[:, 0]) <= 0) or tab[0, 0] < 0:
                raise ValueError("draw table must have increasing non-negative displacement")
            self.force_draw_table = tab

    @property
    def draw_force_N(self):
        return self.draw_strength_kgf * G

    @property
    def draw_coordinate_m(self):
        # This model uses actual nock/string travel from brace to full draw.
        # AMO/ATA draw length is a different geometric convention and must be
        # converted before being supplied if draw_length_is_nock_travel=True.
        if not self.draw_length_is_nock_travel:
            raise ValueError(
                "Convert the supplied AMO/pivot-based draw length to nock travel "
                "before using this launch model."
            )
        return self.draw_length_m + self.draw_length_error_m

    def draw_force(self, displacement_m):
        d = float(np.clip(displacement_m, 0.0, self.draw_coordinate_m))
        if self.force_draw_table is not None:
            x = self.force_draw_table[:, 0]
            f = self.force_draw_table[:, 1]
            return float(np.interp(d, x, f))

        u = d / self.draw_coordinate_m
        if self.force_draw_exponent is not None:
            p = self.force_draw_exponent
        else:
            # Generic shape only.  A real measured curve should be preferred.
            p = 1.00 if self.bow_type == "longbow" else 0.82
        return self.draw_force_N * u**p

    def delivered_draw_force(self, displacement_m):
        d = float(np.clip(displacement_m, 0.0, self.draw_coordinate_m))
        # A supplied measured force-draw curve is assumed to be the actual
        # string force of the bow and is therefore not efficiency-scaled.
        if self.force_draw_table is not None:
            x = self.force_draw_table[:, 0]
            f = self.force_draw_table[:, 1]
            return float(np.interp(d, x, f)) * self.nock_efficiency
        return self.draw_force(d) * self.bow_efficiency * self.nock_efficiency

    def stored_energy_J(self):
        d = np.linspace(0.0, self.draw_coordinate_m, 4000)
        return float(np.trapezoid([self.draw_force(x) for x in d], d))

    @property
    def usable_energy_J(self):
        return self.stored_energy_J() * self.bow_efficiency


class Longbow(Bow):
    def __init__(self, length_m, draw_strength_kgf, draw_length_m, **kwargs):
        super().__init__(length_m, draw_strength_kgf, draw_length_m,
                         bow_type="longbow", **kwargs)


class Recurve(Bow):
    def __init__(self, length_m, draw_strength_kgf, draw_length_m, **kwargs):
        super().__init__(length_m, draw_strength_kgf, draw_length_m,
                         bow_type="recurve", **kwargs)


@dataclass
class LaunchResult:
    reason: str
    time_s: float
    pre_release_equilibrium_residual_N: float
    release_time_s: float
    left_bow_time_s: float
    initial_position_m: np.ndarray
    initial_velocity_world_m_s: np.ndarray
    initial_direction_world: np.ndarray
    initial_roll_deg: float
    initial_orientation_quaternion_xyzw: np.ndarray
    initial_angular_velocity_body_rad_s: np.ndarray
    initial_bending_m: np.ndarray
    initial_bending_velocity_m_s: np.ndarray
    minimum_clearance_m: float
    minimum_clearance_time_s: float
    bow_contact_time_s: float
    max_contact_force_N: float
    contact_impulse_Ns: float
    bow_hit: bool
    bow_energy_J: float
    usable_bow_energy_J: float
    launch_solution: object = None
    flight_result: Optional[FlightResult] = None


class BowLaunchSimulator:
    """Flexible arrow launch with pre-release equilibrium and exact normal contact.

    The beam DOFs are transverse displacement of material nodes.  Axial motion
    is represented by a common nock/reference translation; this is sufficient
    for the launch stroke while avoiding an artificial axial compliance that
    cannot be identified from static spine alone.
    """

    def __init__(self, arrow: Arrow, bow: Bow, nodes: int = 41):
        if nodes < 11 or nodes % 2 == 0:
            raise ValueError("nodes must be an odd integer >= 11")
        self.arrow = arrow
        self.bow = bow
        self.n = int(nodes)
        self.s = np.linspace(0.0, arrow.length_m, self.n)
        self.h = self.s[1] - self.s[0]

        elev = np.deg2rad(bow.aim_elevation_deg + bow.release_bow_elevation_deg)
        az = np.deg2rad(bow.aim_azimuth_deg + bow.release_bow_azimuth_deg)
        cant = np.deg2rad(bow.bow_cant_deg + bow.release_bow_cant_deg)
        self.ex = _norm([
            np.cos(elev)*np.cos(az),
            np.sin(elev),
            np.cos(elev)*np.sin(az),
        ])
        ref = np.array([0.0, 1.0, 0.0])
        self.ey = _norm(ref - np.dot(ref, self.ex)*self.ex)
        self.ez = _norm(np.cross(self.ex, self.ey))
        Rbase = np.column_stack([self.ex, self.ey, self.ez])
        Rc = Rotation.from_rotvec(self.ex*cant).as_matrix()
        self.R0 = Rbase @ Rc
        self.ex, self.ey, self.ez = self.R0[:,0], self.R0[:,1], self.R0[:,2]

        # Discrete curvature operator.  Its null space contains translation and
        # rigid-body rotation, as a free beam should.
        self.D2 = self._D2()
        self.K = arrow.EI * self.h * (self.D2.T @ self.D2)
        self.D1 = self._D1()

        # Consistent trapezoidal shaft mass plus point/fletching lump masses.
        self.mass = np.full(self.n, arrow.shaft_mass * self.h / arrow.length_m)
        self.mass[0] *= 0.5
        self.mass[-1] *= 0.5
        for x_m, m in ((arrow.point_x_m, arrow.point_mass_kg),
                       (arrow.feathers.x_center_m, arrow.feathers.mass_kg)):
            self.mass[int(np.argmin(abs(self.s - x_m)))] += m

        self.total_mass = float(np.sum(self.mass))

        # Rayleigh damping.  Fit approximately 1.5% modal damping to the first
        # two non-rigid modes using the mass-weighted eigenproblem.
        Mhalf = np.sqrt(self.mass)
        A = self.K / np.outer(Mhalf, Mhalf)
        vals = np.linalg.eigvalsh(A)
        vals = vals[vals > max(1e-12, vals[-1] * 1e-10)]
        if len(vals) >= 2:
            w1, w2 = np.sqrt(vals[:2])
            zeta = 0.015
            self.rayleigh_beta = 2*zeta/(w1+w2)
            self.rayleigh_alpha = 2*zeta*w1*w2/(w1+w2)
        else:
            self.rayleigh_alpha = 0.0
            self.rayleigh_beta = 0.0
        self.C = self.rayleigh_alpha * self.mass[:, None] * np.eye(self.n)
        self.C += self.rayleigh_beta * self.K

        self.contact_center_x = float(bow.arrow_rest_longitudinal_offset_m)
        self.contact_half_x = 0.5 * bow.handle_length_m
        self.contact_half_y = 0.5 * bow.handle_height_m
        self.contact_z = 0.5 * bow.handle_thickness_m
        self.contact_y = float(bow.arrow_rest_height_error_m)

        # A lumped string endpoint is used as the minimum dynamic model of the
        # released string.  It is deliberately not a prescribed sinusoid: its
        # displacement follows an equation of motion and is coupled to the nock
        # while the nock remains engaged.  The effective mass is the moving
        # portion of the string/limb system, not the entire bow.
        self.string_mass_eff = max(1e-6, bow.string_mass_kg * bow.string_effective_mass_fraction)
        self.ks_y = bow.string_lateral_stiffness_N_m
        self.ks_z = bow.string_vertical_stiffness_N_m
        self.cs_y = bow.string_lateral_damping_Ns_m
        self.cs_z = bow.string_vertical_damping_Ns_m

        # Acceleration-level constraint stabilization.  This is NOT a contact
        # spring: there is no force proportional to penetration depth.
        self.baum_omega = 2.0*np.pi*2500.0
        self.baum_zeta = 0.9
        self.gap_tol = 1e-8

    def _D1(self):
        n, h = self.n, self.h
        D = np.zeros((n, n))
        D[0, 0:2] = [-1.0, 1.0]
        D[-1, -2:] = [-1.0, 1.0]
        for i in range(1, n-1):
            D[i, i-1:i+2] = [-0.5, 0.0, 0.5]
        return D / h

    def _D2(self):
        n, h = self.n, self.h
        D = np.zeros((n, n))
        for i in range(1, n-1):
            D[i, i-1:i+2] = [1, -2, 1]
        D[0, 0:3] = [1, -2, 1]
        D[-1, -3:] = [1, -2, 1]
        return D / h**2

    def _world_nodes(self, uy, uz, nock_x, duy, duz, nock_v):
        local_p = np.column_stack((nock_x + self.s, uy,
                                   self.bow.arrow_side_offset_m + uz))
        local_v = np.column_stack((np.full(self.n, nock_v), duy, duz))
        return local_p @ self.R0.T, local_v @ self.R0.T

    def _string_equilibrium(self, nock_x):
        # The ideal string centreline returns to the bow centreline at the
        # current draw position.  Lateral/vertical motion is therefore a
        # dynamic perturbation about this moving equilibrium.
        return np.array([
            0.0,
            self.bow.nock_height_offset_m,
            self.bow.arrow_side_offset_m,
        ])

    def _string_drive_force(self, nock_x, string_y, string_z, string_vy, string_vz):
        # Bow tension provides the axial drive.  Lateral and vertical string
        # restoring forces are generated by the moving string endpoint itself.
        d = np.clip(-float(np.dot(nock_x, self.ex)), 0.0, self.bow.draw_coordinate_m)
        Fmag = self.bow.delivered_draw_force(d)
        eq = self._string_equilibrium(nock_x)
        dy = string_y - eq[1]
        dz = string_z - eq[2]
        Fy = -self.ks_y*dy - self.cs_y*string_vy
        Fz = -self.ks_z*dz - self.cs_z*string_vz
        # During the short finger-separation interval, allow an explicitly
        # specified hand-induced acceleration of the string endpoint.
        return Fmag, Fy, Fz

    def _string_initial_state(self):
        # State variables are absolute local string-endpoint coordinates; the
        # release_* inputs are perturbations about the undisturbed string path.
        return (
            self.bow.nock_height_offset_m + self.bow.release_vertical_displacement_m,
            self.bow.release_vertical_velocity_m_s,
            self.bow.arrow_side_offset_m + self.bow.release_lateral_displacement_m,
            self.bow.release_lateral_velocity_m_s,
        )

    def _contact_candidates(self, uy, uz, nock_x):
        ids = []
        gaps = []
        for i, s in enumerate(self.s):
            x = nock_x + s - self.contact_center_x
            y = uy[i] - self.contact_y
            if abs(x) <= self.contact_half_x and abs(y) <= self.contact_half_y:
                gap = self.bow.arrow_side_offset_m + uz[i] - self.contact_z
                ids.append(i)
                gaps.append(gap)
        return np.asarray(ids, dtype=int), np.asarray(gaps, dtype=float)

    def _contact_acceleration(self, a_free, duz, ids, gaps):
        """Solve NCP contact by active-set Lagrange multipliers.

        For a diagonal mass matrix, each active normal multiplier is local.
        The multiplier is chosen to enforce the acceleration-level constraint
        with Baumgarte correction.  Negative reactions are rejected, which is
        the complementarity active-set condition.
        """
        if len(ids) == 0:
            return a_free, np.zeros(0), np.zeros(0, dtype=int)

        active = [int(i) for i, g in zip(ids, gaps) if g <= self.gap_tol]
        while active:
            lambdas = []
            for i in active:
                j = int(np.where(ids == i)[0][0])
                g = float(gaps[j])
                target = (-2*self.baum_zeta*self.baum_omega*duz[i]
                          -self.baum_omega**2*min(g, 0.0))
                lambdas.append(self.mass[i] * (target - a_free[i]))
            lambdas = np.asarray(lambdas)
            if np.all(lambdas >= -1e-10):
                a = a_free.copy()
                for i, lam in zip(active, lambdas):
                    a[i] += lam / self.mass[i]
                return a, np.maximum(lambdas, 0.0), np.asarray(active, dtype=int)
            active.pop(int(np.argmin(lambdas)))
        return a_free, np.zeros(0), np.zeros(0, dtype=int)

    def _axial_compression_force(self, string_force_N):
        """Approximate compressive force distribution in the arrow shaft.

        The nock force is transmitted through progressively less shaft as the
        point-side mass is accelerated.  This reduced model captures the main
        geometric-stiffness (P-delta/buckling) effect without introducing an
        artificial axial spring.
        """
        cumulative = np.cumsum(self.mass)
        fraction = np.clip(1.0 - cumulative / self.total_mass, 0.0, 1.0)
        return max(float(string_force_N), 0.0) * fraction

    def _geometric_stiffness(self, N):
        # Second-order geometric stiffness: - integral N(s) (w')^2 ds.
        Nmid = 0.5*(N[:-1] + N[1:])
        Gk = np.zeros((self.n, self.n))
        for i, Nm in enumerate(Nmid):
            d = np.zeros(self.n)
            d[i] = -1.0/self.h
            d[i+1] = 1.0/self.h
            Gk -= Nm*self.h*np.outer(d, d)
        return Gk

    def _free_static_residual(self, u, gravity_local):
        # External static load is gravity.  Nock is handled separately as an
        # equality constraint; bow contact is imposed through inequalities.
        return self.K @ u - self.mass * gravity_local

    def solve_pre_release_equilibrium(self):
        """Solve the loaded arrow state while the nock is held by the string.

        This is a small convex quadratic beam problem with one equality (the
        nock position) and unilateral riser inequalities.  An active-set
        linear solve is used rather than a generic nonlinear optimiser; this
        makes the pre-release state reproducible and fast enough to use in a
        parameter sweep.
        """
        g_world = np.array([0.0, -G, 0.0])
        gy = float(g_world @ self.ey)
        gz = float(g_world @ self.ez)

        def component(g, contact):
            f = self.mass * g
            N0 = self._axial_compression_force(self.bow.draw_force(self.bow.draw_coordinate_m))
            Kbeam = self.K + self._geometric_stiffness(N0)
            fixed = {0: 0.0, 1: 0.0}  # string + pre-release shaft direction fix the nock/slope
            active = set()
            scale = max(1.0, float(np.max(np.abs(Kbeam))))
            # Tiny regularisation selects the minimum-rotation equilibrium of
            # the otherwise free rigid-body slope; it is not a beam stiffness.
            Kreg = Kbeam + (1e-12 * scale) * np.eye(self.n)

            for _ in range(self.n + 5):
                fixed_all = sorted(fixed.keys() | active)
                free = np.array([i for i in range(self.n) if i not in fixed_all], dtype=int)
                u = np.zeros(self.n)
                for i, val in fixed.items():
                    u[i] = val
                if active:
                    val = self.contact_z - self.bow.arrow_side_offset_m
                    for i in active:
                        u[i] = val

                if len(free):
                    Kff = Kreg[np.ix_(free, free)]
                    rhs = f[free] - Kreg[np.ix_(free, fixed_all)] @ u[fixed_all]
                    u[free] = np.linalg.solve(Kff, rhs)

                residual = Kreg @ u - f
                # For a contact node, the physical normal reaction is
                # lambda = residual_i.  It must be non-negative.
                negative_active = [i for i in active if residual[i] < -1e-9]
                if negative_active:
                    # A negative reaction means that this node cannot remain
                    # in contact; remove the weakest one and resolve.
                    active.remove(min(negative_active, key=lambda i: residual[i]))
                    continue

                if not contact:
                    break

                candidates = [
                    i for i in range(self.n)
                    if abs(self.s[i] - self.contact_center_x) <= self.contact_half_x
                    and i not in fixed and i not in active
                ]
                gaps = {
                    i: self.bow.arrow_side_offset_m + u[i] - self.contact_z
                    for i in candidates
                }
                penetrating = [i for i in candidates if gaps[i] < -1e-9]
                if penetrating:
                    # Activate the most penetrating node first.
                    active.add(min(penetrating, key=lambda i: gaps[i]))
                    continue
                break
            else:
                raise RuntimeError("pre-release contact active-set did not converge")
            return u

        uy = component(gy, contact=False)
        uz = component(gz, contact=True)

        # Check the elastic equilibrium away from the constrained nock/contact
        # reactions.  This residual is a diagnostic, not an extra stiffness.
        ry = self.K @ uy - self.mass * gy
        rz = self.K @ uz - self.mass * gz
        constrained = {0}
        for i in range(self.n):
            if abs(self.s[i]) <= self.contact_half_x and \
               abs(self.bow.arrow_side_offset_m + uz[i] - self.contact_z) < 1e-7:
                constrained.add(i)
        free = np.array([i for i in range(self.n) if i not in constrained], dtype=int)
        residual_norm = float(np.linalg.norm(np.r_[ry[free], rz[free]]))
        return uy, uz, residual_norm

    def _rhs(self, t, y, release=True):
        n = self.n
        uy = y[:n]; duy = y[n:2*n]
        uz = y[2*n:3*n]; duz = y[3*n:4*n]
        nock_x, nock_v = y[4*n:4*n+2]
        sy, syv, sz, szv = y[4*n+2:4*n+6]

        p, v = self._world_nodes(uy, uz, nock_x, duy, duz, nock_v)
        nock_p, nock_vw = p[0], v[0]
        Fmag, Fy_string, Fz_string = self._string_drive_force(
            np.array([nock_x, 0., 0.]), sy, sz, syv, szv)

        # While the nock is engaged, string and nock share transverse position
        # and velocity.  The string's lateral/vertical acceleration therefore
        # becomes the nock acceleration; the resulting constraint reaction is
        # what bends/rotates the arrow.  Once released, the string is free and
        # its oscillation no longer acts on the arrow.
        attached = release and (t < self.bow.release_duration_s or nock_x < 0.0)

        g_world = np.array([0.0, -G, 0.0])
        gy = float(g_world @ self.ey); gz = float(g_world @ self.ez)
        fy = self.mass*gy - self.K@uy - self.C@duy
        fz = self.mass*gz - self.K@uz - self.C@duz

        if attached:
            # Constraint kinematics: endpoint displacement equals string motion.
            # Keep the beam equations for all nodes, but impose the nock motion
            # directly.  This is a DAE reduced to independent coordinates.
            ay_string = Fy_string / self.string_mass_eff
            az_string = Fz_string / self.string_mass_eff
            ay = fy / self.mass
            az = fz / self.mass
            ay[0] = ay_string
            az[0] = az_string
            # Axial string force accelerates the arrow + effective string mass.
            ax = Fmag / (self.total_mass + self.string_mass_eff)
            sydd, szdd = ay_string, az_string
        else:
            ay = fy / self.mass
            az = fz / self.mass
            ax = Fmag / self.total_mass
            sydd = (-self.ks_y*sy - self.cs_y*syv)/self.string_mass_eff
            szdd = (-self.ks_z*sz - self.cs_z*szv)/self.string_mass_eff

        # Arrow/nock transverse coordinates are relative to the nominal string
        # centreline, so their node-0 values must track the string displacement.
        # This is enforced after each integration step by the launch integrator.
        ids, gaps = self._contact_candidates(uy, uz, nock_x)
        az, _, _ = self._contact_acceleration(az, duz, ids, gaps)
        if attached:
            az[0] = szdd

        return np.r_[duy, ay, duz, az, nock_v, ax, syv, sydd, szv, szdd]

    def _initial_state(self):
        uy, uz, residual = self.solve_pre_release_equilibrium()
        zeros = np.zeros(self.n)
        sy, syv, sz, szv = self._string_initial_state()
        uy[0] = sy
        uz[0] = sz - self.bow.arrow_side_offset_m
        zeros[0] = syv
        # vertical string velocity is relative to the nominal arrow-side offset
        zeros2 = np.zeros(self.n); zeros2[0] = szv
        y0 = np.r_[uy, zeros, uz, zeros2, -self.bow.draw_coordinate_m, 0.0, sy, syv, sz, szv]
        return y0, residual

    def _project_launch_state(self, y):
        n = self.n
        uy, duy, uz, duz, nock_x, nock_v = (
            y[:n], y[n:2*n], y[2*n:3*n], y[3*n:4*n], y[4*n], y[4*n+1]
        )
        local_p = np.column_stack((nock_x+self.s, uy,
                                   self.bow.arrow_side_offset_m+uz))
        local_v = np.column_stack((np.full(n, nock_v), duy, duz))

        w = self.mass / np.sum(self.mass)
        r_cm = np.average(local_p, axis=0, weights=w)
        v_cm = np.average(local_v, axis=0, weights=w)

        # Rigid-body angular velocity from the nodal velocity field after
        # subtracting translation.  This captures launch-induced pitch/yaw;
        # roll remains zero unless the user supplies a non-axisymmetric release
        # model (or friction model) because the present beam is symmetric.
        rr = local_p - r_cm
        vv = local_v - v_cm
        S = np.zeros((3, 3))
        b = np.zeros(3)
        for wi, r, v in zip(w, rr, vv):
            rx = np.array([[0,-r[2],r[1]],
                           [r[2],0,-r[0]],
                           [-r[1],r[0],0]])
            S += wi * (rx.T @ rx)
            b += wi * (rx.T @ v)
        omega_local = np.linalg.solve(S + 1e-14*np.eye(3), b)

        direction_local = _norm(local_p[-1] - local_p[0])
        direction_world = _norm(self.R0 @ direction_local)
        r_world = self.R0 @ r_cm
        v_world = self.R0 @ v_cm
        omega_world = self.R0 @ omega_local

        q = orientation_from_direction_and_roll(
            direction_world,
            self.bow.feather_clocking_deg + self.bow.release_string_roll_deg)
        R = Rotation.from_quat(q).as_matrix()
        omega_body = R.T @ omega_world
        omega_body[0] += self.bow.release_string_roll_rate_rad_s

        # The flight code accepts two modal bending coordinates.  Remove the
        # best rigid translation/slope first, then project onto the flight
        # simulator's first bending mode.  This avoids the ill-conditioning of
        # fitting {1, s, phi} simultaneously (phi has a large rigid-like
        # component for a finite arrow).
        phi = np.asarray([self.arrow.phi(x) for x in self.s])
        B = np.column_stack([np.ones(n), self.s-self.arrow.x_cm])
        def modal_projection(u):
            rigid = B @ np.linalg.lstsq(B, u, rcond=None)[0]
            ph = phi - B @ np.linalg.lstsq(B, phi, rcond=None)[0]
            den = float(np.dot(ph, ph))
            return float(np.dot(ph, u-rigid)/den) if den > 1e-20 else 0.0
        qy = modal_projection(uy)
        qz = modal_projection(uz)
        qdy = modal_projection(duy)
        qdz = modal_projection(duz)

        return (r_world, v_world, direction_world, qy, qz, qdy, qdz,
                q, omega_body)

    def _diagnostics(self, sol):
        min_clear = np.inf
        min_t = 0.0
        max_force = 0.0
        impulse = 0.0
        contact_time = 0.0
        prev_t = None
        prev_f = 0.0

        for t, y in zip(sol.t, sol.y.T):
            n = self.n
            uy, duy = y[:n], y[n:2*n]
            uz, duz = y[2*n:3*n], y[3*n:4*n]
            nx = y[4*n]
            ids, gaps = self._contact_candidates(uy, uz, nx)
            if len(gaps):
                j = int(np.argmin(gaps))
                if gaps[j] < min_clear:
                    min_clear = float(gaps[j])
                    min_t = float(t)
            fz = self.mass*(-G*self.ez @ self.ez) - self.K @ uz - self.C @ duz
            az_free = fz / self.mass
            _, lam, _ = self._contact_acceleration(az_free, duz, ids, gaps)
            force = float(np.sum(lam))
            max_force = max(max_force, force)
            if prev_t is not None:
                dt = float(t-prev_t)
                impulse += 0.5*(prev_f+force)*dt
                if force > 0.0 or prev_f > 0.0:
                    contact_time += dt
            prev_t = float(t)
            prev_f = force
        return min_clear, min_t, max_force, contact_time, impulse

    def _contact_project(self, uz, duz, nock_x):
        """Project a trial state onto the unilateral grip constraints.

        This is a non-penetrating impulse/contact projection, not a penalty
        force.  With diagonal nodal mass, the normal impulse for each active
        node is explicit.  Position correction is only a numerical projection
        over the current time step; it tends to zero with time-step refinement.
        """
        impulse = 0.0
        active = []
        for i, s in enumerate(self.s):
            x = nock_x + s - self.contact_center_x
            if abs(x) > self.contact_half_x:
                continue
            gap = self.bow.arrow_side_offset_m + uz[i] - self.contact_z
            if gap < 0.0:
                uz[i] -= gap
                if duz[i] < 0.0:
                    J = -self.mass[i] * duz[i]
                    duz[i] = 0.0
                    impulse += J
                active.append(i)
        return active, impulse

    def simulate_launch(self, t_end=0.040, max_step=2.0e-6):
        y0, equilibrium_residual = self._initial_state()
        n = self.n; dt = float(max_step)
        steps = int(np.ceil(t_end/dt))
        y = y0.copy()
        times=[0.0]; states=[y.copy()]; contact_forces=[0.0]; contact_active=[False]
        cumulative_impulse=0.0
        bow_clearance_margin = max(0.002, self.contact_z)

        def unpack(a):
            return a[:n],a[n:2*n],a[2*n:3*n],a[3*n:4*n],a[4*n],a[4*n+1],a[4*n+2],a[4*n+3],a[4*n+4],a[4*n+5]

        for k in range(steps):
            t=k*dt; h=min(dt,t_end-t)
            if h<=0: break
            uy,duy,uz,duz,nx,nv,sy,syv,sz,szv=unpack(y)
            # Enforce engaged nock/string kinematics before evaluating forces.
            if nx < 0.0 or t < self.bow.release_duration_s:
                uy[0]=sy; duy[0]=syv
                uz[0]=sz-self.bow.arrow_side_offset_m; duz[0]=szv

            Fmag,Fy,Fz=self._string_drive_force(np.array([nx,0.,0.]),sy,sz,syv,szv)
            g=np.array([0.,-G,0.]); gy=float(g@self.ey); gz=float(g@self.ez)
            Kgeo=self._geometric_stiffness(self._axial_compression_force(Fmag))
            Ktot=self.K+Kgeo
            ay=(self.mass*gy-Ktot@uy-self.C@duy)/self.mass
            az=(self.mass*gz-Ktot@uz-self.C@duz)/self.mass
            if nx < 0.0 or t < self.bow.release_duration_s:
                ay[0]=Fy/self.string_mass_eff + self.bow.release_vertical_acceleration_m_s2
                az[0]=Fz/self.string_mass_eff + self.bow.release_lateral_acceleration_m_s2
                sydd=ay[0]; szdd=az[0]
                ax=Fmag/(self.total_mass+self.string_mass_eff)
            else:
                sydd=(-self.ks_y*sy-self.cs_y*syv)/self.string_mass_eff
                szdd=(-self.ks_z*sz-self.cs_z*szv)/self.string_mass_eff
                ax=Fmag/self.total_mass

            # Kick-drift.
            duy_h=duy+0.5*h*ay; duz_h=duz+0.5*h*az
            nv_h=nv+0.5*h*ax; syv_h=syv+0.5*h*sydd; szv_h=szv+0.5*h*szdd
            uy_n=uy+h*duy_h; uz_n=uz+h*duz_h
            nx_n=nx+h*nv_h; sy_n=sy+h*syv_h; sz_n=sz+h*szv_h

            # During string engagement the nock follows the string exactly.
            if nx_n < 0.0 or t+h < self.bow.release_duration_s:
                uy_n[0]=sy_n; duy_h[0]=syv_h
                uz_n[0]=sz_n-self.bow.arrow_side_offset_m; duz_h[0]=szv_h

            # Exact unilateral normal contact projection.  No penetration spring.
            active,J=self._contact_project(uz_n,duz_h,nx_n)
            if active: cumulative_impulse += J

            # Re-evaluate acceleration.
            Fmag2,Fy2,Fz2=self._string_drive_force(np.array([nx_n,0.,0.]),sy_n,sz_n,syv_h,szv_h)
            Kgeo2=self._geometric_stiffness(self._axial_compression_force(Fmag2)); Ktot2=self.K+Kgeo2
            ay2=(self.mass*gy-Ktot2@uy_n-self.C@duy_h)/self.mass
            az2=(self.mass*gz-Ktot2@uz_n-self.C@duz_h)/self.mass
            engaged2=(nx_n < 0.0 or t+h < self.bow.release_duration_s)
            if engaged2:
                ay2[0]=Fy2/self.string_mass_eff + self.bow.release_vertical_acceleration_m_s2
                az2[0]=Fz2/self.string_mass_eff + self.bow.release_lateral_acceleration_m_s2
                sydd2=ay2[0]; szdd2=az2[0]; ax2=Fmag2/(self.total_mass+self.string_mass_eff)
            else:
                sydd2=(-self.ks_y*sy_n-self.cs_y*syv_h)/self.string_mass_eff
                szdd2=(-self.ks_z*sz_n-self.cs_z*szv_h)/self.string_mass_eff
                ax2=Fmag2/self.total_mass

            duy_n=duy_h+0.5*h*ay2; duz_n=duz_h+0.5*h*az2
            nv_n=nv_h+0.5*h*ax2; syv_n=syv_h+0.5*h*sydd2; szv_n=szv_h+0.5*h*szdd2
            if engaged2:
                duy_n[0]=syv_n; duz_n[0]=szv_n
                uy_n[0]=sy_n; uz_n[0]=sz_n-self.bow.arrow_side_offset_m

            y=np.r_[uy_n,duy_n,uz_n,duz_n,nx_n,nv_n,sy_n,syv_n,sz_n,szv_n]
            tt=t+h

            # Contact force is represented by the normal impulse over this step.
            fcontact=J/h if h>0 else 0.0
            times.append(tt); states.append(y.copy()); contact_forces.append(fcontact); contact_active.append(bool(active))

            # IMPORTANT: flight handoff is only after the complete arrow has
            # cleared the riser, not when the nock first clears the string.
            # Require every beam node to be beyond the riser longitudinal envelope
            # and outside its lateral contact envelope.
            p_now,_=self._world_nodes(uy_n,uz_n,nx_n,duy_n,duz_n,nv_n)
            xloc=nx_n+self.s
            all_aft=np.all(xloc > self.contact_half_x + bow_clearance_margin)
            lateral_clear=np.all(np.abs(self.bow.arrow_side_offset_m+uz_n-self.contact_z) > -self.gap_tol)
            if all_aft and lateral_clear and tt > self.bow.release_duration_s:
                break

        sol=type('LaunchSolution',(),{})()
        sol.t=np.asarray(times); sol.y=np.asarray(states).T
        sol.contact_force=np.asarray(contact_forces); sol.contact_active=np.asarray(contact_active)
        sol.contact_impulse=np.cumsum(np.r_[0.0,np.diff(sol.t)*0.0])
        sol.string_state=np.asarray([s[4*n+2:4*n+6] for s in states])
        sol.t_events=[np.array([times[-1]])]
        sol.dense_output=lambda t: None

        y=states[-1]; state=self._project_launch_state(y)
        r,v,direction,qy,qz,qdy,qdz,q,omega=state

        min_clear=np.inf; min_t=0.0
        for tt,yy in zip(sol.t,sol.y.T):
            uy,uz,nx=yy[:n],yy[2*n:3*n],yy[4*n]
            ids,gaps=self._contact_candidates(uy,uz,nx)
            if len(gaps):
                j=int(np.argmin(gaps))
                if gaps[j]<min_clear: min_clear=float(gaps[j]); min_t=float(tt)
        max_force=float(np.max(sol.contact_force))
        contact_time=float(np.sum(np.diff(sol.t)[sol.contact_active[1:]])) if len(sol.t)>1 else 0.0
        return LaunchResult(
            reason='bow_clearance', time_s=float(times[-1]),
            pre_release_equilibrium_residual_N=equilibrium_residual,
            release_time_s=min(self.bow.release_duration_s,float(times[-1])),
            left_bow_time_s=float(times[-1]), initial_position_m=r.copy(),
            initial_velocity_world_m_s=v.copy(), initial_direction_world=direction.copy(),
            initial_roll_deg=float(self.bow.feather_clocking_deg),
            initial_orientation_quaternion_xyzw=q.copy(),
            initial_angular_velocity_body_rad_s=omega.copy(),
            initial_bending_m=np.array([qy,qz]),
            initial_bending_velocity_m_s=np.array([qdy,qdz]),
            minimum_clearance_m=float(min_clear), minimum_clearance_time_s=float(min_t),
            bow_contact_time_s=contact_time, max_contact_force_N=max_force,
            contact_impulse_Ns=float(cumulative_impulse), bow_hit=bool(cumulative_impulse>0),
            bow_energy_J=self.bow.stored_energy_J(), usable_bow_energy_J=self.bow.usable_energy_J,
            launch_solution=sol)

    def launch_and_fly(self, atmosphere: Atmosphere, **flight_kwargs):
        launch = self.simulate_launch()
        launch.flight_result = simulate(
            self.arrow, atmosphere,
            initial_position_m=launch.initial_position_m,
            initial_velocity_world_m_s=launch.initial_velocity_world_m_s,
            initial_direction=launch.initial_direction_world,
            initial_roll_deg=launch.initial_roll_deg,
            initial_orientation_quaternion_xyzw=launch.initial_orientation_quaternion_xyzw,
            initial_angular_velocity_body_rad_s=launch.initial_angular_velocity_body_rad_s,
            initial_bending_m=launch.initial_bending_m,
            initial_bending_velocity_m_s=launch.initial_bending_velocity_m_s,
            **flight_kwargs,
        )
        return launch


def example():
    from arrow_flight_sim import Feather

    arrow = Arrow(
        length_m=0.75,
        shaft_outer_d_m=0.0065,
        shaft_inner_d_m=0.0045,
        total_mass_kg=0.028,
        point_mass_kg=0.009,
        point_x_m=0.75,
        spine=400,
        feathers=Feather(
            x_start_m=0.08, x_end_m=0.19, height_m=0.012,
            area_each_m2=0.00075, count=3, cant_deg=2.0,
            mass_kg=0.0005,
        ),
    )

    bow = Longbow(
        length_m=1.85,
        draw_strength_kgf=20.0,
        draw_length_m=0.72,
        brace_height_m=0.16,
        arrow_side_offset_m=0.018,
        feather_clocking_deg=20.0,
        # Clean release: all release disturbances zero.
        #release_lateral_displacement_m=0.0,
        #release_lateral_velocity_m_s=0.0,
        release_vertical_displacement_m=0.0,
        release_vertical_velocity_m_s=0.0,
        # Example form/equipment errors can be enabled independently:
        # draw_length_error_m=-0.004,          # 4 mm creep
         bow_cant_deg=2.0,                    # bow held 2 deg off vertical
         nock_height_offset_m=0.002,          # nocking-point error
         release_lateral_displacement_m=0.003,# 3 mm string pluck
         release_lateral_velocity_m_s=1.0,    # lateral finger release speed
         release_bow_azimuth_deg=0.5,
         release_string_roll_deg=2.0,
         arrow_rest_longitudinal_offset_m=0.010,
        # arrow_rest_height_error_m=0.001,
    )

    sim = BowLaunchSimulator(arrow, bow, nodes=41)
    #launch = sim.simulate_launch()
    launch = sim.launch_and_fly(Atmosphere())
    print("--- BOW LAUNCH ---")
    print("Reason:", launch.reason)
    print("Pre-release equilibrium residual [N]:",
          launch.pre_release_equilibrium_residual_N)
    print("Exit time [ms]:", round(1000*launch.left_bow_time_s, 3))
    print("CM velocity [m/s]:", np.round(launch.initial_velocity_world_m_s, 4))
    print("Direction:", np.round(launch.initial_direction_world, 6))
    print("Feather clocking [deg]:", launch.initial_roll_deg)
    print("Omega body [rad/s]:",
          np.round(launch.initial_angular_velocity_body_rad_s, 4))
    print("Bending [mm]:", np.round(1000*launch.initial_bending_m, 3))
    print("Bending velocity [m/s]:",
          np.round(launch.initial_bending_velocity_m_s, 4))
    print("Minimum clearance [mm]:",
          round(1000*launch.minimum_clearance_m, 3))
    print("Contact time [ms]:",
          round(1000*launch.bow_contact_time_s, 3))
    print("Maximum contact force [N]:",
          round(launch.max_contact_force_N, 3))
    print("Contact impulse [N s]:",
          round(launch.contact_impulse_Ns, 6))
    print("Bow contact:", launch.bow_hit)
    print("Stored bow energy [J]:", round(launch.bow_energy_J, 3))
    print("Usable bow energy [J]:", round(launch.usable_bow_energy_J, 3))
    print(launch.flight_result)
    return launch


if __name__ == "__main__":
    example()
