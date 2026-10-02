from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

# Coordinate system used everywhere:
#   +X = forward / range
#   +Y = upward
#   +Z = sideways
#
# Arrow body frame:
#   +x = nock -> point (shaft/longitudinal axis)
#   +y, +z = transverse axes fixed in the arrow.
#
# Quaternions use scipy Rotation convention [x, y, z, w].


G = 9.80665
R_AIR = 287.05


@dataclass
class Atmosphere:
    pressure_pa: float = 101325.0
    temperature_k: float = 288.15

    @property
    def rho(self) -> float:
        return self.pressure_pa / (R_AIR * self.temperature_k)


@dataclass
class Feather:
    x_start_m: float = 0.05
    x_end_m: float = 0.20
    height_m: float = 0.012
    area_each_m2: float = 0.0008
    count: int = 3
    cant_deg: float = 0.0
    mass_kg: float = 0.0005
    cd: float = 1.2
    lift_slope_per_rad: float = 3.0
    stall_deg: float = 18.0

    @property
    def x_center_m(self) -> float:
        return 0.5 * (self.x_start_m + self.x_end_m)


@dataclass
class Arrow:
    length_m: float
    shaft_outer_d_m: float
    shaft_inner_d_m: float
    total_mass_kg: float
    point_mass_kg: float
    point_x_m: float
    spine: float
    feathers: Feather = field(default_factory=Feather)

    # Existing aerodynamic parameters are retained.
    shaft_cd: float = 1.0
    point_cd: float = 0.5
    point_area_m2: float = 0.0001

    # Optional refinements.  Defaults are conservative and can be replaced
    # with measured/manufacturer data.
    shaft_axial_cd: float = 0.02
    air_viscosity_pa_s: float = 1.81e-5
    shaft_drag_segments: int = 12
    center_of_pressure_point_fraction: float = 1.0

    # Static-spine reference: 28 in span, 1.94 lb center load.
    spine_test_span_m: float = 28.0 * 0.0254
    spine_test_mass_kg: float = 1.94 * 0.45359237

    def __post_init__(self):
        if self.length_m <= 0 or self.shaft_outer_d_m <= 0:
            raise ValueError("length_m and shaft_outer_d_m must be positive")
        if not 0 <= self.shaft_inner_d_m < self.shaft_outer_d_m:
            raise ValueError("shaft_inner_d_m must be >= 0 and < outer diameter")
        if self.total_mass_kg <= 0 or self.point_mass_kg < 0:
            raise ValueError("invalid mass values")
        if not 0 <= self.point_x_m <= self.length_m:
            raise ValueError("point_x_m must lie between 0 and length_m")
        if self.spine <= 0:
            raise ValueError("spine must be positive")
        if self.feathers.count < 0:
            raise ValueError("feather count cannot be negative")
        if self.shaft_drag_segments < 1:
            raise ValueError("shaft_drag_segments must be >= 1")

        self.A = np.pi / 4 * (
            self.shaft_outer_d_m**2 - self.shaft_inner_d_m**2
        )
        self.I = np.pi / 64 * (
            self.shaft_outer_d_m**4 - self.shaft_inner_d_m**4
        )

        # Static spine number = deflection in inches * 1000.
        self.spine_deflection_m = self.spine / 1000.0 * 0.0254
        F_test = self.spine_test_mass_kg * G

        # Existing reference interpretation is retained:
        # delta = F L^3 / (48 E I)
        self.EI = (
            F_test * self.spine_test_span_m**3
            / (48.0 * self.spine_deflection_m)
        )

        self.shaft_mass = (
            self.total_mass_kg
            - self.point_mass_kg
            - self.feathers.mass_kg
        )
        if self.shaft_mass <= 0:
            raise ValueError(
                "total_mass_kg must exceed point_mass_kg + feather mass"
            )

        # First bending eigenvalue for a uniform cantilever-like mode.
        self.lambda_b = 1.875104068711961
        m, L = self.shaft_mass, self.length_m
        mp, mf = self.point_mass_kg, self.feathers.mass_kg

        self.x_cm = (
            m * L / 2
            + mp * self.point_x_m
            + mf * self.feathers.x_center_m
        ) / self.total_mass_kg

        # Longitudinal mass moments of inertia.
        self.Iyy = (
            m * (L**2 / 12 + (L / 2 - self.x_cm) ** 2)
            + mp * (self.point_x_m - self.x_cm) ** 2
            + mf * (self.feathers.x_center_m - self.x_cm) ** 2
        )
        self.Izz = self.Iyy

        radius = self.shaft_outer_d_m / 2
        # Approximation: point/shaft mass distributed about a solid circular
        # section. Feather contribution about x is normally very small.
        self.Ixx = 0.5 * (m + mp) * radius**2

        # Two transverse first bending modes.  Keeping separate q_y/q_z
        # permits arbitrary 3-D initial velocity and yaw/pitch disturbances.
        self.modal_mass = 0.236 * self.shaft_mass
        self.modal_k = (
            self.lambda_b**4 * self.EI / self.length_m**3
        ) * 0.236
        self.modal_damping_ratio = 0.02
        self.modal_c = (
            2 * self.modal_damping_ratio
            * np.sqrt(self.modal_k * self.modal_mass)
        )

    def _mode_raw(self, x):
        s = np.asarray(x) / self.length_m
        b = self.lambda_b
        sigma = (np.cosh(b) + np.cos(b)) / (np.sinh(b) + np.sin(b))
        return (
            np.cosh(b * s)
            - np.cos(b * s)
            - sigma * (np.sinh(b * s) - np.sin(b * s))
        )

    def phi(self, x):
        return self._mode_raw(x) / self._mode_raw(self.length_m)

    def phi_prime(self, x):
        """Derivative d(phi)/dx, useful for the local flexible centerline."""
        x = np.asarray(x, dtype=float)
        b = self.lambda_b
        L = self.length_m
        sigma = (np.cosh(b) + np.cos(b)) / (np.sinh(b) + np.sin(b))
        denom = self._mode_raw(L)
        return (
            (b / L)
            * (
                np.sinh(b * x / L)
                + np.sin(b * x / L)
                - sigma * (
                    np.cosh(b * x / L) - np.cos(b * x / L)
                )
            )
            / denom
        )


@dataclass
class Target:
    """Plane/point target specification.

    distance_m is the desired target X coordinate relative to the launch
    origin. height_m is the desired Y coordinate. The target is a plane at
    X=distance_m unless a finite radius is supplied.

    z_m is optional. If supplied, the target point is (distance, height, z).
    If None, z is unconstrained and the target is a horizontal/vertical
    reference at the requested x/y.
    """
    distance_m: float
    height_m: float
    z_m: Optional[float] = None
    radius_m: Optional[float] = None


@dataclass
class CheckpointCrossing:
    name: str
    x_m: float
    time_s: float
    tip_position_world_m: np.ndarray
    cm_position_world_m: np.ndarray
    tip_minus_checkpoint_m: np.ndarray
    velocity_tip_world_m_s: np.ndarray
    arrow_direction_world: np.ndarray


@dataclass
class FlightResult:
    reason: str
    time_s: float

    # Requested dual position tracking.
    position_cm_world_m: np.ndarray
    position_tip_world_m: np.ndarray

    direction_world: np.ndarray
    orientation_quaternion_xyzw: np.ndarray
    velocity_cm_world_m_s: np.ndarray
    velocity_tip_world_m_s: np.ndarray
    angular_velocity_body_rad_s: np.ndarray

    bending_amplitude_y_m: float
    bending_amplitude_z_m: float

    target_error_m: Optional[np.ndarray] = None
    target_hit: bool = False
    checkpoint_crossings: list = field(default_factory=list)
    solution: object = None

    @property
    def speed_m_s(self) -> float:
        return float(np.linalg.norm(self.velocity_cm_world_m_s))


def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Direction/vector must have non-zero length")
    return v / n


def orientation_from_direction_and_roll(
    direction_world,
    roll_deg=0.0,
    reference_up=np.array([0.0, 1.0, 0.0]),
):
    """Build attitude with body +X along direction and a specified roll."""
    ex = _normalize(direction_world)
    up = _normalize(reference_up)

    if abs(np.dot(ex, up)) > 0.98:
        up = np.array([0.0, 0.0, 1.0])

    ey = _normalize(up - np.dot(up, ex) * ex)
    ez = _normalize(np.cross(ex, ey))
    R0 = np.column_stack([ex, ey, ez])

    a = np.deg2rad(roll_deg)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(a), -np.sin(a)],
            [0, np.sin(a), np.cos(a)],
        ]
    )
    return Rotation.from_matrix(R0 @ Rx).as_quat()


def quat_derivative(q, omega_body):
    """dq/dt for scipy's [x,y,z,w] quaternion convention."""
    x, y, z, w = q
    wx, wy, wz = omega_body
    return 0.5 * np.array(
        [
            w * wx + y * wz - z * wy,
            w * wy + z * wx - x * wz,
            w * wz + x * wy - y * wx,
            -x * wx - y * wy - z * wz,
        ]
    )


def aerodynamic_force(
    v_rel_world,
    axis_world,
    rho,
    area,
    cd,
    lift_slope=0.0,
    stall_rad=np.deg2rad(18.0),
):
    """Simple body/point aerodynamic model.

    Drag is opposite relative wind. Lift is perpendicular to the
    velocity/axis plane and uses a clipped linear lift curve.
    """
    speed = np.linalg.norm(v_rel_world)
    if speed < 1e-12:
        return np.zeros(3)

    flow = v_rel_world / speed
    qdyn = 0.5 * rho * speed**2

    n = axis_world - np.dot(axis_world, flow) * flow
    nn = np.linalg.norm(n)
    drag = -qdyn * cd * area * flow

    if nn < 1e-10 or lift_slope == 0:
        return drag

    n /= nn
    alpha = np.arccos(np.clip(np.dot(axis_world, -flow), -1, 1))
    alpha *= np.sign(np.dot(n, axis_world))
    alpha = np.clip(alpha, -stall_rad, stall_rad)

    return drag + qdyn * (lift_slope * alpha) * area * n


def shaft_segment_force(
    v_rel_world,
    tangent_world,
    rho,
    ds,
    diameter,
    cd_crossflow,
    cd_axial,
):
    """Aerodynamic force on a short cylindrical shaft segment.

    Cross-flow drag uses projected area d*ds*sin(alpha), while a small
    axial skin-friction-like term prevents the shaft from having literally
    zero axial drag. This is substantially less angle-insensitive than
    applying Cd*d*L at the CM.
    """
    speed = np.linalg.norm(v_rel_world)
    if speed < 1e-12:
        return np.zeros(3)

    flow = v_rel_world / speed
    tangent = _normalize(tangent_world)
    qdyn = 0.5 * rho * speed**2

    axial_fraction = np.dot(tangent, flow)
    cross_fraction = np.sqrt(max(0.0, 1.0 - axial_fraction**2))

    A_cross = diameter * ds * cross_fraction
    F_cross = -qdyn * cd_crossflow * A_cross * flow

    A_axial = np.pi * diameter * ds
    F_axial = (
        -qdyn * cd_axial * A_axial * axial_fraction * tangent
    )

    return F_cross + F_axial


def _tip_state(arrow, y):
    """Return tip position/velocity from CM state, including both bends."""
    r = y[:3]
    v = y[3:6]
    quat = y[6:10] / np.linalg.norm(y[6:10])
    omega = y[10:13]
    qb_y, qd_y, qb_z, qd_z = y[13:17]

    Rbw = Rotation.from_quat(quat).as_matrix()
    ex, ey, ez = Rbw[:, 0], Rbw[:, 1], Rbw[:, 2]

    phi_t = float(arrow.phi(arrow.length_m))
    transverse_bend_b = np.array(
        [arrow.length_m - arrow.x_cm, qb_y * phi_t, qb_z * phi_t]
    )

    tip_position = r + Rbw @ transverse_bend_b

    local_tip_velocity = (
        np.cross(omega, transverse_bend_b)
        + np.array([0.0, qd_y * phi_t, qd_z * phi_t])
    )
    tip_velocity = v + Rbw @ local_tip_velocity

    direction = _normalize(
        ex
        + arrow.phi_prime(arrow.length_m) * (qb_y * ey + qb_z * ez)
    )

    return tip_position, tip_velocity, direction


def _initial_quaternion_from_inputs(
    initial_direction,
    initial_roll_deg,
    initial_orientation_quaternion_xyzw,
):
    if initial_orientation_quaternion_xyzw is not None:
        q = _normalize(initial_orientation_quaternion_xyzw)
        R = Rotation.from_quat(q).as_matrix()
        # The caller is responsible for supplying a physically consistent
        # attitude.  Direction is not forced to equal velocity.
        return q, R

    q = orientation_from_direction_and_roll(
        initial_direction, initial_roll_deg
    )
    return q, Rotation.from_quat(q).as_matrix()


def simulate(
    arrow,
    atmosphere,
    initial_position_m,
    initial_velocity_world_m_s,
    initial_direction,
    initial_roll_deg=0.0,
    initial_angular_velocity_body_rad_s=None,
    wind_world_m_s=None,
    target: Optional[Target] = None,
    checkpoints: Optional[Sequence[float]] = None,
    checkpoint_names: Optional[Sequence[str]] = None,
    ground_y_m=0.0,
    stop_at_ground=True,
    t_end=5.0,
    dt=0.001,
    initial_bending_m=(0.0, 0.0),
    initial_bending_velocity_m_s=(0.0, 0.0),
    initial_orientation_quaternion_xyzw=None,
    target_radius_m=0.005,
):
    """Integrate a 6-DOF arrow with a two-axis first bending mode.

    Required initial conditions:
      * initial_velocity_world_m_s: arbitrary 3-D CM velocity.
      * initial_direction: arrow longitudinal direction, independent of velocity.
      * initial_roll_deg: feather/shaft roll at t=0.
      * initial_angular_velocity_body_rad_s: [pitch, yaw, spin] in body axes.

    Optional:
      * initial_orientation_quaternion_xyzw can replace direction+roll when
        a complete measured attitude is available.
      * target is specified by distance (X) and height (Y), optionally Z.
        Integration stops at the target X crossing and reports the target
        error of the arrow tip.
      * checkpoints may be X coordinates, [X,Y,Z] points, or dictionaries
        with x_m/y_m/z_m. A checkpoint is recorded when the ARROW TIP
        crosses its X coordinate in the forward direction.

    Returned positions are world coordinates [x, y, z], with Y upward.
    """

    r0 = np.asarray(initial_position_m, dtype=float)
    v0 = np.asarray(initial_velocity_world_m_s, dtype=float)
    if v0.shape != (3,) or np.linalg.norm(v0) < 1e-12:
        raise ValueError("initial_velocity_world_m_s must be a non-zero 3-D vector")

    ex0 = _normalize(initial_direction)
    omega0 = (
        np.zeros(3)
        if initial_angular_velocity_body_rad_s is None
        else np.asarray(initial_angular_velocity_body_rad_s, dtype=float)
    )
    if omega0.shape != (3,):
        raise ValueError("initial_angular_velocity_body_rad_s must have 3 components")

    wind = (
        np.zeros(3)
        if wind_world_m_s is None
        else np.asarray(wind_world_m_s, dtype=float)
    )

    q0, R0 = _initial_quaternion_from_inputs(
        ex0, initial_roll_deg, initial_orientation_quaternion_xyzw
    )

    qb0 = np.asarray(initial_bending_m, dtype=float)
    qbd0 = np.asarray(initial_bending_velocity_m_s, dtype=float)
    if qb0.shape != (2,) or qbd0.shape != (2,):
        raise ValueError("initial_bending_m and initial_bending_velocity_m_s must be [y,z]")

    # State:
    # r(3), v(3), quaternion(4), omega_body(3),
    # q_y, qdot_y, q_z, qdot_z
    y0 = np.r_[
        r0, v0, q0, omega0,
        qb0[0], qbd0[0], qb0[1], qbd0[1]
    ]

    rho = atmosphere.rho

    def rhs(t, y):
        r = y[:3]
        v = y[3:6]
        quat = y[6:10] / np.linalg.norm(y[6:10])
        omega = y[10:13]
        qb_y, qbd_y, qb_z, qbd_z = y[13:17]

        Rbw = Rotation.from_quat(quat).as_matrix()
        ex, ey, ez = Rbw[:, 0], Rbw[:, 1], Rbw[:, 2]

        F = np.array([0.0, -arrow.total_mass_kg * G, 0.0])
        M_body = np.zeros(3)
        Qb_y = 0.0
        Qb_z = 0.0

        # ---- Shaft aerodynamic load ----
        # Integrate over multiple segments so drag acts at the actual
        # aerodynamic locations rather than all at the CM.
        nseg = arrow.shaft_drag_segments
        edges = np.linspace(0.0, arrow.length_m, nseg + 1)

        for xa, xb in zip(edges[:-1], edges[1:]):
            xmid = 0.5 * (xa + xb)
            ds = xb - xa
            phi = float(arrow.phi(xmid))
            phi_p = float(arrow.phi_prime(xmid))

            rb = np.array(
                [xmid - arrow.x_cm, qb_y * phi, qb_z * phi]
            )
            tangent_b = _normalize(
                np.array([1.0, qb_y * phi_p, qb_z * phi_p])
            )

            rb_w = Rbw @ rb
            tangent_w = _normalize(Rbw @ tangent_b)
            v_seg = v + Rbw @ np.cross(omega, rb)

            Fseg = shaft_segment_force(
                v_seg - wind,
                tangent_w,
                rho,
                ds,
                arrow.shaft_outer_d_m,
                arrow.shaft_cd,
                arrow.shaft_axial_cd,
            )

            F += Fseg
            M_body += np.cross(rb, Rbw.T @ Fseg)

            # Virtual-work generalized force for each bending coordinate.
            Qb_y += np.dot(Rbw.T @ Fseg, np.array([0.0, phi, 0.0]))
            Qb_z += np.dot(Rbw.T @ Fseg, np.array([0.0, 0.0, phi]))

        # ---- Point / pile / field-point aerodynamic load ----
        rp_b = np.array(
            [arrow.point_x_m - arrow.x_cm, 0.0, 0.0]
        )
        rp_w = Rbw @ rp_b
        vp = v + np.cross(Rbw @ omega, rp_w)
        Fp = aerodynamic_force(
            vp - wind,
            ex,
            rho,
            arrow.point_area_m2,
            arrow.point_cd,
        )
        F += Fp
        M_body += np.cross(rp_b, Rbw.T @ Fp)

        # ---- Feathers ----
        f = arrow.feathers
        xf = f.x_center_m
        phi_f = float(arrow.phi(xf))
        phi_fp = float(arrow.phi_prime(xf))

        rf_b_center = np.array(
            [xf - arrow.x_cm, qb_y * phi_f, qb_z * phi_f]
        )
        vf_base = v + Rbw @ np.cross(omega, rf_b_center)

        rel = vf_base - wind
        speed = np.linalg.norm(rel)

        if speed > 1e-12 and f.count > 0 and f.area_each_m2 > 0:
            flow = rel / speed
            qdyn = 0.5 * rho * speed**2
            cant = np.deg2rad(f.cant_deg)
            stall = np.deg2rad(f.stall_deg)

            for k in range(f.count):
                a = 2.0 * np.pi * k / f.count

                radial = np.cos(a) * ey + np.sin(a) * ez
                normal = (
                    np.cos(cant) * radial
                    + np.sin(cant) * np.cross(ex, radial)
                )
                normal = _normalize(normal)

                alpha = np.arcsin(
                    np.clip(np.dot(flow, normal), -1.0, 1.0)
                )
                alpha = np.clip(alpha, -stall, stall)

                Ff = (
                    -qdyn * f.cd * f.area_each_m2 * flow
                    + qdyn
                    * (f.lift_slope_per_rad * alpha)
                    * f.area_each_m2
                    * normal
                )

                # Approximate each feather force at its centroid.
                rf_b = rf_b_center
                F += Ff
                M_body += np.cross(rf_b, Rbw.T @ Ff)

                Qb_y += np.dot(
                    Rbw.T @ Ff, np.array([0.0, phi_f, 0.0])
                )
                Qb_z += np.dot(
                    Rbw.T @ Ff, np.array([0.0, 0.0, phi_f])
                )

        # ---- Flexible shaft equations ----
        qydd = (
            Qb_y - arrow.modal_c * qbd_y - arrow.modal_k * qb_y
        ) / arrow.modal_mass
        qzdd = (
            Qb_z - arrow.modal_c * qbd_z - arrow.modal_k * qb_z
        ) / arrow.modal_mass

        # ---- Rigid-body rotation ----
        I = np.diag([arrow.Ixx, arrow.Iyy, arrow.Izz])
        omega_dot = np.linalg.solve(
            I,
            M_body - np.cross(omega, I @ omega),
        )

        return np.r_[
            v,
            F / arrow.total_mass_kg,
            quat_derivative(quat, omega),
            omega_dot,
            qbd_y,
            qydd,
            qbd_z,
            qzdd,
        ]

    # ---- Events ----
    events = []
    names = []

    if stop_at_ground:
        def ground_event(t, y):
            # Stop at the ground based on the actual arrow tip, not CM.
            return _tip_state(arrow, y)[0][1] - ground_y_m

        ground_event.terminal = True
        ground_event.direction = -1
        events.append(ground_event)
        names.append("ground_tip")

    if target is not None:
        def target_x_event(t, y):
            return _tip_state(arrow, y)[0][0] - target.distance_m

        # The requested target distance defines the simulation endpoint.
        # Height/Z are evaluated at that exact tip-X crossing.
        target_x_event.terminal = True
        target_x_event.direction = 1
        events.append(target_x_event)
        names.append("target_x")

    # Integrate all checkpoint crossings as non-terminal events.  This is
    # deliberately done separately because solve_ivp's event arrays are
    # otherwise awkward to associate with individual checkpoint labels.
    # Checkpoints may be supplied as:
    #   x
    #   (x, y, z)  -> relative vector is tip - [x,y,z]
    #   {"x_m": x, "y_m": y, "z_m": z}
    # Only X determines the crossing event.
    checkpoint_specs = []
    for cp in ([] if checkpoints is None else checkpoints):
        if np.isscalar(cp):
            checkpoint_specs.append(np.array([float(cp), 0.0, 0.0]))
        elif isinstance(cp, dict):
            checkpoint_specs.append(
                np.array([
                    float(cp["x_m"]),
                    float(cp.get("y_m", 0.0)),
                    float(cp.get("z_m", 0.0)),
                ])
            )
        else:
            a = np.asarray(cp, dtype=float)
            if a.shape != (3,):
                raise ValueError("Each checkpoint must be x, [x,y,z], or a dict")
            checkpoint_specs.append(a)
    checkpoint_values = [float(cp[0]) for cp in checkpoint_specs]

    checkpoint_names_local = (
        list(checkpoint_names)
        if checkpoint_names is not None
        else [f"checkpoint_{i+1}" for i in range(len(checkpoint_values))]
    )
    if len(checkpoint_names_local) != len(checkpoint_values):
        raise ValueError("checkpoint_names must have the same length as checkpoints")

    checkpoint_events = []
    for xcp in checkpoint_values:
        def make_cp_event(x_value):
            def cp_event(t, y):
                return _tip_state(arrow, y)[0][0] - x_value
            cp_event.terminal = False
            cp_event.direction = 1
            return cp_event

        checkpoint_events.append(make_cp_event(xcp))

    all_events = events + checkpoint_events

    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        y0,
        max_step=dt,
        rtol=2e-8,
        atol=1e-10,
        events=all_events if all_events else None,
        dense_output=True,
    )

    reason = "time_limit"
    hit_index = None
    if events:
        hit_time = np.inf
        for i, ts in enumerate(sol.t_events[:len(events)]):
            if len(ts) and ts[0] < hit_time:
                hit_time = ts[0]
                hit_index = i
        if hit_index is not None:
            reason = names[hit_index]

    yf = sol.y[:, -1]
    qf = yf[6:10] / np.linalg.norm(yf[6:10])
    Rf = Rotation.from_quat(qf).as_matrix()

    tip_pos, tip_vel, direction = _tip_state(arrow, yf)

    # Target result: interpolate exactly at target X crossing when available.
    target_error = None
    target_hit = False
    if target is not None:
        # Find first target_x event if it exists.
        target_event_idx = names.index("target_x") if "target_x" in names else None
        if target_event_idx is not None and len(sol.t_events[target_event_idx]):
            tt = sol.t_events[target_event_idx][0]
            yt = sol.sol(tt)
            tip_t, _, _ = _tip_state(arrow, yt)
            if target.z_m is None:
                target_error = np.array(
                    [tip_t[0] - target.distance_m,
                     tip_t[1] - target.height_m]
                )
            else:
                target_error = (
                    tip_t
                    - np.array(
                        [target.distance_m, target.height_m, target.z_m]
                    )
                )
            target_hit = (
                np.linalg.norm(target_error) <= (
                    target.radius_m
                    if target.radius_m is not None
                    else target_radius_m
                )
            )

    # Checkpoint results, calculated at the solver's interpolated crossing
    # state so the reported X is not quantized by dt.
    crossings = []
    cp_offset = len(events)
    for i, xcp in enumerate(checkpoint_values):
        ts = sol.t_events[cp_offset + i]
        if not len(ts):
            continue

        for j, tc in enumerate(ts):
            yc = sol.sol(tc)
            tip_c, vel_c, dir_c = _tip_state(arrow, yc)
            checkpoint_point = checkpoint_specs[i]
            crossings.append(
                CheckpointCrossing(
                    name=checkpoint_names_local[i],
                    x_m=xcp,
                    time_s=float(tc),
                    tip_position_world_m=tip_c.copy(),
                    cm_position_world_m=yc[:3].copy(),
                    tip_minus_checkpoint_m=tip_c - checkpoint_point,
                    velocity_tip_world_m_s=vel_c.copy(),
                    arrow_direction_world=dir_c.copy(),
                )
            )

    return FlightResult(
        reason=reason,
        time_s=float(sol.t[-1]),
        position_cm_world_m=yf[:3].copy(),
        position_tip_world_m=tip_pos.copy(),
        direction_world=direction.copy(),
        orientation_quaternion_xyzw=qf.copy(),
        velocity_cm_world_m_s=yf[3:6].copy(),
        velocity_tip_world_m_s=tip_vel.copy(),
        angular_velocity_body_rad_s=yf[10:13].copy(),
        bending_amplitude_y_m=float(yf[13]),
        bending_amplitude_z_m=float(yf[15]),
        target_error_m=target_error,
        target_hit=target_hit,
        checkpoint_crossings=crossings,
        solution=sol,
    )


def mm(values):
    """Round scalar/vector values to millimetres for presentation."""
    return np.round(np.asarray(values, dtype=float), 3)


if __name__ == "__main__":
    arrow = Arrow(
        length_m=0.75,
        shaft_outer_d_m=0.0065,
        shaft_inner_d_m=0.0045,
        total_mass_kg=0.028,
        point_mass_kg=0.009,
        point_x_m=0.75,
        spine=1000,
        feathers=Feather(
            x_start_m=0.08,
            x_end_m=0.19,
            height_m=0.012,
            area_each_m2=0.00075,
            count=3,
            cant_deg=2.0,
            mass_kg=0.0005,
        ),
    )

    atmosphere = Atmosphere()
    print("Air density:", atmosphere.rho, "kg/m^3")
    print("Spine:", arrow.spine)
    print("Static deflection:", arrow.spine_deflection_m, "m")
    print("Effective EI:", arrow.EI, "N*m^2")
    print("Center of mass from nock:", arrow.x_cm, "m")

    # Example with the requested independent velocity and shaft direction.
    # World coordinates are [x forward, y up, z sideways].
    result = simulate(
        arrow,
        atmosphere,
        initial_position_m=[0.0, 1.5, 0.0],
        initial_velocity_world_m_s=[70.0, 2.0, 0.5],
        initial_direction=[1.0, 0.02, -0.01],
        initial_roll_deg=15.0,
        initial_angular_velocity_body_rad_s=[0.2, -0.1, 80.0],
        target=Target(distance_m=20.0, height_m=1.25, z_m=0.0),
        checkpoints=[5.0, 10.0, 15.0, 20.0],
        checkpoint_names=["5 m", "10 m", "15 m", "20 m"],
        ground_y_m=0.0,
    )

    print("\n--- RESULT ---")
    print("Reason:", result.reason, "time:", result.time_s)
    print("Final CM [m]:", mm(result.position_cm_world_m))
    print("Final tip [m]:", mm(result.position_tip_world_m))
    print("Final direction:", np.round(result.direction_world, 6))
    print("Final CM velocity [m/s]:", np.round(result.velocity_cm_world_m_s, 6))
    print("Final tip velocity [m/s]:", np.round(result.velocity_tip_world_m_s, 6))
    print(
        "Final angular velocity [rad/s]:",
        np.round(result.angular_velocity_body_rad_s, 6),
    )

    if result.target_error_m is not None:
        print("Target error [m]:", mm(result.target_error_m))
        print("Target hit:", result.target_hit)

    print("\n--- CHECKPOINTS ---")
    for cp in result.checkpoint_crossings:
        print(
            cp.name,
            "t=", round(cp.time_s, 6),
            "tip=", mm(cp.tip_position_world_m),
            "tip-checkpoint=", mm(cp.tip_minus_checkpoint_m),
            "tip velocity=", np.round(cp.velocity_tip_world_m_s, 6),
        )
