
from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

G = 9.80665
R_AIR = 287.05


@dataclass
class Atmosphere:
    pressure_pa: float = 101325.0
    temperature_k: float = 288.15

    @property
    def rho(self):
        return self.pressure_pa / (R_AIR * self.temperature_k)


@dataclass
class Feather:
    # Position measured from the nock/rear of the arrow.
    x_start_m: float = 0.05 #my value 0.07
    x_end_m: float = 0.2 #my value0.13

    # Geometry / aerodynamic properties
    height_m: float = 0.012
    area_each_m2: float = 0.0008#my value 0.0004
    count: int = 3
    cant_deg: float = 0.0

    # Mass and approximate aerodynamic coefficients
    mass_kg: float = 0.0005
    cd: float = 1.2
    lift_slope_per_rad: float = 3.0
    stall_deg: float = 18.0

    @property
    def x_center_m(self):
        return 0.5 * (self.x_start_m + self.x_end_m)


@dataclass
class Arrow:
    # Basic physical properties
    length_m: float
    shaft_outer_d_m: float
    shaft_inner_d_m: float
    total_mass_kg: float

    # Point mass and location
    point_mass_kg: float
    point_x_m: float

    # Shaft material
    youngs_modulus_pa: float = 70e9

    # Aerodynamics
    shaft_cd: float = 1.0
    point_cd: float = 0.5
    point_area_m2: float = 0.0001

    feathers: Feather = field(default_factory=Feather)

    def __post_init__(self):
        if self.length_m <= 0:
            raise ValueError("length_m must be positive")
        if self.shaft_outer_d_m <= 0:
            raise ValueError("shaft_outer_d_m must be positive")
        if self.shaft_inner_d_m < 0 or self.shaft_inner_d_m >= self.shaft_outer_d_m:
            raise ValueError("shaft_inner_d_m must be >= 0 and smaller than outer diameter")
        if self.point_mass_kg < 0:
            raise ValueError("point_mass_kg cannot be negative")
        if not 0 <= self.point_x_m <= self.length_m:
            raise ValueError("point_x_m must lie between 0 and length_m")

        # Shaft cross-sectional area and second moment of area.
        self.A = np.pi / 4 * (
            self.shaft_outer_d_m**2 - self.shaft_inner_d_m**2
        )
        self.I = np.pi / 64 * (
            self.shaft_outer_d_m**4 - self.shaft_inner_d_m**4
        )

        # Bending stiffness EI.
        self.EI = self.youngs_modulus_pa * self.I

        # Remaining mass belongs to shaft after point and feathers.
        self.shaft_mass = (
            self.total_mass_kg
            - self.point_mass_kg
            - self.feathers.mass_kg
        )
        if self.shaft_mass <= 0:
            raise ValueError(
                "total_mass_kg must exceed point_mass_kg + feather mass"
            )

        # First cantilever bending eigenvalue.
        self.lambda_b = 1.875104068711961

        # Center of mass measured from the nock.
        m = self.shaft_mass
        L = self.length_m
        mp = self.point_mass_kg
        mf = self.feathers.mass_kg

        self.x_cm = (
            m * L / 2
            + mp * self.point_x_m
            + mf * self.feathers.x_center_m
        ) / self.total_mass_kg

        # Approximate principal moments of inertia.
        self.Iyy = (
            m * (L**2 / 12 + (L / 2 - self.x_cm)**2)
            + mp * (self.point_x_m - self.x_cm)**2
            + mf * (self.feathers.x_center_m - self.x_cm)**2
        )
        self.Izz = self.Iyy

        # Approximate polar inertia around shaft axis.
        radius = self.shaft_outer_d_m / 2
        self.Ixx = (
            0.5 * m * radius**2
            + 0.5 * mp * radius**2
        )

        # First bending mode approximation.
        # For a cantilever, modal mass is approximately 0.236*m.
        self.modal_mass = 0.236 * self.shaft_mass
        self.modal_k = (
            self.lambda_b**4 * self.EI / self.length_m**3
        ) * 0.236

        # ~2% modal damping. This is a tunable approximation.
        self.modal_c = (
            2 * 0.02 * np.sqrt(self.modal_k * self.modal_mass)
        )

    def _mode_raw(self, x):
        s = np.asarray(x) / self.length_m
        b = self.lambda_b
        sigma = (
            (np.cosh(b) + np.cos(b))
            / (np.sinh(b) + np.sin(b))
        )
        return (
            np.cosh(b * s)
            - np.cos(b * s)
            - sigma * (
                np.sinh(b * s)
                - np.sin(b * s)
            )
        )

    def phi(self, x):
        """First bending mode, normalized to 1 at the tip."""
        tip = self._mode_raw(self.length_m)
        return self._mode_raw(x) / tip


def quat_derivative(q, omega_body):
    """Quaternion derivative for scipy's [x,y,z,w] convention."""
    x, y, z, w = q
    wx, wy, wz = omega_body

    return 0.5 * np.array([
        w * wx + y * wz - z * wy,
        w * wy + z * wx - x * wz,
        w * wz + x * wy - y * wx,
        -x * wx - y * wy - z * wz,
    ])


def aero_force(
    v_rel_world,
    axis_world,
    rho,
    area,
    cd,
    lift_slope=0.0,
    stall_rad=np.deg2rad(18.0),
):
    """
    Simple aerodynamic force model.

    v_rel_world:
        Velocity of the aerodynamic element relative to the air.

    axis_world:
        Local longitudinal/chord direction.

    Returns:
        Force in world coordinates.
    """
    speed = np.linalg.norm(v_rel_world)
    if speed < 1e-9:
        return np.zeros(3)

    flow = v_rel_world / speed
    dynamic_pressure = 0.5 * rho * speed**2

    # Direction perpendicular to flow and lying in the
    # axis/flow plane.
    n = axis_world - np.dot(axis_world, flow) * flow
    nn = np.linalg.norm(n)

    drag = -dynamic_pressure * cd * area * flow

    if nn < 1e-10 or lift_slope == 0:
        return drag

    n /= nn

    # Unsigned angle between axis and incoming flow.
    c = np.clip(np.dot(axis_world, -flow), -1.0, 1.0)
    alpha = np.arccos(c)

    # A simple signed approximation.
    signed_alpha = np.sign(np.dot(n, axis_world)) * alpha
    signed_alpha = np.clip(
        signed_alpha,
        -stall_rad,
        stall_rad,
    )

    cl = lift_slope * signed_alpha
    lift = dynamic_pressure * cl * area * n

    return drag + lift


def simulate(
    arrow,
    atmosphere,
    r0,
    v0,
    q0=None,
    omega0=None,
    wind_world=None,
    t_end=1.5,
    dt=0.001,
):
    """
    Simulate arrow flight.

    State:
        r(3)             world position
        v(3)             world velocity
        quaternion(4)    body -> world orientation
        omega(3)         body angular velocity
        qb               first bending-mode amplitude
        qbd              first bending-mode velocity

    q0 uses scipy Rotation convention [x, y, z, w].
    """

    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    if q0 is None:
        # Arrow body +X initially aligned with world +X.
        q0 = np.array([0., 0., 0., 1.])

    q0 = np.asarray(q0, dtype=float)
    q0 /= np.linalg.norm(q0)

    if omega0 is None:
        omega0 = np.zeros(3)

    omega0 = np.asarray(omega0, dtype=float)

    if wind_world is None:
        wind_world = np.zeros(3)

    wind_world = np.asarray(wind_world, dtype=float)

    # r, v, quaternion, omega, bending amplitude, bending velocity
    y0 = np.r_[r0, v0, q0, omega0, 0., 0.]

    L = arrow.length_m
    rho = atmosphere.rho

    # Projected area approximation for cylindrical shaft.
    shaft_area = arrow.shaft_outer_d_m * L

    def rhs(t, y):
        r = y[0:3]
        v = y[3:6]
        quat = y[6:10]
        omega = y[10:13]
        qb = y[13]
        qbd = y[14]

        quat = quat / np.linalg.norm(quat)
        Rbw = Rotation.from_quat(quat).as_matrix()

        # Body axes in world coordinates.
        ex = Rbw[:, 0]
        ey = Rbw[:, 1]
        ez = Rbw[:, 2]

        # Gravity.
        F = np.array([
            0.,
            0.,
            -arrow.total_mass_kg * G,
        ])

        M_body = np.zeros(3)

        # Relative air velocity.
        v_air = v - wind_world

        # Shaft drag.
        F_shaft = aero_force(
            v_air,
            ex,
            rho,
            shaft_area,
            arrow.shaft_cd,
        )
        F += F_shaft

        # Point aerodynamic force.
        r_point_body = np.array([
            arrow.point_x_m - arrow.x_cm,
            0.,
            0.,
        ])
        r_point_world = Rbw @ r_point_body

        v_point = (
            v
            + np.cross(
                Rbw @ omega,
                r_point_world,
            )
        )

        F_point = aero_force(
            v_point - wind_world,
            ex,
            rho,
            arrow.point_area_m2,
            arrow.point_cd,
        )

        F += F_point
        M_body += np.cross(
            r_point_body,
            Rbw.T @ F_point,
        )

        # Feather aerodynamics.
        f = arrow.feathers
        xf = f.x_center_m
        phi_f = arrow.phi(xf)

        # Local bending displacement in body Y.
        y_f = qb * phi_f
        ydot_f = qbd * phi_f

        r_f_body = np.array([
            xf - arrow.x_cm,
            y_f,
            0.,
        ])

        v_f = (
            v
            + Rbw @ (
                np.cross(omega, r_f_body)
                + np.array([0., ydot_f, 0.])
            )
        )

        speed = np.linalg.norm(v_f - wind_world)

        if speed > 1e-8:
            flow = (v_f - wind_world) / speed
            qdyn = 0.5 * rho * speed**2

            cant = np.deg2rad(f.cant_deg)

            for k in range(f.count):
                angle = 2 * np.pi * k / f.count

                radial = (
                    np.cos(angle) * ey
                    + np.sin(angle) * ez
                )

                # Canting changes the aerodynamic-force direction
                # and can produce roll/stabilizing moments.
                normal = (
                    np.cos(cant) * radial
                    + np.sin(cant)
                    * np.cross(ex, radial)
                )

                alpha = np.arcsin(
                    np.clip(
                        np.dot(flow, normal),
                        -1.,
                        1.,
                    )
                )

                alpha = np.clip(
                    alpha,
                    -np.deg2rad(f.stall_deg),
                    np.deg2rad(f.stall_deg),
                )

                cl = f.lift_slope_per_rad * alpha

                F_feather = (
                    -qdyn * f.cd
                    * f.area_each_m2
                    * flow
                    + qdyn * cl
                    * f.area_each_m2
                    * normal
                )

                F += F_feather

                M_body += np.cross(
                    r_f_body,
                    Rbw.T @ F_feather,
                )

        # First bending-mode equation.
        #
        # This is deliberately a reduced-order flexible-body model.
        # It captures shaft flex qualitatively and lets stiffness affect
        # the flight, but it is not a full finite-element beam solver.
        F_body = Rbw.T @ F
        Qb = F_body[1] * arrow.phi(L)

        qbdd = (
            Qb
            - arrow.modal_c * qbd
            - arrow.modal_k * qb
        ) / arrow.modal_mass

        # Rigid-body rotational dynamics.
        I = np.diag([
            arrow.Ixx,
            arrow.Iyy,
            arrow.Izz,
        ])

        omega_dot = np.linalg.solve(
            I,
            M_body - np.cross(
                omega,
                I @ omega,
            ),
        )

        r_dot = v
        v_dot = F / arrow.total_mass_kg
        q_dot = quat_derivative(
            quat,
            omega,
        )

        return np.r_[
            r_dot,
            v_dot,
            q_dot,
            omega_dot,
            qbd,
            qbdd,
        ]

    return solve_ivp(
        rhs,
        (0, t_end),
        y0,
        max_step=dt,
        rtol=2e-7,
        atol=1e-9,
        dense_output=True,
    )


if __name__ == "__main__":
    # Example arrow.
    arrow = Arrow(
        length_m=0.75,
        shaft_outer_d_m=0.0065,
        shaft_inner_d_m=0.0045,

        total_mass_kg=0.028,

        point_mass_kg=0.009,
        point_x_m=0.75,

        # Carbon-like example value. Change for your actual shaft.
        youngs_modulus_pa=70e9,

        shaft_cd=1.0,
        point_cd=0.5,
        point_area_m2=0.0001,

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

    atmosphere = Atmosphere(
        pressure_pa=101325.0,
        temperature_k=288.15,
    )

    # Initial position and velocity.
    sol = simulate(
        arrow,
        atmosphere,
        r0=[0., 0., 1.5],
        v0=[70., 0., 0.],
        t_end=1.0,
    )

    print("Air density:", atmosphere.rho, "kg/m^3")
    print("Center of mass:", arrow.x_cm, "m")
    print("Bending stiffness EI:", arrow.EI, "N*m^2")
    print("Final position:", sol.y[:3, -1], "m")
    print("Final speed:", np.linalg.norm(sol.y[3:6, -1]), "m/s")
