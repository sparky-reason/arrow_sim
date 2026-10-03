"""
Flexible bow-launch model feeding arrow_flight_sim_updated.simulate().

This version uses a discretised Euler-Bernoulli beam for the arrow and an
active-set unilateral contact constraint for the bow/grip.  The contact is
not represented by a penetration spring: at each time step the active contact
nodes are solved as constrained normal accelerations (with a small Baumgarte
velocity correction), while inactive nodes carry no contact force.

The model is deliberately explicit about what is and is not identified by the
three primary bow inputs.  Bow length, draw weight and draw length do not by
themselves determine the dynamic force-draw curve, limb inertia, string path,
or riser geometry.  Those are parameterised below and should be measured when
high fidelity is required.

Coordinate convention inherited from the flight simulator:
    world +X = range / nock -> point
    world +Y = up
    world +Z = lateral

The arrow is launched from a traditional side-shot geometry by default.  The
new ``feather_clocking_deg`` input specifies the angular orientation of the
fletching around the shaft at release.  It becomes the launch roll angle, so
the flight simulator sees the same feather orientation immediately after the
bow phase.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
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


def _smoothstep01(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


@dataclass
class Bow:
    """Parametric longbow/recurve model.

    Primary inputs, common to both bow types:
        length_m, draw_strength_kgf, draw_length_m

    ``draw_strength_kgf`` is the force at the supplied full draw.  The
    normalised force-draw shape is an engineering approximation unless
    ``force_draw_exponent`` is calibrated to measured data.
    """

    length_m: float
    draw_strength_kgf: float
    draw_length_m: float
    bow_type: str = "longbow"

    brace_height_m: float = 0.16
    handle_length_m: float = 0.22
    handle_thickness_m: float = 0.035
    handle_height_m: float = 0.10

    limb_mass_kg: float = 0.08
    string_mass_kg: float = 0.010
    bow_efficiency: float = 0.78
    nock_efficiency: float = 0.995

    # None = built-in generic curve; otherwise use this exponent.
    force_draw_exponent: Optional[float] = None

    # Position of the arrow centreline from the bow median plane.  +Z is
    # toward the arrow and therefore away from the traditional grip surface.
    arrow_side_offset_m: Optional[float] = None
    nock_height_offset_m: float = 0.0
    aim_elevation_deg: float = 0.0

    # Mediterranean release / string departure disturbance.
    release_lateral_velocity_m_s: float = 0.0
    release_lateral_displacement_m: float = 0.0
    release_vertical_velocity_m_s: float = 0.0
    release_duration_s: float = 0.0015

    # New: angular clocking of the feathers about the shaft at launch.
    # 0 deg is the reference orientation used by orientation_from_direction...
    feather_clocking_deg: float = 0.0

    def __post_init__(self):
        self.bow_type = self.bow_type.lower()
        if self.bow_type not in {"longbow", "recurve"}:
            raise ValueError("bow_type must be 'longbow' or 'recurve'")
        if min(self.length_m, self.draw_length_m) <= 0:
            raise ValueError("bow length and draw length must be positive")
        if self.draw_strength_kgf <= 0:
            raise ValueError("draw_strength_kgf must be positive")
        if not (0.0 < self.brace_height_m < self.draw_length_m):
            raise ValueError("brace_height_m must be between 0 and draw length")
        if not (0.0 < self.bow_efficiency <= 1.0):
            raise ValueError("bow_efficiency must be in (0,1]")
        if not (0.0 < self.nock_efficiency <= 1.0):
            raise ValueError("nock_efficiency must be in (0,1]")
        if self.release_duration_s <= 0:
            raise ValueError("release_duration_s must be positive")
        if self.arrow_side_offset_m is None:
            self.arrow_side_offset_m = 0.018 if self.bow_type == "longbow" else 0.004
        if self.arrow_side_offset_m < 0:
            raise ValueError("arrow_side_offset_m must be non-negative")
        if self.force_draw_exponent is not None and self.force_draw_exponent <= 0:
            raise ValueError("force_draw_exponent must be positive")

    @property
    def draw_force_N(self):
        return self.draw_strength_kgf * G

    @property
    def draw_coordinate_m(self):
        return self.draw_length_m - self.brace_height_m

    @property
    def curve_exponent(self):
        if self.force_draw_exponent is not None:
            return self.force_draw_exponent
        # Generic curves only.  These are not a substitute for a measured
        # force-draw curve of the actual bow.
        return 1.00 if self.bow_type == "longbow" else 0.82

    def draw_force(self, displacement_m):
        d = np.clip(displacement_m, 0.0, self.draw_coordinate_m)
        u = d / self.draw_coordinate_m
        return self.draw_force_N * u ** self.curve_exponent

    def stored_energy_J(self):
        d = np.linspace(0.0, self.draw_coordinate_m, 4000)
        return float(np.trapezoid(self.draw_force(d), d) * self.bow_efficiency)


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
    bow_hit: bool
    bow_contact_time_s: float
    max_contact_force_N: float
    contact_impulse_Ns: float
    bow_energy_J: float
    launch_solution: object = None
    flight_result: Optional[FlightResult] = None


class BowLaunchSimulator:
    """Discretised flexible-arrow launch solver with unilateral grip contact.

    The shaft is represented by transverse displacement nodes.  Bending energy
    is obtained from the discrete curvature (second derivative), giving a
    free-free beam: no artificial clamping is imposed at either end.

    Grip contact is solved as a unilateral constraint:
        gap >= 0, normal_reaction >= 0, gap*normal_reaction = 0.
    At an active contact node the normal acceleration is constrained to zero
    (with a small Baumgarte correction for numerical drift).  There is no
    penetration spring and no arbitrary "hit" penetration threshold.
    """

    def __init__(self, arrow: Arrow, bow: Bow, nodes: int = 31):
        if nodes < 9 or nodes % 2 == 0:
            raise ValueError("nodes must be an odd integer >= 9")
        self.arrow = arrow
        self.bow = bow
        self.n = int(nodes)
        self.x = np.linspace(0.0, arrow.length_m, self.n)
        self.dx = self.x[1] - self.x[0]

        elev = np.deg2rad(bow.aim_elevation_deg)
        self.ex = _norm(np.array([np.cos(elev), np.sin(elev), 0.0]))
        self.ey = _norm(np.array([-np.sin(elev), np.cos(elev), 0.0]))
        self.ez = np.array([0.0, 0.0, 1.0])
        self.R0 = np.column_stack([self.ex, self.ey, self.ez])

        # Discrete curvature matrix.  The first/last rows use one-sided
        # second differences.  Its null space contains constant and linear
        # transverse displacement, i.e. the rigid translation/rotation modes
        # expected from a free-free beam.
        self.D2 = self._second_difference_matrix()
        self.K = arrow.EI * self.dx * (self.D2.T @ self.D2)

        # Lumped physical mass.  Point and fletching masses are placed at the
        # nearest material node.  This is particularly convenient for the
        # unilateral contact solve because the normal mass matrix is diagonal.
        self.mass = np.full(self.n, arrow.shaft_mass / self.n)
        self.mass[0] *= 0.5
        self.mass[-1] *= 0.5
        for x_m, m in ((arrow.point_x_m, arrow.point_mass_kg),
                       (arrow.feathers.x_center_m, arrow.feathers.mass_kg)):
            j = int(np.argmin(abs(self.x - x_m)))
            self.mass[j] += m

        # Light modal damping, estimated from the first two non-rigid discrete
        # beam frequencies.  Rigid translation/rotation modes are excluded.
        evals = np.linalg.eigvalsh(self.K / np.sqrt(np.outer(self.mass, self.mass)))
        evals = evals[evals > 1e-6]
        if len(evals) >= 2:
            w1, w2 = np.sqrt(evals[:2])
            zeta = 0.015
            beta = 2*zeta/(w1+w2)
            alpha = 2*zeta*w1*w2/(w1+w2)
        elif len(evals) == 1:
            alpha, beta = 0.0, 2*0.015/np.sqrt(evals[0])
        else:
            alpha = beta = 0.0
        self.C_diag = alpha*self.mass
        self.C_stiff = beta

        self.contact_gap_tol_m = 2.0e-8
        self.baumgarte_omega = 2.0*np.pi*3000.0
        self.baumgarte_zeta = 0.9

        self.contact_half_x = 0.5*bow.handle_length_m
        self.contact_half_y = 0.5*bow.handle_height_m
        if bow.arrow_side_offset_m < 0.5*bow.handle_thickness_m:
            raise ValueError(
                "initial arrow_side_offset_m places the shaft centreline inside "
                "the grip surface; supply a non-penetrating initial geometry"
            )

    def _second_difference_matrix(self):
        n = self.n
        h = self.dx
        D = np.zeros((n, n))
        if n >= 3:
            # Interior central second derivative.
            for i in range(1, n-1):
                D[i, i-1] = 1.0/h**2
                D[i, i] = -2.0/h**2
                D[i, i+1] = 1.0/h**2
            # Second-order one-sided approximations at the ends.
            D[0, 0] = 1.0/h**2
            D[0, 1] = -2.0/h**2
            D[0, 2] = 1.0/h**2
            D[-1, -1] = 1.0/h**2
            D[-1, -2] = -2.0/h**2
            D[-1, -3] = 1.0/h**2
        return D

    def _beam_state_to_world(self, uy, uz, duy, duz, axial_x, axial_v):
        p = np.empty((self.n, 3))
        v = np.empty((self.n, 3))
        for i, xi in enumerate(self.x):
            p[i] = (axial_x + xi)*self.ex + uy[i]*self.ey + \
                   (self.bow.arrow_side_offset_m + uz[i])*self.ez
            v[i] = axial_v*self.ex + duy[i]*self.ey + duz[i]*self.ez
        return p, v

    def _draw_displacement(self, nock_x):
        return np.clip(-nock_x, 0.0, self.bow.draw_coordinate_m)

    def _string_force(self, t, nock_p, nock_v):
        d = self._draw_displacement(float(np.dot(nock_p, self.ex)))
        fmag = self.bow.draw_force(d)
        target = (self.bow.nock_height_offset_m*self.ey +
                  self.bow.arrow_side_offset_m*self.ez)
        if t < self.bow.release_duration_s:
            u = t/self.bow.release_duration_s
            s = 1.0 - _smoothstep01(u)
            target += s*self.bow.release_lateral_displacement_m*self.ez
        target_world = target
        delta = target_world - nock_p
        F = np.zeros(3) if np.linalg.norm(delta) < 1e-12 else fmag*_norm(delta)

        if t < self.bow.release_duration_s:
            u = t/self.bow.release_duration_s
            s = 1.0 - _smoothstep01(u)
            target_v = s*(self.bow.release_vertical_velocity_m_s*self.ey +
                          self.bow.release_lateral_velocity_m_s*self.ez)
            F += 0.003*self.bow.draw_force_N*(target_v-nock_v)

        # This launch model does not explicitly integrate bow-limb/string
        # inertia.  Therefore the user-supplied bow efficiency is represented
        # here as the fraction of force-draw work delivered to the arrow.  If
        # limb dynamics are added later, remove this factor and let the limb
        # model determine the dynamic string force.
        return self.bow.nock_efficiency*self.bow.bow_efficiency*F

    def _contact_candidates(self, uy, uz, axial_x):
        candidates=[]
        gaps=[]
        for i, xi in enumerate(self.x):
            px = axial_x + xi
            py = uy[i]
            if abs(px) <= self.contact_half_x and abs(py) <= self.contact_half_y:
                # +Z is the free side of the grip.  The grip surface is at
                # z = +half_thickness; the arrow starts farther +Z.
                gap = (self.bow.arrow_side_offset_m + uz[i] -
                       0.5*self.bow.handle_thickness_m)
                candidates.append(i)
                gaps.append(gap)
        return np.asarray(candidates,dtype=int), np.asarray(gaps,dtype=float)

    def _unilateral_contact(self, az_free, duz, candidates, gaps):
        if len(candidates)==0:
            return az_free, np.zeros(0), np.zeros(0,dtype=int)

        # First choose nodes which are touching or would enter the grip on the
        # current step.  Reactions are then solved simultaneously.
        active=[int(i) for i,g in zip(candidates,gaps)
                if g <= self.contact_gap_tol_m]
        if not active:
            return az_free, np.zeros(0), np.zeros(0,dtype=int)

        # Diagonal mass => exact acceleration-level contact solve per active
        # node.  The active-set loop enforces lambda >= 0.
        while active:
            lam=[]
            for i in active:
                gi=float(gaps[np.where(candidates==i)[0][0]])
                target_a = (-2*self.baumgarte_zeta*self.baumgarte_omega*duz[i]
                            - self.baumgarte_omega**2*min(gi,0.0))
                lam.append(self.mass[i]*(target_a-az_free[i]))
            lam=np.asarray(lam)
            if np.all(lam >= -1e-10):
                az=az_free.copy()
                for i,L in zip(active,lam):
                    az[i]+=L/self.mass[i]
                return az,lam,np.asarray(active,dtype=int)
            active.pop(int(np.argmin(lam)))

        return az_free,np.zeros(0),np.zeros(0,dtype=int)

    def _rhs(self,t,y):
        n=self.n
        uy,duy,uz,duz=y[:n],y[n:2*n],y[2*n:3*n],y[3*n:4*n]
        axial_x,axial_v=y[4*n:4*n+2]
        p,v=self._beam_state_to_world(uy,uz,duy,duz,axial_x,axial_v)
        Fs=self._string_force(t,p[0],v[0])

        g=np.array([0.0,-G,0.0])
        gy=float(np.dot(g,self.ey))
        gz=float(np.dot(g,self.ez))
        gx=float(np.dot(g,self.ex))

        fy=self.mass*gy
        fz=self.mass*gz
        fy[0]+=np.dot(Fs,self.ey)
        fz[0]+=np.dot(Fs,self.ez)

        ay=(fy-self.K@uy-self.C_diag*duy-self.C_stiff*(self.K@duy))/self.mass
        az_free=(fz-self.K@uz-self.C_diag*duz-self.C_stiff*(self.K@duz))/self.mass

        cand,gaps=self._contact_candidates(uy,uz,axial_x)
        az,_,_=self._unilateral_contact(az_free,duz,cand,gaps)

        ax=(np.dot(Fs,self.ex)+self.arrow.total_mass_kg*gx)/self.arrow.total_mass_kg
        return np.r_[duy,ay,duz,az,axial_v,ax]

    def _unpack(self,y):
        n=self.n
        return y[:n],y[n:2*n],y[2*n:3*n],y[3*n:4*n],y[4*n],y[4*n+1]

    def _nodal_weights(self):
        w=np.full(self.n,self.arrow.shaft_mass/self.n)
        for x_m,m in ((self.arrow.point_x_m,self.arrow.point_mass_kg),
                      (self.arrow.feathers.x_center_m,self.arrow.feathers.mass_kg)):
            w[int(np.argmin(abs(self.x-x_m)))]+=m
        return w

    def _project_launch_state(self,y):
        uy,duy,uz,duz,axial_x,axial_v=self._unpack(y)
        phi=np.asarray([self.arrow.phi(xi) for xi in self.x])

        # Remove rigid translation and rotation before projecting bending.  The
        # existing flight model carries only its first bending coordinate.
        A=np.column_stack([phi,np.ones(self.n),self.x-self.arrow.x_cm])
        qy=np.linalg.lstsq(A,uy,rcond=None)[0][0]
        qz=np.linalg.lstsq(A,uz,rcond=None)[0][0]
        qdy=np.linalg.lstsq(A,duy,rcond=None)[0][0]
        qdz=np.linalg.lstsq(A,duz,rcond=None)[0][0]

        centerline=np.column_stack([axial_x+self.x,uy,
                                    self.bow.arrow_side_offset_m+uz])
        direction_local=_norm(centerline[-1]-centerline[0])
        direction_world=_norm(self.R0@direction_local)

        weights=self._nodal_weights()
        rmean=np.average(centerline,axis=0,weights=weights)
        vel_local=np.column_stack([np.full(self.n,axial_v),duy,duz])
        vmean=np.average(vel_local,axis=0,weights=weights)
        rrel=centerline-rmean
        vrel=vel_local-vmean
        B=[]; b=[]
        for r,vel in zip(rrel,vrel):
            B.append(np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]], dtype=float))
            b.extend(vel)
        B=np.vstack(B)
        omega_local=np.linalg.lstsq(B,np.asarray(b),rcond=None)[0]
        omega_world=self.R0@omega_local

        q_exit=orientation_from_direction_and_roll(
            direction_world,self.bow.feather_clocking_deg)
        R_exit=Rotation.from_quat(q_exit).as_matrix()
        omega_body=R_exit.T@omega_world

        r_cm_world=self.R0@rmean
        v_cm_world=self.R0@vmean
        return r_cm_world,v_cm_world,direction_world,qy,qz,qdy,qdz,q_exit,omega_body

    def _contact_diagnostics(self,sol):
        min_clear=np.inf; min_t=0.0; max_force=0.0; impulse=0.0; contact_time=0.0
        last_t=None; last_force=0.0
        for t,y in zip(sol.t,sol.y.T):
            uy,duy,uz,duz,ax,av=self._unpack(y)
            cand,gaps=self._contact_candidates(uy,uz,ax)
            if len(gaps):
                j=int(np.argmin(gaps))
                if gaps[j]<min_clear:
                    min_clear=float(gaps[j]); min_t=float(t)
            az_free=(-self.K@uz-self.C_diag*duz-self.C_stiff*(self.K@duz))/self.mass
            _,lam,active=self._unilateral_contact(az_free,duz,cand,gaps)
            f=float(np.sum(lam)) if len(lam) else 0.0
            max_force=max(max_force,f)
            if f>0 and last_t is not None:
                contact_time+=t-last_t
            if last_t is not None:
                impulse+=0.5*(last_force+f)*(t-last_t)
            last_t=float(t); last_force=f
        return min_clear,min_t,max_force,contact_time,impulse

    def simulate_launch(self,t_end=0.04,max_step=2e-5):
        n=self.n
        zeros=np.zeros(n)
        y0=np.r_[zeros,zeros,zeros,zeros,-self.bow.draw_coordinate_m,0.0]

        def exit_event(t,y):
            _,_,_,_,ax,_=self._unpack(y)
            # The nock is the last shaft material point to cross the riser
            # plane; once it clears the grip envelope the arrow is fully out.
            return ax-self.contact_half_x
        exit_event.terminal=True; exit_event.direction=1

        sol=solve_ivp(self._rhs,(0.0,t_end),y0,method='RK45',max_step=max_step,
                      rtol=2e-6,atol=1e-9,events=exit_event,dense_output=True)
        y=sol.y[:,-1]
        exit_t=float(sol.t[-1])
        release_t=min(self.bow.release_duration_s,exit_t)
        (r_cm,v_cm,direction,qy,qz,qdy,qdz,q_exit,omega_body)=self._project_launch_state(y)
        min_clear,min_t,max_force,contact_time,impulse=self._contact_diagnostics(sol)

        return LaunchResult(
            reason='bow_exit' if len(sol.t_events[0]) else 'time_limit',
            time_s=exit_t,release_time_s=release_t,left_bow_time_s=exit_t,
            initial_position_m=r_cm.copy(),initial_velocity_world_m_s=v_cm.copy(),
            initial_direction_world=direction.copy(),
            initial_roll_deg=float(self.bow.feather_clocking_deg),
            initial_orientation_quaternion_xyzw=q_exit.copy(),
            initial_angular_velocity_body_rad_s=omega_body.copy(),
            initial_bending_m=np.array([qy,qz]),
            initial_bending_velocity_m_s=np.array([qdy,qdz]),
            minimum_clearance_m=float(min_clear),minimum_clearance_time_s=float(min_t),
            bow_hit=bool(max_force>0.0),bow_contact_time_s=float(contact_time),
            max_contact_force_N=float(max_force),contact_impulse_Ns=float(impulse),
            bow_energy_J=self.bow.stored_energy_J(),launch_solution=sol)

    def launch_and_fly(self,atmosphere:Atmosphere,**flight_kwargs):
        launch=self.simulate_launch()
        launch.flight_result=simulate(
            self.arrow,atmosphere,
            initial_position_m=launch.initial_position_m,
            initial_velocity_world_m_s=launch.initial_velocity_world_m_s,
            initial_direction=launch.initial_direction_world,
            initial_roll_deg=launch.initial_roll_deg,
            initial_orientation_quaternion_xyzw=launch.initial_orientation_quaternion_xyzw,
            initial_angular_velocity_body_rad_s=launch.initial_angular_velocity_body_rad_s,
            initial_bending_m=launch.initial_bending_m,
            initial_bending_velocity_m_s=launch.initial_bending_velocity_m_s,
            **flight_kwargs)
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
            x_start_m=0.08,
            x_end_m=0.19,
            height_m=0.012,
            area_each_m2=0.00075,
            count=3,
            cant_deg=2.0,
            mass_kg=0.0005,
        ),
    )

    bow = Longbow(
        length_m=1.85,
        draw_strength_kgf=20.0,
        draw_length_m=0.72,
        brace_height_m=0.16,
        arrow_side_offset_m=0.018,
        # Example: feathers clocked 20 degrees around the shaft at release.
        feather_clocking_deg=20.0,
    )

    sim = BowLaunchSimulator(arrow, bow, nodes=25)
    launch = sim.simulate_launch()

    print("--- FLEXIBLE BEAM BOW LAUNCH ---")
    print("Reason:", launch.reason)
    print("Exit time [ms]:", round(1000 * launch.left_bow_time_s, 3))
    print("CM velocity [m/s]:", np.round(launch.initial_velocity_world_m_s, 4))
    print("Direction:", np.round(launch.initial_direction_world, 6))
    print("Feather clocking [deg]:", launch.initial_roll_deg)
    print("Omega body [rad/s]:", np.round(launch.initial_angular_velocity_body_rad_s, 4))
    print("Bending [mm]:", np.round(1000 * launch.initial_bending_m, 3))
    print("Bending velocity [m/s]:", np.round(launch.initial_bending_velocity_m_s, 4))
    print("Minimum clearance [mm]:", round(1000 * launch.minimum_clearance_m, 6))
    print("Bow contact:", launch.bow_hit)
    print("Max contact force [N]:", round(launch.max_contact_force_N, 3))
    print("Contact impulse [N s]:", round(launch.contact_impulse_Ns, 6))
    print("Stored usable bow energy [J]:", round(launch.bow_energy_J, 3))
    return launch


if __name__ == "__main__":
    example()
