"""
Traditional longbow launch model feeding arrow_flight_sim_updated.simulate().

The launch phase models:
  * a longbow force-draw curve and stored elastic energy;
  * moving string/nock acceleration;
  * a traditional side-shot (Mediterranean) arrow geometry;
  * lateral string/finger-release disturbance;
  * flexible shaft bending in two transverse directions;
  * contact with the bow grip (moving-boundary/Archer's Paradox);
  * launch attitude, velocity, angular velocity and bending state.

The returned release state is passed directly to the existing free-flight
simulation.

Important: bow length + draw weight + draw length do not uniquely identify a
real bow's dynamic response.  The additional geometry/release/efficiency
parameters are therefore explicit and should be measured for high fidelity.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

from arrow_flight_sim_updated import Arrow, Atmosphere, FlightResult, simulate, orientation_from_direction_and_roll

G = 9.80665


def _norm(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.asarray(v, float)
    return np.asarray(v, float) / n


@dataclass
class Longbow:
    length_m: float
    draw_strength_kgf: float
    draw_length_m: float

    # Geometry of a traditional longbow and its grip.
    brace_height_m: float = 0.16
    handle_length_m: float = 0.22
    handle_thickness_m: float = 0.035
    handle_height_m: float = 0.10

    # Dynamic bow parameters not determined by draw weight alone.
    limb_mass_kg: float = 0.08
    string_mass_kg: float = 0.010
    bow_efficiency: float = 0.78
    nock_efficiency: float = 0.995
    force_draw_exponent: float = 1.10

    # Traditional side-shot geometry. +Z is the arrow side of the grip.
    arrow_side_offset_m: float = 0.018
    nock_height_offset_m: float = 0.0
    aim_elevation_deg: float = 0.0

    # Mediterranean finger release. These describe the short lateral motion
    # of the string as it leaves the fingers. Zero gives an idealized release.
    release_lateral_velocity_m_s: float = 0.0
    release_lateral_displacement_m: float = 0.0
    release_duration_s: float = 0.0015

    # Optional vertical finger disturbance (useful for a high/low nocking point).
    release_vertical_velocity_m_s: float = 0.0

    def __post_init__(self):
        if min(self.length_m, self.draw_length_m) <= 0:
            raise ValueError("bow length and draw length must be positive")
        if self.draw_strength_kgf <= 0:
            raise ValueError("draw_strength_kgf must be positive")
        if not (0 < self.brace_height_m < self.draw_length_m):
            raise ValueError("brace_height_m must be between 0 and draw length")
        if not (0 < self.bow_efficiency <= 1):
            raise ValueError("bow_efficiency must be in (0,1]")
        if self.release_duration_s <= 0:
            raise ValueError("release_duration_s must be positive")

    @property
    def draw_force_N(self):
        return self.draw_strength_kgf * G

    @property
    def draw_coordinate_m(self):
        return self.draw_length_m - self.brace_height_m

    def draw_force(self, displacement_m):
        d = np.clip(displacement_m, 0.0, self.draw_coordinate_m)
        u = d / self.draw_coordinate_m
        return self.draw_force_N * u ** self.force_draw_exponent

    def stored_energy_J(self):
        # Numerical integral of force-draw curve.
        d = np.linspace(0, self.draw_coordinate_m, 2000)
        return float(np.trapz(self.draw_force(d), d) * self.bow_efficiency)


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
    bow_energy_J: float
    launch_solution: object = None
    flight_result: Optional[FlightResult] = None


class BowLaunchSimulator:
    """Launch-stage flexible-arrow model for a traditional longbow.

    During the very short bow-contact phase the launch model keeps the mean
    arrow attitude fixed to the full-draw aiming direction and solves CM
    translation plus two transverse bending coordinates. This avoids falsely
    treating the large local nock/contact moments as rigid-body spin. At bow
    exit, the resulting deformed/velocity field is decomposed into an initial
    attitude and angular velocity and handed to the full 6-DOF flight model.

    The grip is represented as a one-sided moving contact boundary. Thus the
    model can report both ordinary sliding contact (the normal mechanism of
    Archer's Paradox) and penetration/strike.
    """

    def __init__(self, arrow: Arrow, bow: Longbow):
        self.arrow = arrow
        self.bow = bow
        # Free-free first elastic mode: appropriate for an unsupported shaft
        # once released, unlike the cantilever mode in the existing flight file.
        self.beta = 4.730040744862704
        b = self.beta
        self.sigma = (np.cosh(b)-np.cos(b))/(np.sinh(b)-np.sin(b))
        grid = np.linspace(0, arrow.length_m, 2001)
        raw = self.mode_raw(grid)
        self._scale = np.max(np.abs(raw))
        self.modal_mass = 0.50 * arrow.shaft_mass
        self.modal_k = 0.50 * self.beta**4 * arrow.EI / arrow.length_m**3
        self.modal_zeta = 0.015
        self.modal_c = 2*self.modal_zeta*np.sqrt(self.modal_mass*self.modal_k)
        self.contact_k = 2.0e4
        self.contact_c = 15.0

        self.direction0 = _norm(np.array([
            np.cos(np.deg2rad(bow.aim_elevation_deg)),
            np.sin(np.deg2rad(bow.aim_elevation_deg)),
            -bow.arrow_side_offset_m/arrow.length_m,
        ]))
        self.q0 = orientation_from_direction_and_roll(self.direction0, 0.0)
        self.R0 = Rotation.from_quat(self.q0).as_matrix()

    def mode_raw(self, s):
        u = np.asarray(s,float)/self.arrow.length_m
        b=self.beta
        return np.cosh(b*u)+np.cos(b*u)-self.sigma*(np.sinh(b*u)+np.sin(b*u))

    def phi(self,s):
        return self.mode_raw(s)/self._scale

    def phi_prime(self,s):
        u=np.asarray(s,float)/self.arrow.length_m
        b=self.beta
        return (b/self.arrow.length_m)*(
            np.sinh(b*u)-np.sin(b*u)-self.sigma*(np.cosh(b*u)+np.cos(b*u))
        )/self._scale

    def _unpack(self,y):
        return y[:3],y[3:6],y[6],y[7],y[8],y[9],y[10],y[11]

    def _point(self,y,s):
        r,v,qy,qdy,qz,qdz,wy,wz=self._unpack(y)
        ph=float(self.phi(s))
        rb=np.array([s-self.arrow.x_cm,qy*ph,qz*ph])
        p=r+self.R0@rb
        vb=np.array([0,qdy*ph,qdz*ph])
        vw=v+self.R0@vb
        return p,vw,rb

    def _string_force(self,t,y):
        p,v,_=self._point(y,0.0)
        remaining=max(-p[0],0.0)
        Fmag=self.bow.draw_force(remaining)
        target=np.array([0.0,self.bow.nock_height_offset_m,0.0])
        if t < self.bow.release_duration_s:
            target[2]+=self.bow.release_lateral_displacement_m
        d=target-p
        L=np.linalg.norm(d)
        F=np.zeros(3) if L<1e-12 else Fmag*d/L
        if t < self.bow.release_duration_s:
            F[2]+=0.01*self.bow.draw_force_N*(self.bow.release_lateral_velocity_m_s-v[2])
            F[1]+=0.01*self.bow.draw_force_N*(self.bow.release_vertical_velocity_m_s-v[1])
        return self.bow.nock_efficiency*self.bow.bow_efficiency*F

    def _contact(self,y):
        F=np.zeros(3); Qy=0.; Qz=0.; min_clear=np.inf; min_s=None
        # Several material points approximate the finite-width grip.
        for s in np.linspace(0,self.arrow.length_m,11):
            p,v,_=self._point(y,float(s))
            if abs(p[0])>0.5*self.bow.handle_thickness_m or abs(p[1])>0.5*self.bow.handle_height_m:
                continue
            clear=p[2]-0.5*self.bow.handle_thickness_m
            if clear<min_clear:
                min_clear=clear; min_s=s
            if clear<0:
                fn=-self.contact_k*clear-self.contact_c*min(v[2],0.0)
                f=np.array([0.,0.,fn])
                F+=f
                ph=float(self.phi(s))
                Qy+=np.dot(self.R0.T@f,[0,ph,0])
                Qz+=np.dot(self.R0.T@f,[0,0,ph])
        return F,Qy,Qz,min_clear,min_s

    def _rhs(self,t,y):
        r,v,qy,qdy,qz,qdz,wy,wz=self._unpack(y)
        F=np.array([0.,-self.arrow.total_mass_kg*G,0.])
        Qy=Qz=0.
        Fs=self._string_force(t,y)
        F+=Fs
        ph0=float(self.phi(0.0))
        Qy+=np.dot(self.R0.T@Fs,[0,ph0,0])
        Qz+=np.dot(self.R0.T@Fs,[0,0,ph0])
        Fc,Qcy,Qcz,_,_=self._contact(y)
        F+=Fc; Qy+=Qcy; Qz+=Qcz
        # External rigid-body pitch/yaw impulse is integrated separately from
        # the flexible modal coordinates. This prevents the modal bending from
        # being mistaken for whole-arrow rotation, while still allowing a high
        # nock point or asymmetric finger release to leave the bow with angular
        # velocity. Roll is not driven by this symmetric side-shot geometry.
        nock_p,_,_=self._point(y,0.0)
        M_world=np.cross(nock_p-r,Fs)+np.cross(self._point(y,0.0)[0]-r,Fc)
        Iyy=self.arrow.Iyy; Izz=self.arrow.Izz
        wy_dot=M_world[1]/Iyy
        wz_dot=M_world[2]/Izz
        qydd=(Qy-self.modal_c*qdy-self.modal_k*qy)/self.modal_mass
        qzdd=(Qz-self.modal_c*qdz-self.modal_k*qz)/self.modal_mass
        return np.r_[v,F/self.arrow.total_mass_kg,qdy,qydd,qdz,qzdd,wy_dot,wz_dot]

    def _exit_event(self,t,y):
        return self._point(y,0.0)[0][0]-0.5*self.bow.handle_thickness_m

    def simulate_launch(self,t_end=0.04,max_step=2e-5):
        # At full draw the nock is behind the grip and the point is forward.
        nock_z=self.bow.arrow_side_offset_m
        cm0=np.array([
            -self.bow.draw_coordinate_m+self.arrow.x_cm*self.direction0[0],
            self.bow.nock_height_offset_m+self.arrow.x_cm*self.direction0[1],
            nock_z+self.arrow.x_cm*self.direction0[2],
        ])
        y0=np.r_[cm0,np.zeros(3),0.,0.,0.,0.,0.,0.]
        def event(t,y): return self._exit_event(t,y)
        event.terminal=True; event.direction=1
        sol=solve_ivp(self._rhs,(0,t_end),y0,method='RK45',max_step=max_step,rtol=2e-7,atol=1e-10,events=event,dense_output=True)
        y=sol.y[:,-1]
        exit_t=float(sol.t[-1]); release_t=min(self.bow.release_duration_s,exit_t)
        r,v,qy,qdy,qz,qdz,wy,wz=self._unpack(y)
        tip_p,tip_v,tip_rb=self._point(y,self.arrow.length_m)
        omega_body=np.array([0.0,wy,wz])

        direction_exit=_norm(self.R0@np.array([1.,float(self.phi_prime(self.arrow.length_m))*qy,float(self.phi_prime(self.arrow.length_m))*qz]))
        q_exit=orientation_from_direction_and_roll(direction_exit,0.0)

        clear=[]
        for tt,yy in zip(sol.t,sol.y.T):
            clear.append(self._contact(yy)[3])
        clear=np.asarray(clear)
        finite=np.isfinite(clear)
        if np.any(finite):
            k=np.nanargmin(np.where(finite,clear,np.inf)); min_clear=float(clear[k]); min_t=float(sol.t[k])
        else: min_clear=float('inf'); min_t=0.
        # Penetration is a numerical/geometry diagnostic. A traditional shot
        # can legitimately contact the grip; report contact separately.
        contact_seen=np.any(finite)
        bow_hit=bool(min_clear< -0.002)
        return LaunchResult(
            reason='bow_exit' if sol.t_events[0].size else 'time_limit',
            time_s=exit_t,release_time_s=release_t,left_bow_time_s=exit_t,
            initial_position_m=r.copy(),initial_velocity_world_m_s=v.copy(),
            initial_direction_world=direction_exit.copy(),initial_roll_deg=0.0,
            initial_orientation_quaternion_xyzw=q_exit.copy(),
            initial_angular_velocity_body_rad_s=omega_body.copy(),
            initial_bending_m=np.array([qy,qz]),
            initial_bending_velocity_m_s=np.array([qdy,qdz]),
            minimum_clearance_m=min_clear,minimum_clearance_time_s=min_t,
            bow_hit=bow_hit,bow_energy_J=self.bow.stored_energy_J(),launch_solution=sol
        )

    def launch_and_fly(self,atmosphere:Atmosphere,**flight_kwargs):
        launch=self.simulate_launch()
        launch.flight_result=simulate(
            self.arrow,atmosphere,initial_position_m=launch.initial_position_m,
            initial_velocity_world_m_s=launch.initial_velocity_world_m_s,
            initial_direction=launch.initial_direction_world,
            initial_roll_deg=launch.initial_roll_deg,
            initial_orientation_quaternion_xyzw=launch.initial_orientation_quaternion_xyzw,
            initial_angular_velocity_body_rad_s=launch.initial_angular_velocity_body_rad_s,
            initial_bending_m=launch.initial_bending_m,
            initial_bending_velocity_m_s=launch.initial_bending_velocity_m_s,**flight_kwargs)
        return launch


def example():
    arrow=Arrow(length_m=.75,shaft_outer_d_m=.0065,shaft_inner_d_m=.0045,total_mass_kg=.028,point_mass_kg=.009,point_x_m=.75,spine=1000)
    bow=Longbow(length_m=1.85,draw_strength_kgf=40.,draw_length_m=.72,brace_height_m=.16,arrow_side_offset_m=.018)
    sim=BowLaunchSimulator(arrow,bow); launch=sim.simulate_launch()
    print('--- BOW LAUNCH ---')
    print('Reason:',launch.reason)
    print('Exit time [ms]:',round(1000*launch.left_bow_time_s,3))
    print('CM velocity [m/s]:',np.round(launch.initial_velocity_world_m_s,4))
    print('Direction:',np.round(launch.initial_direction_world,6))
    print('Omega body [rad/s]:',np.round(launch.initial_angular_velocity_body_rad_s,4))
    print('Bending [mm]:',np.round(1000*launch.initial_bending_m,3))
    print('Bending velocity [m/s]:',np.round(launch.initial_bending_velocity_m_s,4))
    print('Min clearance [mm]:',round(1000*launch.minimum_clearance_m,3))
    print('Bow hit/penetration:',launch.bow_hit)
    print('Stored usable bow energy [J]:',round(launch.bow_energy_J,3))
    return launch


if __name__=='__main__': example()
