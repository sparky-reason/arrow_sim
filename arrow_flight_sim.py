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
    def x_center_m(self):
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
    shaft_cd: float = 1.0
    point_cd: float = 0.5
    point_area_m2: float = 0.0001
    # Common static-spine reference: 28 in span, 1.94 lb center load.
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

        self.A = np.pi / 4 * (self.shaft_outer_d_m**2 - self.shaft_inner_d_m**2)
        self.I = np.pi / 64 * (self.shaft_outer_d_m**4 - self.shaft_inner_d_m**4)

        # Static spine number = deflection in inches * 1000.
        self.spine_deflection_m = self.spine / 1000.0 * 0.0254
        F_test = self.spine_test_mass_kg * G
        # Simply supported beam with a center load: delta = F L^3 / (48 EI).
        self.EI = F_test * self.spine_test_span_m**3 / (48.0 * self.spine_deflection_m)

        self.shaft_mass = self.total_mass_kg - self.point_mass_kg - self.feathers.mass_kg
        if self.shaft_mass <= 0:
            raise ValueError("total_mass_kg must exceed point_mass_kg + feather mass")

        self.lambda_b = 1.875104068711961
        m, L = self.shaft_mass, self.length_m
        mp, mf = self.point_mass_kg, self.feathers.mass_kg
        self.x_cm = (m*L/2 + mp*self.point_x_m + mf*self.feathers.x_center_m) / self.total_mass_kg
        self.Iyy = (m*(L**2/12 + (L/2-self.x_cm)**2)
                    + mp*(self.point_x_m-self.x_cm)**2
                    + mf*(self.feathers.x_center_m-self.x_cm)**2)
        self.Izz = self.Iyy
        radius = self.shaft_outer_d_m / 2
        self.Ixx = 0.5 * (m + mp) * radius**2

        # First bending mode, reduced-order flexible shaft model.
        self.modal_mass = 0.236 * self.shaft_mass
        self.modal_k = (self.lambda_b**4 * self.EI / self.length_m**3) * 0.236
        self.modal_c = 2 * 0.02 * np.sqrt(self.modal_k * self.modal_mass)

    def _mode_raw(self, x):
        s = np.asarray(x) / self.length_m
        b = self.lambda_b
        sigma = (np.cosh(b)+np.cos(b)) / (np.sinh(b)+np.sin(b))
        return np.cosh(b*s)-np.cos(b*s)-sigma*(np.sinh(b*s)-np.sin(b*s))

    def phi(self, x):
        return self._mode_raw(x) / self._mode_raw(self.length_m)

@dataclass
class FlightResult:
    reason: str
    time_s: float
    position_cm_world_m: np.ndarray
    direction_world: np.ndarray
    orientation_quaternion_xyzw: np.ndarray
    velocity_cm_world_m_s: np.ndarray
    angular_velocity_body_rad_s: np.ndarray
    bending_amplitude_m: float
    solution: object = None
    @property
    def speed_m_s(self):
        return float(np.linalg.norm(self.velocity_cm_world_m_s))

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Direction/vector must have non-zero length")
    return v / n

def orientation_from_direction_and_roll(direction_world, roll_deg=0.0,
                                        reference_up=np.array([0.,0.,1.])):
    """Body +X is nock->point. roll_deg rotates the arrow around +X."""
    ex = _normalize(direction_world)
    up = _normalize(reference_up)
    if abs(np.dot(ex, up)) > 0.98:
        up = np.array([0.,1.,0.])
    ey = _normalize(up - np.dot(up, ex)*ex)
    ez = _normalize(np.cross(ex, ey))
    R0 = np.column_stack([ex, ey, ez])
    a = np.deg2rad(roll_deg)
    Rx = np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
    return Rotation.from_matrix(R0 @ Rx).as_quat()

def quat_derivative(q, omega_body):
    x,y,z,w = q
    wx,wy,wz = omega_body
    return 0.5*np.array([w*wx+y*wz-z*wy, w*wy+z*wx-x*wz,
                         w*wz+x*wy-y*wx, -x*wx-y*wy-z*wz])

def aero_force(v_rel_world, axis_world, rho, area, cd,
               lift_slope=0.0, stall_rad=np.deg2rad(18.0)):
    speed = np.linalg.norm(v_rel_world)
    if speed < 1e-9:
        return np.zeros(3)
    flow = v_rel_world / speed
    q = 0.5*rho*speed**2
    n = axis_world - np.dot(axis_world, flow)*flow
    nn = np.linalg.norm(n)
    drag = -q*cd*area*flow
    if nn < 1e-10 or lift_slope == 0:
        return drag
    n /= nn
    alpha = np.arccos(np.clip(np.dot(axis_world, -flow), -1, 1))
    alpha *= np.sign(np.dot(n, axis_world))
    alpha = np.clip(alpha, -stall_rad, stall_rad)
    return drag + q*(lift_slope*alpha)*area*n

def simulate(arrow, atmosphere, initial_position_m, initial_direction,
             initial_speed_m_s, initial_roll_deg=0.0,
             initial_angular_velocity_body_rad_s=None,
             wind_world_m_s=None, wall_x_m=None, ground_z_m=0.0,
             stop_at_ground=True, t_end=5.0, dt=0.001):
    """
    initial_position_m is the CENTER-OF-MASS position.
    initial_direction is the shaft direction nock -> point.
    initial_speed_m_s is the center-of-mass speed along that direction.
    initial_roll_deg defines the feather orientation around the shaft axis.

    The simulation terminates at a wall x=wall_x_m, at the ground z=ground_z_m,
    or at t_end. Wall and ground positions refer to the arrow CENTER OF MASS.
    """
    r0 = np.asarray(initial_position_m, dtype=float)
    ex0 = _normalize(initial_direction)
    v0 = float(initial_speed_m_s)*ex0
    omega0 = np.zeros(3) if initial_angular_velocity_body_rad_s is None else np.asarray(initial_angular_velocity_body_rad_s, dtype=float)
    wind = np.zeros(3) if wind_world_m_s is None else np.asarray(wind_world_m_s, dtype=float)
    q0 = orientation_from_direction_and_roll(ex0, initial_roll_deg)
    y0 = np.r_[r0,v0,q0,omega0,0.,0.]
    rho = atmosphere.rho
    shaft_area = arrow.shaft_outer_d_m * arrow.length_m

    def rhs(t,y):
        r,v = y[:3],y[3:6]
        quat = y[6:10] / np.linalg.norm(y[6:10])
        omega = y[10:13]
        qb,qbd = y[13:15]
        Rbw = Rotation.from_quat(quat).as_matrix()
        ex,ey,ez = Rbw[:,0],Rbw[:,1],Rbw[:,2]
        F = np.array([0.,0.,-arrow.total_mass_kg*G])
        M = np.zeros(3)
        F += aero_force(v-wind,ex,rho,shaft_area,arrow.shaft_cd)

        rp_b = np.array([arrow.point_x_m-arrow.x_cm,0.,0.])
        rp_w = Rbw @ rp_b
        vp = v + np.cross(Rbw@omega,rp_w)
        Fp = aero_force(vp-wind,ex,rho,arrow.point_area_m2,arrow.point_cd)
        F += Fp
        M += np.cross(rp_b,Rbw.T@Fp)

        f = arrow.feathers
        xf,phi_f = f.x_center_m,arrow.phi(f.x_center_m)
        rf_b = np.array([xf-arrow.x_cm,qb*phi_f,0.])
        vf = v + Rbw@(np.cross(omega,rf_b)+np.array([0.,qbd*phi_f,0.]))
        rel = vf-wind
        speed = np.linalg.norm(rel)
        if speed > 1e-8:
            flow=rel/speed; qdyn=0.5*rho*speed**2; cant=np.deg2rad(f.cant_deg)
            for k in range(f.count):
                a=2*np.pi*k/f.count
                radial=np.cos(a)*ey+np.sin(a)*ez
                normal=np.cos(cant)*radial+np.sin(cant)*np.cross(ex,radial)
                alpha=np.arcsin(np.clip(np.dot(flow,normal),-1,1))
                alpha=np.clip(alpha,-np.deg2rad(f.stall_deg),np.deg2rad(f.stall_deg))
                Ff=-qdyn*f.cd*f.area_each_m2*flow + qdyn*(f.lift_slope_per_rad*alpha)*f.area_each_m2*normal
                F += Ff
                M += np.cross(rf_b,Rbw.T@Ff)

        Fbody=Rbw.T@F
        Qb=Fbody[1]*arrow.phi(arrow.length_m)
        qbdd=(Qb-arrow.modal_c*qbd-arrow.modal_k*qb)/arrow.modal_mass
        I=np.diag([arrow.Ixx,arrow.Iyy,arrow.Izz])
        omega_dot=np.linalg.solve(I,M-np.cross(omega,I@omega))
        return np.r_[v,F/arrow.total_mass_kg,quat_derivative(quat,omega),omega_dot,qbd,qbdd]

    events=[]; names=[]
    if wall_x_m is not None:
        def wall_event(t,y): return y[0]-wall_x_m
        wall_event.terminal=True; wall_event.direction=0
        events.append(wall_event); names.append("wall")
    if stop_at_ground:
        def ground_event(t,y): return y[2]-ground_z_m
        ground_event.terminal=True; ground_event.direction=-1
        events.append(ground_event); names.append("ground")

    sol=solve_ivp(rhs,(0.,t_end),y0,max_step=dt,rtol=2e-7,atol=1e-9,
                  events=events if events else None,dense_output=True)
    reason="time_limit"
    if events:
        hit=None; hit_time=np.inf
        for i,ts in enumerate(sol.t_events):
            if len(ts) and ts[0]<hit_time: hit_time=ts[0]; hit=i
        if hit is not None: reason=names[hit]

    yf=sol.y[:,-1]
    qf=yf[6:10]/np.linalg.norm(yf[6:10])
    Rf=Rotation.from_quat(qf).as_matrix()
    return FlightResult(reason, float(sol.t[-1]), yf[:3], _normalize(Rf[:,0]), qf,
                        yf[3:6],yf[10:13],float(yf[13]),sol)

if __name__ == "__main__":
    arrow=Arrow(length_m=0.75,shaft_outer_d_m=0.0065,shaft_inner_d_m=0.0045,
                total_mass_kg=0.028,point_mass_kg=0.009,point_x_m=0.75,spine=1000,
                feathers=Feather(x_start_m=0.08,x_end_m=0.19,height_m=0.012,
                                 area_each_m2=0.00075,count=3,cant_deg=2.,mass_kg=0.0005))
    atmosphere=Atmosphere()
    print("Air density:",atmosphere.rho,"kg/m^3")
    print("Spine:",arrow.spine)
    print("Static deflection:",arrow.spine_deflection_m,"m")
    print("Effective EI:",arrow.EI,"N*m^2")
    print("Center of mass from nock:",arrow.x_cm,"m")

    wall=simulate(arrow,atmosphere,[0.,0.,1.5],[1.,0.,0.],70.,initial_roll_deg=0.,wall_x_m=20.)
    print("\n--- WALL ---")
    print("Reason:",wall.reason,"time:",wall.time_s)
    print("Final CM position:",wall.position_cm_world_m)
    print("Final arrow direction:",wall.direction_world)
    print("Final speed:",wall.speed_m_s)

    ground=simulate(arrow,atmosphere,[0.,0.,1.5],[1.,0.,0.],70.,initial_roll_deg=0.,ground_z_m=0.)
    print("\n--- GROUND ---")
    print("Reason:",ground.reason,"time:",ground.time_s)
    print("Final CM position:",ground.position_cm_world_m)
    print("Final arrow direction:",ground.direction_world)
    print("Final speed:",ground.speed_m_s)
