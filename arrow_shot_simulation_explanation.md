# Arrow shot simulation: equations and approximations

This document describes the equations actually implemented in `arrow_shot_simulator(1).py`. The model is a **reduced-order simulation** consisting of:

1. bow energy and force–draw modeling;
2. a discretized Euler–Bernoulli beam model for the arrow during launch;
3. unilateral riser contact and a reduced dynamic string model;
4. projection of the launch state into rigid-body + two-mode flight variables;
5. free-flight 6-DOF rigid-body dynamics with two transverse bending coordinates;
6. aerodynamic drag, normal/lift forces, gravity, wind, and event detection.

The coordinate system is

\[
+X=\text{range},\qquad +Y=\text{up},\qquad +Z=\text{sideways}.
\]

The arrow body axis is \(+x\), from nock to point. World quantities are expressed in \(X,Y,Z\); body quantities are expressed in \(x,y,z\). The source explicitly describes the implementation as reduced-order and states that bow-limb and string inertia are not explicitly finite-element modeled. [Source: code comments/docstrings, lines 899–925.]

---

## 1. Input parameters and derived material/geometry quantities

### 1.1 Air density

The atmosphere is specified by pressure \(p\) and temperature \(T\). Density is calculated from the ideal-gas equation of state:

\[
\boxed{\rho=\frac{p}{R_{\rm air}T}}
\]

where

- \(\rho\) = air density [kg m\(^{-3}\)];
- \(p\) = air pressure [Pa];
- \(T\) = absolute air temperature [K];
- \(R_{\rm air}=287.05\) J kg\(^{-1}\) K\(^{-1}\) = specific gas constant for dry air.

This is the ideal-gas law for air. The default values are \(p=101325\) Pa and \(T=288.15\) K.

### 1.2 Shaft cross-section

For outer diameter \(d_o\) and inner diameter \(d_i\), the shaft area is

\[
\boxed{A_s=\frac{\pi}{4}(d_o^2-d_i^2)}
\]

and the second moment of area is

\[
\boxed{I_s=\frac{\pi}{64}(d_o^4-d_i^4)}.
\]

These are the standard geometric properties of a hollow circular section.

The bending stiffness is represented by \(EI\), where

- \(E\) = effective Young's modulus;
- \(I_s\) = shaft second moment of area.

### 1.3 Static spine -> effective \(EI\)

The code interprets the supplied spine number as a deflection in inches divided by 1000:

\[
\delta_{\rm spine}
=
\frac{\text{spine}}{1000}(0.0254).
\]

The effective stiffness is then obtained from

\[
\boxed{\delta=\frac{F L^3}{48EI}}
\]

or

\[
\boxed{EI=\frac{F L^3}{48\delta}}.
\]

Here

- \(F=m_{\rm test}g\) = specified center test load;
- \(L\) = 28 in test span;
- \(\delta\) = measured/encoded center deflection;
- \(g=9.80665\) m s\(^{-2}\).

**Origin:** this is the classical small-deflection Euler–Bernoulli result for a simply supported beam with a point load at mid-span.

**Approximation:** the resulting \(EI\) is used as an effective arrow stiffness even though the later launch model treats the shaft as a free beam with moving loads and contact. Thus the spine test is only a calibration of \(EI\), not the actual launch boundary condition.

### 1.4 Mass distribution

Shaft mass is

\[
m_s=m_{\rm total}-m_{\rm point}-m_{\rm feathers}.
\]

The shaft is otherwise treated as uniformly distributed.

The longitudinal center of mass is

\[
\boxed{
x_{\rm CM}
=
\frac{
m_s L/2
+m_p x_p
+m_f x_f
}{
m_{\rm total}
}}
\]

where

- \(L\) = arrow length;
- \(m_p,x_p\) = point mass and point position;
- \(m_f,x_f\) = feather mass and feather centroid;
- \(m_s\) = shaft mass.

For transverse rigid-body moments the code uses

\[
\boxed{
I_y=I_z
=
m_s\left[\frac{L^2}{12}
+\left(\frac L2-x_{\rm CM}\right)^2\right]
+m_p(x_p-x_{\rm CM})^2
+m_f(x_f-x_{\rm CM})^2
}
\]

which is the parallel-axis theorem applied to the uniform shaft and lumped masses.

The axial spin inertia is approximated as

\[
\boxed{I_x=\frac12(m_s+m_p)r^2},
\qquad r=\frac{d_o}{2}.
\]

**Approximation:** the shaft/point contribution is treated as a solid circular mass distribution for \(I_x\), and feather spin inertia is neglected.

---

# 2. Bow model

## 2.1 Draw force

Draw strength \(S\) is converted from kgf to newtons:

\[
\boxed{F_{\rm draw,max}=Sg}.
\]

The actual draw coordinate is

\[
\boxed{d_{\rm max}=d_{\rm draw}+d_{\rm error}}.
\]

If a measured force–draw table is supplied, the code linearly interpolates the measured force:

\[
F_{\rm draw}(d)=\operatorname{interp}(d).
\]

Otherwise the generic model is

\[
\boxed{
F_{\rm draw}(d)
=
F_{\rm draw,max}
\left(\frac{d}{d_{\rm max}}\right)^p
}
\]

with

\[
p=
\begin{cases}
1.00,&\text{longbow},\\
0.82,&\text{recurve},
\end{cases}
\]

unless an explicit exponent is supplied.

**Origin:** this is a phenomenological force–draw approximation, not a fundamental bow equation. The code itself recommends measured force–draw data for realistic prediction.

## 2.2 Stored bow energy

Stored elastic energy is the work required to draw the bow:

\[
\boxed{
E_{\rm bow}
=
\int_0^{d_{\rm max}}F_{\rm draw}(d)\,{\rm d}d
}.
\]

The code evaluates this numerically with a 4000-point trapezoidal integration.

The usable energy is

\[
\boxed{E_{\rm usable}=\eta_{\rm bow}E_{\rm bow}}
\]

where \(\eta_{\rm bow}\) is the prescribed bow efficiency.

For the reduced force–draw model the force delivered to the nock is

\[
\boxed{
F_{\rm string}(d)
=
\eta_{\rm bow}\eta_{\rm nock}F_{\rm draw}(d)
}.
\]

If a measured force–draw table is supplied, it is treated as the actual string force and only \(\eta_{\rm nock}\) is applied.

**Approximation:** bow efficiency is a scalar multiplier. Limb deformation, limb inertia, string vibration, and detailed energy transfer are not solved from first principles.

---

# 3. Arrow launch: discretized beam

During launch the shaft is represented by \(N\) material nodes at

\[
s_i=i h,\qquad h=\frac{L}{N-1}.
\]

The transverse displacements are

\[
u_y(s,t),\qquad u_z(s,t).
\]

Axial motion is represented by one common nock coordinate \(x_n(t)\), rather than by an axial finite-element field.

## 3.1 Euler–Bernoulli beam stiffness

For an Euler–Bernoulli beam, the bending strain energy is

\[
U_b=\frac12\int_0^L EI\left(\frac{\partial^2u}{\partial s^2}\right)^2ds.
\]

The corresponding stiffness operator is

\[
EI\frac{\partial^4u}{\partial s^4}.
\]

**Origin:** Euler–Bernoulli beam theory, under the assumptions of small transverse deflection, plane cross-sections remaining plane, and negligible shear deformation.

The code approximates derivatives with finite differences. The first derivative uses central differences in the interior:

\[
u'_i\approx\frac{u_{i+1}-u_{i-1}}{2h},
\]

with one-sided differences at the ends.

The second derivative is

\[
\boxed{
u''_i\approx
\frac{u_{i-1}-2u_i+u_{i+1}}{h^2}
}.
\]

Let \(D_2\) be this discrete second-derivative matrix. The beam stiffness matrix is assembled as

\[
\boxed{K=EI\,h\,D_2^{T}D_2}.
\]

The factor \(h\) is the quadrature approximation to the integral in the bending-energy expression.

## 3.2 Nodal mass

The shaft mass is distributed uniformly over the nodes using trapezoidal weighting:

\[
m_i\approx m_s\frac{h}{L},
\]

with half weight at the two ends. Point and feather masses are added to the nearest nodes.

Thus the launch equations use a diagonal lumped mass matrix

\[
M=\operatorname{diag}(m_1,\ldots,m_N).
\]

---

# 4. Arrow compression and geometric stiffness during launch

The string force compresses the arrow. Instead of modeling axial deformation, the code estimates the compressive force remaining at each position.

Let \(m_{\le s}\) be the cumulative mass from the nock to position \(s\). The axial force distribution is approximated by

\[
\boxed{
N(s)
=
\max(F_{\rm string},0)
\left(
1-\frac{m_{\le s}}{m_{\rm total}}
\right)
}.
\]

**Approximation:** this assumes the nock force is progressively reduced by the acceleration of the downstream mass. It is a reduced representation of axial compression, introduced mainly to capture second-order bending effects.

The associated geometric stiffness is obtained from the second-order beam term

\[
\boxed{
U_g
=
-\frac12\int_0^L N(s)\,[u'(s)]^2\,ds
}.
\]

After discretization,

\[
\boxed{
K_g
=
-\sum_i N_{i+1/2}\,h\,d_i d_i^T
}
\]

where \(d_i\) is the discrete first-derivative vector on segment \(i\).

The effective launch stiffness is therefore

\[
\boxed{K_{\rm launch}=K+K_g}.
\]

**Origin:** this is the classical second-order/P-\(\Delta\) or geometric-stiffness term for a beam under axial compression.

---

# 5. Pre-release equilibrium

Before release, the nock is constrained by the string. The code solves the transverse static equilibrium

\[
\boxed{
K_{\rm launch}u=f_g
}
\]

subject to the nock constraint and unilateral riser-contact constraints.

Gravity is

\[
\boxed{\mathbf F_g=m\mathbf g},
\qquad
\mathbf g=(0,-g,0).
\]

Its local transverse components are

\[
g_y=\mathbf g\cdot\mathbf e_y,
\qquad
g_z=\mathbf g\cdot\mathbf e_z.
\]

The corresponding nodal load is

\[
f_{g,y,i}=m_i g_y,
\qquad
f_{g,z,i}=m_i g_z.
\]

The nock displacement/slope are constrained during this equilibrium solve.

A tiny numerical regularization is added to the stiffness matrix to select the minimum-rotation solution of the otherwise free rigid-body null space. This is explicitly intended as numerical regularization, not physical stiffness.

---

# 6. Unilateral riser contact

For each possible contact node, the normal gap is

\[
\boxed{
g_i
=
z_{\rm arrow,offset}+u_{z,i}-z_{\rm contact}
}.
\]

The physical contact conditions are the standard complementarity conditions

\[
\boxed{
g_i\ge0,\qquad
\lambda_i\ge0,\qquad
g_i\lambda_i=0
}
\]

where

- \(g_i\) = normal clearance;
- \(\lambda_i\) = normal contact force.

**Origin:** unilateral contact/Kuhn–Tucker complementarity conditions.

A node with penetration is activated as a contact constraint; nodes requiring a negative reaction are removed from the active set.

The implementation therefore does **not** use a penalty spring \(F=k\,\delta\).

During time integration, acceleration-level Baumgarte stabilization is used:

\[
\boxed{
a_{\rm target}
=
-2\zeta_B\omega_B\dot g
-\omega_B^2\min(g,0)
}
\]

with \(\omega_B=2\pi(2500)\) s\(^{-1}\) and \(\zeta_B=0.9\).

The required reaction is

\[
\boxed{
\lambda_i
=
m_i(a_{\rm target}-a_{i,\rm free})
}
\]

for an active node.

**Approximation:** contact is reduced to normal contact with diagonal nodal masses. Friction is not mechanically modeled even though a friction parameter exists.

---

# 7. Dynamic string and release

The string endpoint has an effective mass

\[
\boxed{
m_{\rm string,eff}
=
f_m m_{\rm string}
}
\]

where \(f_m\) is the prescribed effective-mass fraction.

The lateral and vertical restoring forces are linear spring–damper forces:

\[
\boxed{
F_y=-k_y(y_s-y_{\rm eq})-c_y\dot y_s
}
\]

\[
\boxed{
F_z=-k_z(z_s-z_{\rm eq})-c_z\dot z_s
}.
\]

Thus, while attached,

\[
\boxed{
m_{\rm string,eff}\ddot y_s=F_y
}
\]

and

\[
\boxed{
m_{\rm string,eff}\ddot z_s=F_z.
}
\]

The axial arrow acceleration is approximated by

\[
\boxed{
\ddot x_n=
\frac{F_{\rm string}}
{m_{\rm arrow}+m_{\rm string,eff}}
}
\]

while the string and nock are coupled.

After release the arrow alone receives

\[
\boxed{
\ddot x_n=
\frac{F_{\rm string}}{m_{\rm arrow}}
}
\]

and the string endpoint evolves independently under its spring–damper model.

**Approximation:** the bow/string system is not a finite-element bow model. The dynamic string parameters \(k_y,k_z,c_y,c_z,m_{\rm string,eff}\) are explicit tuned/measured parameters.

Release disturbances are added directly as initial string displacement, velocity, acceleration, and roll-rate perturbations.

---

# 8. Launch time integration

The launch solver uses a fixed-step kick–drift / velocity-Verlet-like explicit update.

For a coordinate \(u\),

\[
\dot u_{n+1/2}
=
\dot u_n+\frac12\Delta t\,a_n,
\]

\[
u_{n+1}
=
u_n+\Delta t\,\dot u_{n+1/2},
\]

then the acceleration is recomputed and

\[
\boxed{
\dot u_{n+1}
=
\dot u_{n+1/2}
+\frac12\Delta t\,a_{n+1}.
}
\]

**Origin:** standard second-order symplectic/velocity-Verlet family integration.

The default launch step is \(2\,\mu{\rm s}\), and the launch is stopped only after the **entire arrow** has cleared the riser, not merely when the nock has left the string.

---

# 9. Conversion from launch beam state to flight state

At bow exit the nodal positions are converted into a center-of-mass state.

With mass weights

\[
w_i=\frac{m_i}{\sum_jm_j},
\]

the local center of mass is

\[
\boxed{
\mathbf r_{\rm CM}
=
\sum_i w_i\mathbf r_i
}
\]

and the CM velocity is

\[
\boxed{
\mathbf v_{\rm CM}
=
\sum_i w_i\mathbf v_i.
}
\]

The rigid-body angular velocity is obtained by fitting

\[
\boxed{
\mathbf v_i-\mathbf v_{\rm CM}
\approx
\boldsymbol\omega\times
(\mathbf r_i-\mathbf r_{\rm CM})
}
\]

in a weighted least-squares sense.

This is the kinematic rigid-body velocity relation

\[
\mathbf v=\mathbf v_{\rm CM}+\boldsymbol\omega\times\mathbf r.
\]

The arrow direction is the normalized vector from the nock to the point:

\[
\boxed{
\mathbf e_x=
\frac{\mathbf r_{\rm tip}-\mathbf r_{\rm nock}}
{\|\mathbf r_{\rm tip}-\mathbf r_{\rm nock}\|}
}.
\]

The launch attitude is constructed from this direction and the specified roll/feather clocking.

---

# 10. First free-free bending mode used in flight

The flight model retains one transverse bending mode in each of the \(y\) and \(z\) directions.

For a uniform free-free Euler–Bernoulli beam, the first non-rigid eigenvalue parameter is

\[
\boxed{\lambda_1=4.730040744862704}.
\]

The mode shape used is

\[
\boxed{
\phi(x)=
\frac{
\cosh(\lambda x/L)+\cos(\lambda x/L)
-\sigma[\sinh(\lambda x/L)+\sin(\lambda x/L)]
}{
\phi_{\rm raw}(L)
}
}
\]

where

\[
\boxed{
\sigma=
\frac{\cosh\lambda-\cos\lambda}
{\sinh\lambda-\sin\lambda}
}.
\]

**Origin:** exact Euler–Bernoulli free-free beam eigenfunctions.

The local transverse deformation is represented as

\[
\boxed{
w_y(x,t)=q_y(t)\phi(x),
\qquad
w_z(x,t)=q_z(t)\phi(x).
}
\]

The derivative \(\phi'(x)\) is used to obtain the local centerline tangent.

**Approximation:** only the first bending mode in each transverse direction is retained during flight. Higher modes are discarded.

---

# 11. Modal mass, stiffness, and damping in flight

The effective modal mass is approximated as

\[
\boxed{
m_{\rm modal}=0.25\,m_s.
}
\]

The modal stiffness is

\[
\boxed{
k_{\rm modal}
=
0.25\,
\frac{\lambda_1^4 EI}{L^3}.
}
\]

This follows from the Euler–Bernoulli relation

\[
\boxed{
\omega_n^2
=
\frac{\lambda_n^4EI}{\mu L^4}
}
\]

combined with the reduced modal representation and the code's prescribed \(0.25\) effective-mass factor.

The damping ratio is prescribed as

\[
\zeta=0.02.
\]

The equivalent viscous modal damping is

\[
\boxed{
c_{\rm modal}
=
2\zeta\sqrt{k_{\rm modal}m_{\rm modal}}.
}
\]

**Origin:** standard single-degree-of-freedom modal damping relation.

---

# 12. Flight state vector

The flight ODE state is

\[
\boxed{
\mathbf y=
[
\mathbf r,\,
\mathbf v,\,
q_x,q_y,q_z,q_w,\,
\boldsymbol\omega,\,
q_y^{(b)},\dot q_y^{(b)},
q_z^{(b)},\dot q_z^{(b)}
]
}
\]

with:

- \(\mathbf r\): CM position;
- \(\mathbf v\): CM velocity;
- \(q\): orientation quaternion;
- \(\boldsymbol\omega\): body angular velocity;
- \(q_y^{(b)},q_z^{(b)}\): bending amplitudes;
- \(\dot q_y^{(b)},\dot q_z^{(b)}\): bending velocities.

The quaternion is normalized before constructing the rotation matrix.

---

# 13. Position and velocity of points on the flexible arrow

For a shaft point at local coordinate \(x\), the rigid-body offset from the CM is

\[
\mathbf r_b(x)
=
\begin{bmatrix}
x-x_{\rm CM}\\
q_y^{(b)}\phi(x)\\
q_z^{(b)}\phi(x)
\end{bmatrix}.
\]

If \(R\) maps body coordinates to world coordinates,

\[
\boxed{
\mathbf r(x)
=
\mathbf r_{\rm CM}+R\mathbf r_b(x)
}.
\]

The local point velocity is

\[
\boxed{
\mathbf v_b(x)
=
\boldsymbol\omega\times\mathbf r_b(x)
+
\begin{bmatrix}
0\\
\dot q_y^{(b)}\phi(x)\\
\dot q_z^{(b)}\phi(x)
\end{bmatrix}
}
\]

and therefore

\[
\boxed{
\mathbf v(x)
=
\mathbf v_{\rm CM}+R\mathbf v_b(x).
}
\]

The local tangent is

\[
\boxed{
\mathbf t_b(x)
=
\operatorname{normalize}
\begin{bmatrix}
1\\
q_y^{(b)}\phi'(x)\\
q_z^{(b)}\phi'(x)
\end{bmatrix}
}
\]

and its world version is

\[
\boxed{\mathbf t=R\mathbf t_b}.
\]

---

# 14. Relative airflow

If the world wind velocity is \(\mathbf v_w\), the air-relative velocity of a point is

\[
\boxed{
\mathbf v_{\rm rel}
=
\mathbf v_{\rm point}-\mathbf v_w.
}
\]

Its magnitude and flow direction are

\[
V=\|\mathbf v_{\rm rel}\|,
\qquad
\boxed{\hat{\mathbf f}=\frac{\mathbf v_{\rm rel}}{V}}.
\]

The dynamic pressure is

\[
\boxed{
q_\infty=\frac12\rho V^2.
}
\]

**Origin:** standard incompressible aerodynamic dynamic-pressure definition. Compressibility/Mach-number corrections are not included.

---

# 15. Point/axial-body aerodynamic force

For the arrow point or another slender-body point, the code uses coefficient-based drag:

\[
\boxed{
\mathbf F_D
=
-q_\infty C_D A\,\hat{\mathbf f}.
}
\]

Here

- \(C_D\) = supplied drag coefficient;
- \(A\) = supplied reference area;
- \(q_\infty\) = dynamic pressure.

**Origin:** standard aerodynamic drag equation.

If a restoring normal-force model is enabled, the component of velocity transverse to the arrow axis is

\[
\boxed{
\mathbf v_\perp
=
\mathbf v_{\rm rel}
-(\mathbf v_{\rm rel}\cdot\hat{\mathbf e}_x)\hat{\mathbf e}_x.
}
\]

Its magnitude gives an angle of attack

\[
\boxed{
\alpha
=
\arcsin\left(
\frac{\|\mathbf v_\perp\|}{V}
\right).
}
\]

The code limits \(\alpha\) to the supplied stall angle.

A linear normal-force coefficient is assumed:

\[
\boxed{
C_N=C_{N\alpha}\alpha.
}
\]

The restoring normal force is

\[
\boxed{
\mathbf F_N
=
-q_\infty C_{N\alpha}\alpha A
\frac{\mathbf v_\perp}{\|\mathbf v_\perp\|}.
}
\]

The total force is

\[
\boxed{\mathbf F=\mathbf F_D+\mathbf F_N}.
\]

**Approximation:** this is a linear coefficient model with a hard angle-of-attack limit, not CFD or a measured full aerodynamic polar.

---

# 16. Shaft aerodynamic drag

The shaft is divided into `shaft_drag_segments` segments. For a segment of length \(ds\) and diameter \(d\), the velocity perpendicular to the shaft tangent is

\[
\boxed{
\mathbf v_\perp
=
\mathbf v_{\rm rel}
-(\mathbf v_{\rm rel}\cdot\mathbf t)\mathbf t.
}
\]

The perpendicular dynamic pressure is

\[
\boxed{
q_\perp=\frac12\rho\|\mathbf v_\perp\|^2.
}
\]

The cross-flow drag is

\[
\boxed{
\mathbf F_{\rm cross}
=
-q_\perp C_{D,\rm cross}\,d\,ds\,
\frac{\mathbf v_\perp}{\|\mathbf v_\perp\|}.
}
\]

The code also includes a much smaller axial term. With

\[
v_\parallel=\mathbf v_{\rm rel}\cdot\mathbf t,
\]

\[
q_\parallel=\frac12\rho v_\parallel^2,
\]

and cylindrical surface area

\[
A_\parallel=\pi d\,ds,
\]

the axial force is

\[
\boxed{
\mathbf F_{\rm axial}
=
-q_\parallel C_{D,\rm axial}A_\parallel
\operatorname{sign}(v_\parallel)\mathbf t.
}
\]

Thus

\[
\boxed{
\mathbf F_{\rm shaft}
=
\mathbf F_{\rm cross}+\mathbf F_{\rm axial}.
}
\]

**Approximation:** the shaft is represented as independent short cylindrical segments with supplied constant \(C_D\). Reynolds-number dependence is not calculated; the stored air-viscosity parameter is unused.

---

# 17. Feather aerodynamics

Each feather is represented at its centroid.

For \(N_f\) feathers, their radial clocking angle is

\[
\boxed{
a_k=\frac{2\pi k}{N_f}.
}
\]

A radial direction is constructed from the body transverse axes. Feather cant rotates this direction about the arrow axis.

The aerodynamic drag on each feather is

\[
\boxed{
\mathbf F_{D,f}
=
-q_\infty C_{D,f}A_f\hat{\mathbf f}.
}
\]

The feather normal is projected into the plane perpendicular to the airflow:

\[
\boxed{
\mathbf n_\perp
=
\mathbf n-(\mathbf n\cdot\hat{\mathbf f})\hat{\mathbf f}.
}
\]

The feather angle of attack is

\[
\boxed{
\alpha_f
=
\arcsin(\mathbf{\hat f}\cdot\mathbf n)
}
\]

and is limited to

\[
|\alpha_f|\le\alpha_{\rm stall}.
\]

The linear lift/normal coefficient is

\[
\boxed{
C_L=C_{\alpha}\alpha_f.
}
\]

The restoring force is

\[
\boxed{
\mathbf F_{L,f}
=
-q_\infty C_\alpha\alpha_f A_f
\frac{\mathbf n_\perp}{\|\mathbf n_\perp\|}.
}
\]

Thus

\[
\boxed{
\mathbf F_f=\mathbf F_{D,f}+\mathbf F_{L,f}.
}
\]

The force is applied at the feather centroid.

**Approximation:** feather planform geometry is compressed into the supplied area, drag coefficient, lift slope, cant, and stall angle. The feather height is not used in the aerodynamic calculation.

---

# 18. Translational rigid-body dynamics

All aerodynamic forces are summed with gravity:

\[
\boxed{
\mathbf F_{\rm total}
=
\mathbf F_g
+
\sum\mathbf F_{\rm shaft}
+
\mathbf F_{\rm point}
+
\sum\mathbf F_{\rm feather}.
}
\]

Newton's second law gives

\[
\boxed{
m\dot{\mathbf v}
=
\mathbf F_{\rm total}
}
\]

or

\[
\boxed{
\dot{\mathbf v}
=
\frac{\mathbf F_{\rm total}}{m}.
}
\]

Position follows from

\[
\boxed{
\dot{\mathbf r}=\mathbf v.
}
\]

Gravity is the constant world-frame force

\[
\boxed{
\mathbf F_g=(0,-mg,0).
}
\]

---

# 19. Aerodynamic moments and rigid-body rotation

For every force \(\mathbf F_i\) applied at body-frame position \(\mathbf r_i\) relative to the CM, the body-frame moment is

\[
\boxed{
\mathbf M_i
=
\mathbf r_i\times(R^T\mathbf F_i).
}
\]

The total body moment is

\[
\boxed{
\mathbf M=\sum_i\mathbf M_i.
}
\]

The inertia tensor is approximated as diagonal:

\[
\boxed{
I=
\begin{bmatrix}
I_x&0&0\\
0&I_y&0\\
0&0&I_z
\end{bmatrix}.
}
\]

Rigid-body rotation obeys Euler's equations:

\[
\boxed{
I\dot{\boldsymbol\omega}
+
\boldsymbol\omega\times(I\boldsymbol\omega)
=
\mathbf M.
}
\]

Therefore the implemented equation is

\[
\boxed{
\dot{\boldsymbol\omega}
=
I^{-1}
\left[
\mathbf M-\boldsymbol\omega\times(I\boldsymbol\omega)
\right].
}
\]

**Origin:** Euler's equations for a rigid body in body-fixed coordinates.

---

# 20. Quaternion orientation

The arrow attitude is represented by a unit quaternion

\[
q=(q_x,q_y,q_z,q_w).
\]

For body angular velocity

\[
\boldsymbol\omega=(\omega_x,\omega_y,\omega_z),
\]

the implemented quaternion kinematic equation is

\[
\boxed{
\dot q=\frac12\,q\otimes(0,\boldsymbol\omega)
}
\]

with the explicit component form

\[
\boxed{
\begin{aligned}
\dot q_x&=\frac12(q_w\omega_x+q_y\omega_z-q_z\omega_y),\\
\dot q_y&=\frac12(q_w\omega_y+q_z\omega_x-q_x\omega_z),\\
\dot q_z&=\frac12(q_w\omega_z+q_x\omega_y-q_y\omega_x),\\
\dot q_w&=-\frac12(q_x\omega_x+q_y\omega_y+q_z\omega_z).
\end{aligned}
}
\]

**Origin:** standard quaternion rigid-body kinematics, using the SciPy `[x,y,z,w]` convention.

The quaternion is normalized when used to form the rotation matrix.

---

# 21. Flight bending equations

The two bending coordinates obey independent forced damped oscillator equations:

\[
\boxed{
m_{\rm modal}\ddot q_y
+
c_{\rm modal}\dot q_y
+
k_{\rm modal}q_y
=
Q_y
}
\]

and

\[
\boxed{
m_{\rm modal}\ddot q_z
+
c_{\rm modal}\dot q_z
+
k_{\rm modal}q_z
=
Q_z.
}
\]

Thus

\[
\boxed{
\ddot q_y
=
\frac{Q_y-c_{\rm modal}\dot q_y-k_{\rm modal}q_y}
{m_{\rm modal}}
}
\]

and

\[
\boxed{
\ddot q_z
=
\frac{Q_z-c_{\rm modal}\dot q_z-k_{\rm modal}q_z}
{m_{\rm modal}}.
}
\]

The generalized aerodynamic load is obtained by virtual work. For example,

\[
\boxed{
Q_y
=
\sum_i
(R^T\mathbf F_i)\cdot
\begin{bmatrix}0\\\phi(x_i)\\0\end{bmatrix}
}
\]

and similarly

\[
\boxed{
Q_z
=
\sum_i
(R^T\mathbf F_i)\cdot
\begin{bmatrix}0\\0\\\phi(x_i)\end{bmatrix}.
}
\]

**Origin:** generalized-coordinate/virtual-work formulation of a modal beam model.

---

# 22. Numerical flight integration

The flight equations are integrated with `scipy.integrate.solve_ivp`.

The default maximum step is

\[
\boxed{\Delta t_{\max}=0.001\ {\rm s}}
\]

with relative tolerance

\[
\boxed{\mathrm{rtol}=2\times10^{-8}}
\]

and absolute tolerance

\[
\boxed{\mathrm{atol}=10^{-10}}.
\]

The solver uses dense output so that target/checkpoint quantities can be evaluated at the exact event time rather than at the nearest stored integration step.

---

# 23. Target and ground conditions

The simulation can terminate when the arrow tip reaches the ground:

\[
\boxed{
y_{\rm tip}-y_{\rm ground}=0
}
\]

with a downward crossing.

A target at range \(X_T\) is treated as an event at

\[
\boxed{
x_{\rm tip}-X_T=0
}
\]

with a forward crossing.

At that exact event time the target error is

\[
\boxed{
\mathbf e_T
=
\begin{cases}
[x_{\rm tip}-X_T,\ y_{\rm tip}-Y_T],
&\text{if }Z_T\text{ is unconstrained},\\[4pt]
[x_{\rm tip}-X_T,\ y_{\rm tip}-Y_T,\ z_{\rm tip}-Z_T],
&\text{otherwise}.
\end{cases}
}
\]

The target is declared hit if

\[
\boxed{
\|\mathbf e_T\|\le r_T
}
\]

where \(r_T\) is the supplied target radius (default \(0.25\) m).

Checkpoints work analogously, but their event is only the tip's \(X\)-coordinate crossing; the Euclidean distance to the checkpoint point is then tested against the checkpoint radius.

---

# 24. Complete computational chain

The simulation therefore proceeds as follows.

### Inputs

Arrow:

\[
\{L,d_o,d_i,m_{\rm total},m_p,x_p,\text{spine},\text{feather data},C_D,\ldots\}.
\]

Bow:

\[
\{L_{\rm bow},S,d_{\rm draw},\text{bow type},
\eta_{\rm bow},\eta_{\rm nock},
k_y,k_z,c_y,c_z,m_{\rm string,eff},
\text{aim/release parameters},\ldots\}.
\]

Atmosphere:

\[
\{p,T,\mathbf v_{\rm wind}\}.
\]

Target/checkpoints:

\[
\{X_T,Y_T,Z_T,r_T,\ldots\}.
\]

### Derived arrow properties

\[
A_s,\ I_s,\ EI,\ m_s,\ x_{\rm CM},\ I_x,\ I_y,\ I_z,
\lambda_1,\phi(x),m_{\rm modal},k_{\rm modal},c_{\rm modal}.
\]

### Bow energy and force

\[
F_{\rm draw}(d)
\rightarrow
E_{\rm bow}=\int F_{\rm draw}\,dd
\rightarrow
F_{\rm string}(d).
\]

### Pre-release state

Solve

\[
(K+K_g)u=f_g
\]

with nock equality constraints and

\[
g\ge0,\quad\lambda\ge0,\quad g\lambda=0.
\]

### Launch

Integrate the coupled arrow/string system using the finite-difference beam model, gravity, geometric stiffness, damping, string drive, and unilateral riser contact.

Stop after complete riser clearance.

### Launch-to-flight conversion

From the final beam state obtain

\[
\mathbf r_{\rm CM},\
\mathbf v_{\rm CM},\
\boldsymbol\omega,\
\mathbf e_x,\
q_y^{(b)},q_z^{(b)},
\dot q_y^{(b)},\dot q_z^{(b)},\
q.
\]

### Flight

At each ODE evaluation:

1. reconstruct shaft/point/feather positions and velocities;
2. subtract wind;
3. calculate dynamic pressure \(q_\infty=\frac12\rho V^2\);
4. calculate shaft, point, and feather aerodynamic forces;
5. sum forces and moments;
6. calculate bending generalized forces;
7. solve

\[
m\dot{\mathbf v}=\mathbf F,
\]

\[
I\dot{\boldsymbol\omega}
=
\mathbf M-\boldsymbol\omega\times(I\boldsymbol\omega),
\]

\[
\dot q=\frac12q\otimes(0,\boldsymbol\omega),
\]

\[
m_{\rm modal}\ddot q_{y,z}
+c_{\rm modal}\dot q_{y,z}
+k_{\rm modal}q_{y,z}
=Q_{y,z}.
\]

### Final output

The code reports the CM and tip positions/velocities, arrow direction, orientation quaternion, angular velocity, bending amplitudes, target error/hit status, checkpoint crossings, and the reason/time at which flight terminated.

---

# 25. Principal approximations and limitations

The equations above are physically based, but several quantities are deliberately parameterized or reduced:

- **Bow:** no explicit limb finite-element dynamics; force–draw is measured or phenomenological.
- **Bow efficiency:** represented by a scalar energy/force multiplier.
- **String:** represented by one effective mass plus linear lateral/vertical spring–damper coordinates.
- **Arrow launch:** Euler–Bernoulli beam theory; shear deformation and detailed composite/shaft construction are not modeled.
- **Axial shaft deformation:** omitted; axial motion is a common nock/reference translation.
- **Launch compression:** axial compression is estimated from the mass distribution and used only through geometric stiffness.
- **Riser contact:** normal-only unilateral contact; friction is not mechanically applied.
- **Flight flexibility:** only the first free-free bending mode in each transverse direction is retained.
- **Aerodynamics:** constant user-supplied drag coefficients and linear lift/normal-force slopes; no CFD, Reynolds-number model, Mach-number correction, dynamic stall, or full aerodynamic coefficient tables.
- **Shaft drag:** approximated by independent cylindrical segments.
- **Feathers:** each feather is reduced to centroid force with prescribed area and coefficients.
- **Inertia:** the inertia tensor is diagonal; feather inertia and detailed mass distribution are simplified.
- **Air:** ideal-gas density from pressure and temperature; no atmospheric variation with altitude is included.
- **Numerics:** finite differences, lumped nodal masses, reduced modal coordinates, and numerical contact projection/stabilization are used.

The source code itself explicitly identifies several stored parameters as inactive in the mechanics, including bow limb mass, nock-fit clearance, contact friction, air viscosity, feather height, and an independent center-of-pressure fraction. They therefore must not be interpreted as contributing to the equations above.
