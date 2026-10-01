"""Reference free-free bending-mode test for the arrow model.

This test reuses the production SimConfig and spine->Young's-modulus conversion,
but removes bow, string, contact, aerodynamics, and damping.  The rod is therefore
an unforced, free-free Cosserat rod with the same geometry, shaft density, 100-grain
tip mass, and 3.5 g nock/string-inertia mass as the production simulation.

The previous version initialized a transverse velocity field without initializing
the corresponding Cosserat director/angular-velocity field.  That is not a
consistent bending eigenmode for a shearable Cosserat rod and can excite unwanted
shear/director dynamics.  This version instead starts at a turning point:

    y(x,0) = A phi_1(x),   v(x,0) = 0,

and constructs the directors from the instantaneous centerline tangent.  Thus the
initial centerline and cross-sectional frames are kinematically consistent.  The
initial state is not assumed to be the exact loaded-endpoint-mass eigenmode; the
ring-down is projected onto the analytical first free-free Euler-Bernoulli shape,
and the spectral peaks are reported so that the actual first Cosserat-rod mode can
be identified.

Run from the same directory as simulation.py, for example:
    python free_free_mode_test.py --spine 400
    python free_free_mode_test.py --spine 1000

The simulation itself requires PyElastica, exactly as simulation.py does.
"""

from __future__ import annotations

import argparse
import numpy as np
import elastica as ea
from elastica.rod.cosserat_rod import CosseratRod
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper import integrate
from elastica.timestepper.symplectic_steppers import PositionVerlet

from simulation import SimConfig, youngs_modulus, GRAINS_TO_KG, SHEAR_MODULUS_DIVISOR


# First non-rigid free-free Euler-Bernoulli root.
LAMBDA_1 = 4.730040744862704


class FreeFreeTestSystem(ea.BaseSystemCollection, ea.CallBacks):
    pass


def first_free_free_shape(x: np.ndarray, length: float) -> np.ndarray:
    """First non-rigid free-free Euler-Bernoulli mode shape.

    The normalization is arbitrary.  It is normalized here by max(abs(phi))=1.
    """
    lam = LAMBDA_1
    z = lam * x / length
    sigma = (np.cosh(lam) - np.cos(lam)) / (np.sinh(lam) - np.sin(lam))
    phi = (np.cosh(z) + np.cos(z)) - sigma * (np.sinh(z) + np.sin(z))
    scale = np.max(np.abs(phi))
    if scale <= 0.0:
        raise RuntimeError("Failed to construct the analytical mode shape.")
    return phi / scale


def make_mode_consistent_state(
    x_nodes: np.ndarray,
    phi: np.ndarray,
    amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a small planar bending displacement and matching directors.

    PyElastica stores the director triad with rows corresponding to the local
    normal, binormal, and tangent directions.  We therefore construct, for each
    element, an orthonormal frame

        d1 = normal,
        d2 = binormal,
        d3 = tangent,

    from the deformed centerline.  The rod starts from rest, so no angular
    velocity is imposed.
    """
    y = amplitude * phi
    dy_dx = np.gradient(y, x_nodes, edge_order=2)

    # Deformed centerline.  The small amplitude keeps the geometric stretch
    # O(A^2) and therefore negligible for the linear modal measurement.
    position = np.vstack((x_nodes, y, np.zeros_like(x_nodes)))

    n_elems = len(x_nodes) - 1
    directors = np.zeros((3, 3, n_elems), dtype=float)

    for i in range(n_elems):
        dx = x_nodes[i + 1] - x_nodes[i]
        dy = y[i + 1] - y[i]
        tangent_norm = np.hypot(dx, dy)
        if tangent_norm <= 0.0:
            raise RuntimeError(f"Degenerate element {i} in initial mode shape.")

        t = np.array([dx, dy, 0.0]) / tangent_norm
        b = np.array([0.0, 0.0, 1.0])
        n = np.array([-t[1], t[0], 0.0])

        directors[0, :, i] = n
        directors[1, :, i] = b
        directors[2, :, i] = t

    return position, directors


class ModeCallback(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict, phi: np.ndarray):
        super().__init__()
        self.step_skip = step_skip
        self.callback_params = callback_params
        self.phi = phi

    def make_callback(self, system, time, current_step):
        if current_step % self.step_skip != 0:
            return

        y = system.position_collection[1, :]
        v_y = system.velocity_collection[1, :]
        masses = system.mass

        # Mass-weighted projection onto the analytical first free-free shape.
        # The additive rigid translation contributes only a DC component and is
        # removed in post-processing before the FFT.
        denominator = np.sum(masses * self.phi**2)
        q = np.sum(masses * self.phi * y) / denominator
        qdot = np.sum(masses * self.phi * v_y) / denominator

        self.callback_params["time"].append(time)
        self.callback_params["q"].append(q)
        self.callback_params["qdot"].append(qdot)
        self.callback_params["tip_y"].append(y[-1])
        self.callback_params["mid_y"].append(y[len(y) // 2])


def _quadratic_peak_frequency(freqs: np.ndarray, magnitude: np.ndarray, k: int) -> float:
    """Sub-bin frequency estimate using parabolic interpolation in log magnitude."""
    f_peak = float(freqs[k])
    if 1 <= k < len(freqs) - 1:
        y0 = np.log(magnitude[k - 1] + 1e-30)
        y1 = np.log(magnitude[k] + 1e-30)
        y2 = np.log(magnitude[k + 1] + 1e-30)
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-14:
            delta = 0.5 * (y0 - y2) / denom
            f_peak += delta * (freqs[1] - freqs[0])
    return f_peak


def spectral_peaks(
    times: np.ndarray,
    signal: np.ndarray,
    f_min: float = 10.0,
    f_max: float = 250.0,
    relative_threshold: float = 0.05,
    max_peaks: int = 8,
) -> list[tuple[float, float]]:
    """Return prominent local FFT peaks as (frequency_hz, relative_amplitude)."""
    if len(times) < 16:
        raise RuntimeError("Too few diagnostic samples for an FFT.")

    dt = float(np.mean(np.diff(times)))
    x = np.asarray(signal, dtype=float)

    # Remove a constant offset and linear numerical drift.
    t = times - times[0]
    x = x - np.polyval(np.polyfit(t, x, 1), t)

    window = np.hanning(len(x))
    spectrum = np.fft.rfft(x * window)
    magnitude = np.abs(spectrum)
    freqs = np.fft.rfftfreq(len(x), dt)

    band = (freqs >= f_min) & (freqs <= f_max)
    if not np.any(band):
        return []

    band_indices = np.flatnonzero(band)
    band_max = np.max(magnitude[band_indices])
    if band_max <= 0.0:
        return []

    candidates: list[tuple[float, float]] = []
    for k in band_indices[1:-1]:
        if magnitude[k] >= magnitude[k - 1] and magnitude[k] >= magnitude[k + 1]:
            if magnitude[k] >= relative_threshold * band_max:
                candidates.append(
                    (
                        _quadratic_peak_frequency(freqs, magnitude, int(k)),
                        float(magnitude[k] / band_max),
                    )
                )

    candidates.sort(key=lambda item: item[0])
    return candidates[:max_peaks]


def run_free_free_mode_test(
    cfg: SimConfig,
    final_time: float = 0.25,
    n_steps: int = 500_000,
    diagnostic_step_skip: int = 100,
    initial_amplitude: float = 1.0e-4,
) -> dict:
    """Run a small-amplitude free-free ring-down and estimate modal frequencies."""

    if initial_amplitude <= 0.0:
        raise ValueError("initial_amplitude must be positive.")
    if final_time <= 0.0 or n_steps < 100:
        raise ValueError("final_time must be positive and n_steps must be >= 100.")

    system = FreeFreeTestSystem()

    # Same geometry/material parameters as production simulation.
    start = np.array([-0.5 * cfg.arrow_length, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    E_modulus = youngs_modulus(cfg.spine_value, cfg.outer_radius)

    x_nodes = np.linspace(0.0, cfg.arrow_length, cfg.n_elements + 1)
    phi = first_free_free_shape(x_nodes, cfg.arrow_length)
    position_offset, directors = make_mode_consistent_state(
        x_nodes, phi, initial_amplitude
    )
    position_offset[0, :] += start[0]

    arrow = CosseratRod.straight_rod(
        n_elements=cfg.n_elements,
        start=start,
        direction=direction,
        normal=normal,
        base_length=cfg.arrow_length,
        base_radius=cfg.outer_radius,
        density=cfg.shaft_density,
        youngs_modulus=E_modulus,
        shear_modulus=E_modulus / SHEAR_MODULUS_DIVISOR,
        position=position_offset,
        directors=directors,
    )

    # Same concentrated masses as production simulation.  There is deliberately
    # no damping, external forcing, bow, contact, or aerodynamic load.
    arrow.mass[-1] += cfg.tip_weight_grains * GRAINS_TO_KG
    arrow.mass[0] += cfg.string_effective_mass

    system.append(arrow)

    # Turning-point initial condition: displacement is prescribed, velocities and
    # angular velocities remain zero.  Because the directors were constructed from
    # the centerline tangent, there is no artificial initial shear/director mismatch.
    arrow.velocity_collection[:] = 0.0
    arrow.omega_collection[:] = 0.0

    data = {"time": [], "q": [], "qdot": [], "tip_y": [], "mid_y": []}
    system.collect_diagnostics(arrow).using(
        ModeCallback,
        step_skip=diagnostic_step_skip,
        callback_params=data,
        phi=phi,
    )

    system.finalize()

    timestepper = PositionVerlet()
    integrate(timestepper, system, final_time, n_steps)

    times = np.asarray(data["time"])
    q = np.asarray(data["q"])

    peaks = spectral_peaks(times, q)
    if not peaks:
        raise RuntimeError(
            "No spectral peaks found. Increase final_time or initial_amplitude, "
            "or lower the spectral threshold."
        )

    # The ring-down is initialized close to the first mode, so the lowest
    # prominent peak is the appropriate first-bending candidate.  Also report all
    # prominent peaks to make mode identification transparent.
    frequency_hz = peaks[0][0]

    return {
        "spine": cfg.spine_value,
        "E_GPa": E_modulus / 1e9,
        "times": times,
        "q": q,
        "qdot": np.asarray(data["qdot"]),
        "tip_y": np.asarray(data["tip_y"]),
        "mid_y": np.asarray(data["mid_y"]),
        "frequency_hz": frequency_hz,
        "period_ms": 1000.0 / frequency_hz,
        "spectral_peaks": peaks,
        "total_mass_g": float(np.sum(arrow.mass) * 1000.0),
        "initial_amplitude_mm": float(initial_amplitude * 1000.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spine", type=float, default=1000.0)
    parser.add_argument("--final-time", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--skip", type=int, default=100)
    parser.add_argument(
        "--initial-amplitude",
        type=float,
        default=1.0e-4,
        help="Initial transverse displacement amplitude [m].",
    )
    args = parser.parse_args()

    cfg = SimConfig(spine_value=args.spine)
    result = run_free_free_mode_test(
        cfg,
        final_time=args.final_time,
        n_steps=args.steps,
        diagnostic_step_skip=args.skip,
        initial_amplitude=args.initial_amplitude,
    )

    print()
    print("=== Free-free first bending mode test ===")
    print(f"Spine:                    {result['spine']:.1f}")
    print(f"Equivalent Young E:       {result['E_GPa']:.6g} GPa")
    print(f"Total model mass:         {result['total_mass_g']:.6f} g")
    print(f"Initial amplitude:        {result['initial_amplitude_mm']:.6g} mm")
    print(f"First-mode frequency:     {result['frequency_hz']:.6g} Hz")
    print(f"First-mode period:        {result['period_ms']:.6g} ms")
    print("Prominent spectral peaks:")
    for frequency, relative_amplitude in result["spectral_peaks"]:
        print(f"  {frequency:10.6f} Hz   relative amplitude {relative_amplitude:.4f}")


if __name__ == "__main__":
    main()
