"""Entry point: run the arrow simulation and view the result."""

from simulation import SimConfig, Simulation

USE_PG_VIEWER = True

def main() -> None:
    config = SimConfig()
    if USE_PG_VIEWER:
        config.diagnostic_step_skip = config.diagnostic_step_skip // 10

    result = Simulation(config).run()

    if USE_PG_VIEWER:
        from viewer_pg import PgViewer
        PgViewer(result).show()
    else:
        from viewer import Viewer
        Viewer(result).show()


if __name__ == "__main__":
    main()
