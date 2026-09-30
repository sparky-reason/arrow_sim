"""Entry point: run the arrow simulation and view the result."""

from simulation import SimConfig, Simulation
from viewer import Viewer


def main() -> None:
    config = SimConfig()
    result = Simulation(config).run()
    Viewer(result).show()


if __name__ == "__main__":
    main()
