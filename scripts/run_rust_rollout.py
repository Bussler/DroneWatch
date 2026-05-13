from __future__ import annotations

from dronewatch.sim import SwarmSimulation


def main() -> None:
    sim = SwarmSimulation(seed=42)

    while not sim.is_done():
        actions = [
            (1.0, 0.25) if index % 2 == 0 else (-0.25, 1.0)
            for index in range(sim.num_agents)
        ]
        sim.step(actions)

    metrics = sim.metrics()
    print("Rust scripted rollout complete")
    print(f"episode_length={metrics['timestep']}")
    print(f"discovered_targets={metrics['discovered_target_count']}/{metrics['target_count']}")
    print(f"target_discovery_rate={metrics['target_discovery_rate']:.3f}")
    print(f"coverage_ratio={metrics['coverage_ratio']:.3f}")
    print(f"collision_count={metrics['collision_count']}")
    print(f"obstacle_violation_count={metrics['obstacle_violation_count']}")
    print(f"connectivity_ratio={metrics['connectivity_ratio']:.3f}")
    print(f"average_communication_neighbors={metrics['average_communication_neighbors']:.3f}")


if __name__ == "__main__":
    main()