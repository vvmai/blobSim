"""End-to-end blob simulation example.

Runs 200 steps with RandomWalker + Grazer species on a 30x30 grid
with RegeneratingEnvironment. Validates energy conservation.

Usage:
    python main.py
"""
from blobsim import (
    ActionType,
    BlobAttributes,
    MOORE,
    RegeneratingEnvironment,
    SensingLevel,
    Simulation,
    SimulationConfig,
    SpeciesConfig,
)
from blobsim.species import GrazerRules, RandomWalkerRules


def main() -> None:
    """Run a 200-step blob simulation and validate conservation."""
    walker_attrs = BlobAttributes.create(
        species="walker",
        max_energy=10.0,
        offspring_energy=3.0,
        reproduction_threshold=7.0,
        base_metabolic_cost=0.1,
        move_cost=0.2,
        reproduce_cost=0.5,
        observation_radius=1,
        sensing_level=SensingLevel.BASIC,
        allowed_actions=frozenset(ActionType),
        allowed_directions=MOORE,
    )

    grazer_attrs = BlobAttributes.create(
        species="grazer",
        max_energy=10.0,
        offspring_energy=3.0,
        reproduction_threshold=7.0,
        base_metabolic_cost=0.1,
        move_cost=0.2,
        reproduce_cost=0.5,
        observation_radius=1,
        sensing_level=SensingLevel.ENERGY,
        allowed_actions=frozenset(ActionType),
        allowed_directions=MOORE,
    )

    config = SimulationConfig(
        grid_size=30,
        environment=RegeneratingEnvironment(
            rate=0.05, capacity=2.0, initial_cell_energy=1.0,
        ),
        species=[
            SpeciesConfig(
                rules_factory=RandomWalkerRules,
                attributes=walker_attrs,
                count=25,
                initial_energy=5.0,
            ),
            SpeciesConfig(
                rules_factory=GrazerRules,
                attributes=grazer_attrs,
                count=25,
                initial_energy=5.0,
            ),
        ],
        seed=42,
    )

    sim = Simulation(config)

    print("Running blob simulation (200 steps)...")
    print(f"Grid: {config.grid_size}x{config.grid_size}")
    print(f"Species: {[s.attributes.species for s in config.species]}")
    print(f"Initial population: {sum(s.count for s in config.species)}")
    print()

    for state in sim.iterate(steps=200):
        if state.step % 50 == 0:
            print(
                f"Step {state.step:>3d}: "
                f"alive={state.n_alive:>4d}  "
                f"births={state.n_births:>3d}  "
                f"deaths={state.n_deaths:>3d}  "
                f"E_total={state.e_blobs + state.e_grid:.2f}"
            )

    result = sim.result()
    recorder = result.recorder

    # Validate conservation (SC-6 compatible: uses if/raise, not assert)
    residuals = recorder.conservation_residuals()
    max_residual = float(residuals.max()) if len(residuals) > 0 else 0.0
    if max_residual >= 1e-10:
        raise RuntimeError(f"Conservation violated: max residual = {max_residual}")

    # Print final stats
    print()
    print("=== Final Stats ===")
    print(f"Max conservation residual: {max_residual:.2e}")
    print(f"Mean lifespan: {recorder.mean_lifespan():.1f} steps")

    pop = recorder.population_over_time()
    print(f"Final population: {int(pop[-1])}")

    species_pop = recorder.species_population()
    for sp_name, sp_pop in species_pop.items():
        print(f"  {sp_name}: {int(sp_pop[-1])}")

    # Total births and deaths from ledger records
    total_births = sum(ls.n_births for ls in recorder.ledger_records)
    total_deaths = sum(ls.n_deaths for ls in recorder.ledger_records)
    print(f"Total births: {total_births}")
    print(f"Total deaths: {total_deaths}")

    print()
    print("Simulation complete. All invariants held.")


if __name__ == "__main__":
    main()
