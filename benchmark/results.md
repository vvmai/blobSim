# Benchmark Results

5/5 PASSED (295.6s) | 2026-02-17

| Test | Type | Prediction | Observed | Verdict |
| ---- | ---- | ---------- | -------- | ------- |
| B2a | Calibration | E[tau]=186.16 | mean=185.9 | Centered on prediction |
| SC-1 | Convergence | Extinction guaranteed | 10/10 extinct, mean step=33.5 | All extinct, conservation OK |
| SC-4 | Convergence | Boom then extinction | peak=900, extinct ~step 122 | Reproduction + extinction + conservation |
| SC-5 | Convergence | N*=105.9 | sparse ~133, dense ~134 | Converged (1-4% diff), within 50% of N* |
| SC-6 | Convergence | energy -> 20.0 | mean=20.0, var=0.0 | Saturated at max_energy |

## B2a: Mean Lifespan

50 trials, IC-1 with move_cost=0.5. Two-phase model predicts E[tau]=186.16.

Observed taus are structurally multiples of 5 (algebraic identity: tau = 1000 - 5*n_moves). Distribution: {180: 12, 185: 19, 190: 17, 195: 2}. Mean 185.9, within tolerance of prediction.

## SC-1: Guaranteed Extinction

10 trials, IC-2 (decay grid, 2 species, reproduction enabled). E_initial=810.0, T_max=8100.

All 10 trials reached extinction in 32-36 steps (mean 33.5) — far below T_max. Conservation verified at terminus for all trials. The fast extinction reflects efficient energy absorption from the 2.0-energy grid cells followed by rapid dissipation through metabolism and movement.

## SC-4: Reproduction then Extinction

5 trials, IC-3 (rich decay grid, 5 grazers). E_initial=9005.0.

All trials show identical pattern: population explodes to grid capacity (900 = 30x30 grid), then crashes to extinction around step 120-124. Peak population is grid-bound (900) rather than energy-bound (18010). Conservation residuals < 2e-11 across all steps — negligible floating-point noise.

## SC-5: Carrying Capacity Convergence

5 trials x 2 runs (sparse=10, dense=300), 2000 steps each. Analytical N*=105.9.

Both sparse and dense runs converge to ~130-136 (25% above N*). Relative difference between sparse and dense means: 0.4-3.6% — excellent convergence. The systematic upward bias from N* reflects mean-field approximation error: spatial correlations reduce effective dissipation (blobs cluster less than predicted, so fewer failed claims, so less wasted energy).

Slowest benchmark (273.8s) due to high per-step blob count over 2000 steps.

## SC-6: Single Blob Energy Steady-State

10 trials, IC-5 (50x50 regenerating grid, single walker). max_energy=20.0.

All 10 trials: blob alive at step 500, mean energy = 20.0 exactly, variance = 0.0. The blob saturates at max_energy within the first few steps (capacity 5.0 >> effective_cost 0.278) and stays pinned there. Zero variance confirms the energy cap is applied consistently.
