# Preserved nine-parameter baseline

The accepted run `2026-09-06_183119` is the baseline as of 7 September 2026.
Its original outputs are tracked under [`output/smm_runs/2026-09-06_183119`](../output/smm_runs/2026-09-06_183119).
They retain their original estimates, bounds, logs and metadata without rewriting them.
[`Input/parent_baseline_9param.toml`](../Input/parent_baseline_9param.toml) records the
full-precision estimates recovered from the final checkpoint, original boxes, fixed
parameters, target checksum and SHA-256 checksums of every original run file.

`PARENT_DEFAULTS` now uses these nine fitted values. The model constructor, SMM incumbent
seed, and notebook code that reads `PARENT_DEFAULTS` therefore start from the fitted
baseline. `R_1=0`, `sigma_4_1=0.02`, `mu_1=-0.04`, and the nine-parameter specification
are retained. Full-precision checkpoint values are used rather than the eight-decimal
presentation in `estimates.toml`.

| Parameter | Fitted baseline (rounded for display) | Original bounds | Future search bounds |
|---|---:|---|---|
| `phi_2` | 0.14372194 | [0.01, 20] | [0.01, 20] |
| `phi_3` | 1.02014337 | [0.05, 20] | [0.05, 20] |
| `lambda_2` | 8.67925673 | [0.05, 20] | [0.05, 20] |
| `R_0` | 50.60319558 | [0.5, 100] | [0.5, 100] |
| `sigma_1_0` | −0.22324257 | [−4, −0.2] | **[−4, −0.1]** |
| `sigma_1_1` | −0.14134183 | [−0.2, 0.05] | [−0.2, 0.05] |
| `sigma_2_0` | −3.75506167 | [−5, −0.5] | [−5, −0.5] |
| `sigma_2_1` | −0.04998108 | [−0.05, 0.05] | **[−0.1, 0.05]** |
| `sigma_4_0` | −5.98521880 | [−6, −1] | **[−8, −1]** |

The three expanded limits give near-bound parameters room for future joint searches.
They are exploration choices, not evidence that the expanded box improves the fit, nor
new estimates. The original nine-parameter result still belongs to its original box.
Existing feasibility checks reject combinations with parental-time or money elasticities
at or above one. Asset/HC grid ranges, node counts, random seed and moment definitions
are unchanged. Start a fresh run for the new bounds; do not resume the preserved run.

The Jacobian's optional candidate boxes are also corrected: `sigma_4_1` uses
[−0.05, 0.15] to include the inspected slope range, and `mu_1` uses [−0.08, −0.005]
to keep `mu_t=1+mu_1*(t−5)` between zero and one at ages 6–17. These are diagnostic
candidate ranges, not additions to `SMM_PARAMS`. Conditioning numbers using old boxes
must not be compared as though the column scales were unchanged.

## Boundary derivatives

`jacobian.jl` uses central finite differences when both evaluations fit inside the box.
Near a boundary it uses a second-order forward/backward stencil at the requested point;
it no longer clips one side and divides by the unchanged `2h`. Actual search-coordinate
points, derivative weights, schemes and evaluation counts are saved per step. Invalid or
non-finite perturbed simulations abort the Jacobian instead of entering an inference file.
The inference issues documented in REVIEW_TRIAGE A7 are separate and remain open.

## Validation and interpretation

From the repository root:

```sh
julia --threads=1 --project=. tools/test_smm_boundaries.jl
julia --threads=1 --project=. tools/test_smm_baseline.jl
```

The first tests exact linear/quadratic derivatives, level/log search coordinates, boundary
metadata, invalid input rejection and Jacobian integration. The second checks snapshot
integrity, all nine defaults, fixed parameters, feasibility and the full-grid fit at the
original `simN=2000`, seed 1234. The reference objective is **0.2500261422642604**, with
zero invalid cells and two households above the asset ceiling at the handoff.

The [run inspection](../output/smm_diagnostics/2026-09-06_183119/inspection_notes.md)
remains the interpretation record: study-time misfit dominates, the asset tail needs
numerical sensitivity checks, and fitted-point identification/inference is unfinished.
The original `run_record.toml` contains a malformed child-grid entry and a completion-time
code stamp; those known provenance limitations are retained and described in the inspection,
not silently repaired in the archived evidence. Promoting the fit does not remove them.
