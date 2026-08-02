# Archive

Superseded work, kept for reference. **Nothing here is current.**
The live model is in [`Combined Models/Full model/`](../Combined%20Models/Full%20model/).

Archived 2026-08-02. Folder structure preserved as it was; git history is intact, so
`git log --follow <path>` still works across the move.

---

## `Combined Models/Full model/` — previous versions of the current model

Superseded by `transfer_CRRA_wage.ipynb`. Listed oldest → newest.

| File | What it was | Why superseded |
|---|---|---|
| `transfer_model.ipynb` | Base parent-child model. Log utility, i.i.d. transitory shocks. | Replaced by the AR(1) variant. |
| `transfer_model_AR1.ipynb` | Log utility, AR(1) wage shocks, flat tax. Child module: `ConSavLabor_college_AR1.jl`. | Parent utility is log, not CRRA; no `η`; flat proportional tax; wage is `w₀(1+αk)z` with parental human capital as a state — none of which match the current model spec. |
| `transfer_CRRA.ipynb` | CRRA parent utility + retirement. | Intermediate step; superseded by the `_wage` variant. |
| `transfer_CRRA_wage_ORIGINAL.ipynb` | **Byte-identical copy of `transfer_CRRA_wage.ipynb` before the 2026-08-02 extraction.** | Kept so the extraction can be verified. Diff its cells 8, 12, 14, 15, 16, 44, 49 against `src/parent_family.jl` — they should match verbatim, except that the driver code at the end of cell 49 was moved back into the notebook. |

`ConSavLabor_college_AR1.jl` was **not** archived — it remains in `Full model/` at your
request, though the current notebook includes `ConSavLabor_college_ret.jl` instead.

---

## `Combined Models/Archive/` — earlier combined models

| File | What it was |
|---|---|
| `transfer_model.ipynb` | Earlier transfer model. |
| `Parent_OLG.ipynb` | Overlapping-generations parent model. Abandoned approach. |
| `ConSavLabor_college.jl` | Early child lifecycle module. |
| `consavlabor.jl` | Early consumption-saving-labor module. |
| `issue.md` | Contemporaneous notes on modeling problems. |

## `Combined Models/Child & Parent part/` — intermediate combined models

| File | What it was |
|---|---|
| `Family_with_asset.ipynb` | Parent-child model with the regime switch at t=7. |
| `family_only_HC.ipynb` | Human-capital investment only, no assets. |
| `modeified_family_with_asset.ipynb` | Modified asset-holding variant. |

## `ConSavLabor/` — single-agent building blocks

`consumption_saving.ipynb` (basic T-period consumption-saving) ·
`ConSavLabor.ipynb` (adds endogenous labor + human capital) ·
`ConSavLabor_AR1.ipynb` (persistent AR(1) wage shock, Tauchen) ·
`ConSavLabor_stochastic.ipynb` (i.i.d. transitory shocks)

## `ConSavLabor_college/` — college-choice building blocks

`ConSavLabor_college.ipynb` (college vs. work at 18) ·
`ConSavLabor_college_AR1.ipynb` (adds AR(1) shocks) ·
`ConSavLabor_college_SE.ipynb` (heterogeneous beliefs about college returns) ·
`ConSavLabor_college_retire.ipynb` (adds a retirement phase)

These are the direct ancestors of `ConSavLabor_college_ret.jl`.

## `Family Model/` — early family models

`family.ipynb` (education and care decisions) ·
`parent_child_model.ipynb` (parent-child interaction)

## `test codes/` — scratch variants (never tracked in git)

Untracked before archiving, so these have **no git history**. Not duplicates:

- `ConSavLabor_college_AR1.jl` — differs from the `Full model/` copy by ~906 lines
- `transfer_model.ipynb` — a different file (13.8 MB, Jul 2025) from the tracked one (29.3 MB)
- `familyyyyyy.ipynb`, `new.ipynb`, `test.ipynb` — scratch

Kept because they are genuine variants rather than copies. Delete if you're confident
nothing in them is needed.

## `Thesis_code.ipynb`

Root-level notebook (7.4 MB, Jan 2026). Git-ignored, so no history.

---

## Reviving something

```bash
git mv "archive/<path>" "<destination>"     # for tracked files
mv "archive/<path>" "<destination>"         # for test codes/ and Thesis_code.ipynb
```

Check `git log --follow` first to see what changed before it was archived.
