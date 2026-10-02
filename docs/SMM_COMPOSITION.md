# The data's age composition for the memo-18 skill moments

What `28_smm_moments.do` has to export so the Julia SMM can compute the 44 pooled S moments.
Decision 2026-10-01: the counts come from Stata, and `code/smm/moments.jl` refuses a target
file without them rather than approximating.

## Why the model needs them

Memo 18 §3 builds every pooled moment as a mixture over the ages it pools, weighted by the
**data's** composition:

    mu_B(y)    = sum_c w_c ybar_c
    Cov_B(y,z) = sum_c w_c [cov_c(y,z) + (ybar_c - mu_B(y)) (zbar_c - mu_B(z))]

The S1 rows already give the per-age N of the level sample, which fixes S3. Nothing in
`targets.toml` gives the composition of the other frames:
- the five-year pairs (S4, S5, S7);
- the input-observed sets (S6);
- the diary sets (S8 time);
- the money years (S8 money, S9);
- the odd/even-wave split that decides whether money is a single year or the mean of two (memo 18 D10).

The weights matter because these moments pool very different ages. For example,
`S6_corr_taup_LW_3_17` is −0.37 largely because parental time falls with age while LW rises.

## The twelve frames

Each frame is one set of observations, counted by completed age at assessment (level frames)
or by (base age, end age) (pair frames). For every pooled moment, its frame's counts summed
over the moment's bin must equal the moment's `n`. `check_composition` enforces this and
names the first mismatch.

| frame | kind | observations | must reproduce (`n`) |
|---|---|---|---|
| `O_LW` | level | child-waves with raw LW, waves 1997/2002/2007/2014, ages 3–17 | S1 per age; S3: 2366 / 629 / 708 / 1029 |
| `O_taup` | level | `O_LW` with active parental time at that wave | S6 taup: 2366 |
| `O_tauc` | level | `O_LW` with own study at that wave, ages 6–17 | S6 tauc: 2028 |
| `O_ep` | level + odd | `O_LW`, one-child families, money observed (odd waves: mean of adjacent even years) | S6 ep: 610 |
| `P_LW` | pair | LW at base age `a` in 1997 or 2002 and at `a2` five years later, `a2 ≤ 17` | S4: 405 / 555; S5: 960 / 405 / 555 |
| `P_taup` | pair | `P_LW` with parental time at the base wave | S7 taup: 960 / 405 / 555 |
| `P_tauc` | pair | `P_LW` with own study at the base wave, base 6–12 | S7 tauc: 769 / 225 / 544 |
| `P_ep` | pair + odd | `P_LW`, one-child, money at the base wave (odd = base 1997) | S7 ep: 203 / 95 / 108 |
| `D_taup` | level | diary child-waves with parental time (2019 included), ages 3–17 | S8 taup: 564 / 628 / 905 / 979 |
| `D_tauc` | level | diary child-waves with own study, ages 6–17 | S8 tauc: 616 / 893 / 968 |
| `E_ep` | level | one-child PSID even years 1998–2018 with money, ages 3–17 | S8 money: 478 / 326 / 478 / 1304 |
| `E_epY` | level | `E_ep` with parents' pre-tax labour income > 0 | S9: 2543 |

**Pairs.** Ages are completed ages at each interview, so "five years later" can give
`a2 − a` of 4, 5 or 6. Export the actual `(a, a2)` cells. The model reads each child's skill
at exactly those two ages.

**Odd waves.** For `O_ep` and `P_ep`, `n_odd` is the number of a cell's rows observed at an
odd CDS wave (1997, 2007), where money is the mean of the two adjacent even PSID years. The
model mirrors this: its money at such a row is the mean of ages `a−1` and `a+1`. At age 17
it uses age 16 alone, because the model has no money at 18. The data's single-year fallback,
used when only one adjacent year is observed, is not reproduced row by row. If that fallback
matters, export it as a third count.

## Format

### In `SMM_Constants.csv` (the decision)

One row per cell, `name,value`, using these names:

    comp_<FRAME>_<a>              level count at age a
    comp_<FRAME>_<a>_odd          level count at age a observed at an odd wave   (O_ep only)
    comp_<FRAME>_<a>_<a2>         pair count, base age a, end age a2
    comp_<FRAME>_<a>_<a2>_odd     pair count at an odd base wave                  (P_ep only)

For example, `comp_O_LW_3,44`, `comp_P_LW_3_8,61` or `comp_O_ep_5_odd,12`. Omit cells with a
zero count.

### In `targets.toml` (what Julia reads)

`tools/make_smm_targets.py` collects the `comp_` rows into one table per frame. Because the
tables come after the top-level keys, they must be written with the other `[tables]`:

```toml
[composition.O_LW]
age = [3, 4, 5, ...]
n   = [44, 121, 145, ...]

[composition.O_ep]
age   = [3, 4, ...]
n     = [...]
n_odd = [...]

[composition.P_LW]
a  = [3, 3, 4, ...]
a2 = [8, 9, 9, ...]
n  = [...]
```

A level frame uses `age`; a pair frame uses `a` and `a2`. `n_odd` is present exactly for
`O_ep` and `P_ep`. Every frame must be present.

A pass-through for the generator, to add to its writer:

```python
import re
def composition_tables(const):
    rows = {}
    for name, val in const.items():
        m = re.fullmatch(r"comp_([A-Za-z]+_[A-Za-z]+)_(\d+)(?:_(\d+))?(_odd)?", name)
        if not m:
            continue
        frame, a, a2, odd = m.group(1), int(m.group(2)), m.group(3), bool(m.group(4))
        key = (a, int(a2)) if a2 else (a,)
        cell = rows.setdefault(frame, {}).setdefault(key, [0, 0])
        cell[1 if odd else 0] = int(round(val))
    out = []
    for frame, cells in sorted(rows.items()):
        keys = sorted(cells)
        out.append(f"[composition.{frame}]")
        if len(keys[0]) == 2:
            out.append("a  = [" + ", ".join(str(k[0]) for k in keys) + "]")
            out.append("a2 = [" + ", ".join(str(k[1]) for k in keys) + "]")
        else:
            out.append("age = [" + ", ".join(str(k[0]) for k in keys) + "]")
        out.append("n   = [" + ", ".join(str(cells[k][0]) for k in keys) + "]")
        if frame in ("O_ep", "P_ep"):
            out.append("n_odd = [" + ", ".join(str(cells[k][1]) for k in keys) + "]")
        out.append("")
    return out
```

Here `n` is the cell's total, including its odd-wave rows. So the `comp_<FRAME>_<a>` row is
the total and `_odd` is a subset of it.

## Also not in the target file (the model does without them for now)

- **The p99 winsorisation cuts.** Money is capped at $47,004 and the S9 ratio at 0.443
  (memo 18 §8). Neither is in `targets.toml`, so the model applies no cap. Simulated money
  sits far below the money cap.
- **SDs.** The data use N−1, memo 19 §9. The model uses population moments, as memo 18 §3
  states. With these sample sizes the difference is below 0.1%.
