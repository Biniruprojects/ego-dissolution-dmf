# Ego dissolution as a collapse of integration, not a surge of entropy

A whole-brain **dynamic mean-field (DMF)** model, built in [The Virtual Brain](https://www.thevirtualbrain.org/), of the **combined action of a 5-HT₂A agonist (LSD/DMT) and an NMDA antagonist (N₂O)** on cortical dynamics. The model asks whether the ego-dissolving "psychedelic + dissociative" state is dominated by a rise in local signal complexity (the *entropic brain*) or by a collapse of long-range functional integration.

**Result (rate-matched to a 3 Hz physiological set-point, n = 6 seeds, replicated on a second connectome, dose-dependent, and confirmed with the empirical Beliveau Cimbi-36 5-HT₂A PET map):**

> The combination produces a **robust, dose-dependent collapse of functional integration (≈ −56%)** with only a **small but reproducible increase in signal complexity (≈ +1%)**. It is **dissociation-dominated, not entropy-dominated** — the brain loses its capacity to *integrate*, while local richness is largely preserved.

| Condition (rate-matched) | Δ Lempel–Ziv complexity | Δ functional integration |
|---|---|---|
| 5-HT₂A (LSD/DMT) | +5.4% ± 0.5 (6/6 seeds) | −46% ± 14 (6/6) |
| N₂O (disinhibitory) | −4.6% ± 0.3 (0/6) | −18% ± 7 (6/6) |
| **Combination** | **+0.9% ± 0.5 (6/6)** | **−56% ± 12 (6/6)** |

Full write-up: [`PREPRINT-draft.md`](PREPRINT-draft.md). Development log (every step, including the dead-ends and a rejected artefact): [`FINDINGS.md`](FINDINGS.md).

## Why this repo exists

Computational results are only as trustworthy as the code behind them. Everything here is open so the model, the Feedback Inhibition Control (FIC) tuning, the rate-matching control, and the robustness checks can be inspected and reproduced.

## Method in one paragraph

Reduced Wong–Wang DMF (Deco et al. 2014) on the empirical 76-region Hagmann connectome with conduction delays. Each region is balanced to ~3 Hz by an **analytic Feedback Inhibition Control** (closed-form `J_i`, then a short stochastic polish). The two drugs are *separable* parameters: **5-HT₂A = excitatory gain** (`a_e`, weighted by the 5-HT₂A receptor map), **N₂O = interneuron-preferential NMDA block** (a separate E→I conductance `J_N_EI`, modelling disinhibition) plus reduced long-range coupling. Because both drugs shift firing rate — and complexity/connectivity co-vary with rate — every condition is **re-tuned (rate-matched) to 3 Hz** before read-out, isolating mechanism from a firing-rate confound.

## Reproduce

```bash
# main environment (model + analysis)
python -m venv tvb-env
tvb-env/Scripts/python -m pip install -r requirements.txt   # tvb-library 2.10, numpy, scipy, pandas, matplotlib, networkx

# the empirical 5-HT2A receptor map (separate env to avoid version clashes)
python -m venv nmaps-env
nmaps-env/Scripts/python -m pip install neuromaps nibabel
```

Key scripts (run with the `tvb-env` interpreter unless noted):

| Script | What it does |
|---|---|
| `whole_brain_tvb_emp.py` | DMF + analytic FIC on the empirical connectome (baseline working point) |
| `whole_brain_tvb_n2o.py` | faithful (disinhibitory) N₂O model — separate E→I conductance |
| `whole_brain_tvb_ratematch.py` | **the core result** — rate-matched conditions |
| `whole_brain_tvb_multiseed.py` | n = 6 seeds × different receptor maps (significance + map-robustness) |
| `whole_brain_tvb_dose.py` | dose–response |
| `whole_brain_tvb_conn2.py` | replication on a 192-region connectome |
| `whole_brain_tvb_gsweep.py` / `_fccalib.py` | criticality / FC-magnitude calibration |
| `hansen_fetch.py` *(nmaps-env)* | download + sample the real Beliveau Cimbi-36 5-HT₂A PET map |
| `whole_brain_tvb_realmap.py` | rate-matched run with the empirical receptor map |
| `make_figures.py` | publication figures from the result CSVs |

Earlier modelling generations (`whole_brain_combo.py` Hopf, `whole_brain_ei.py` Wilson–Cowan, `whole_brain_dmf.py` hand-rolled DMF) are kept for transparency; the development log explains why each was superseded.

## Data

The empirical 5-HT₂A map is the [¹¹C]Cimbi-36 atlas (Beliveau et al. 2017) via [netneurolab/hansen_receptors](https://github.com/netneurolab/hansen_receptors). The connectome is the TVB default (Hagmann-type). Result tables (`results_*.csv`) and figures (`*.png`) are included.

## Citation

Preprint in preparation (Zenodo). Until then, cite this repository and `PREPRINT-draft.md`.

## Note on authorship

Independent research by **Sören van Krunckelsven** (Biniru Projects). The modelling and analysis were carried out with substantial AI assistance under the author's direction; all scientific decisions — including rejecting a working-point artefact and the honest framing of a small effect — are the author's responsibility. The study was prompted by the author's own phenomenological report.

## License

MIT — see [`LICENSE`](LICENSE).
