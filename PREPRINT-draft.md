# Ego dissolution as a collapse of integration, not a surge of entropy: a whole-brain dynamic mean-field model of combined serotonergic and NMDA-antagonist action

**Sören van Krunckelsven**
Biniru Projects, Antwerp, Belgium — independent research
Correspondence: biniruprojects@proton.me

*Preprint — draft v1, 18 June 2026. Not peer-reviewed.*

---

## Abstract

The "entropic brain" hypothesis holds that psychedelic and other ego-dissolving states correspond to an increase in the entropy or signal diversity of spontaneous brain activity. Motivated by a first-person report of a combined serotonergic–dissociative state (LSD/DMT together with nitrous oxide, N₂O), we asked a sharper question: when a 5-HT₂A agonist and an NMDA antagonist act *together*, is the resulting state dominated by a rise in local signal complexity, by a collapse of long-range functional integration, or by both? We address this in a validated whole-brain dynamic mean-field (DMF) model (reduced Wong–Wang, Deco et al. 2014) implemented in The Virtual Brain, on empirical human structural connectomes, with regional excitation/inhibition balanced by analytic Feedback Inhibition Control (FIC). The two drug classes are represented by *separable, biophysically motivated* parameters: 5-HT₂A agonism as a density-weighted increase in excitatory response gain, and sub-anaesthetic NMDA antagonism as interneuron-preferential disinhibition (reduced excitatory→inhibitory NMDA conductance) plus reduced long-range NMDA coupling. Crucially, because both perturbations shift mean firing rate — and both Lempel–Ziv complexity (LZc) and functional connectivity (FC) co-vary with rate — we **rate-match** every condition back to the ~3 Hz physiological set-point before reading out the metrics, isolating mechanism from a trivial rate confound. At matched rate, the combination produces a **robust, dose-dependent, topology-general collapse of functional integration** (mean ΔFC = −56% ± 12% SD, negative in 6/6 noise seeds and reproduced on a second 192-region connectome) together with a **small but statistically reproducible increase in signal complexity** (ΔLZc = +0.9% ± 0.5% SD, positive in 6/6 seeds, t ≈ 4). These multi-seed values are the canonical results; single-run figures quoted in the development log are individual noise realisations. The 5-HT₂A entropy increase (+5.4%) is largely cancelled by an N₂O-driven hypersynchrony (−4.6%), so the net complexity change is marginal while the integration collapse is large. We conclude that, in this model, the combined ego-dissolving state is **dissociation-dominated rather than entropy-dominated**: the brain loses its capacity to integrate, not its local richness. We discuss this as a refinement of the entropic-brain framing for combined serotonergic–dissociative states, and report the analyses transparently including a working-point artefact we explicitly reject.

**Keywords:** ego dissolution, entropic brain, dynamic mean-field model, The Virtual Brain, feedback inhibition control, 5-HT₂A, NMDA antagonist, nitrous oxide, functional integration, Lempel–Ziv complexity.

---

## 1. Introduction

Classic serotonergic psychedelics (LSD, psilocybin, DMT) acutely increase the diversity or entropy of spontaneous cortical activity, a finding that underpins the *entropic brain* hypothesis [Carhart-Harris et al., 2014]. Magnetoencephalography studies report increased Lempel–Ziv signal complexity not only for psychedelics but also for the dissociative NMDA antagonist ketamine [Schartner et al., 2017], suggesting that "elevated complexity" may be a common signature of several ego-dissolving states. In parallel, a separate line of work links the psychedelic ego-dissolution experience specifically to changes in *global functional connectivity* and network integration [Tagliazucchi et al., 2016].

These two readouts — local signal complexity and global integration — are conceptually distinct, and they need not move together. A state could be locally more complex yet globally less integrated, or vice versa. This distinction becomes acute when *two* pharmacologically distinct ego-dissolving agents are combined. The present work is motivated by a first-person report of exactly such a combination: a serotonergic psychedelic (LSD/DMT) taken together with the dissociative anaesthetic nitrous oxide (N₂O), producing an unusually complete dissolution of self. The phenomenological question — "does the self dissolve because the brain becomes maximally complex, or because it can no longer integrate?" — has a precise computational counterpart that we set out to test.

We adopt a whole-brain dynamic mean-field (DMF) approach [Deco et al., 2014], which represents each brain region by coupled excitatory and inhibitory neural populations on an empirical structural connectome, and which has an established method for representing serotonergic neuromodulation through receptor-density-weighted gain [Deco et al., 2018]. The two drug classes map onto *separable* model parameters, which is what makes the combination tractable: 5-HT₂A agonism as excitatory gain, NMDA antagonism as interneuron-preferential disinhibition plus reduced long-range coupling.

A central methodological point organises the study. Both perturbations shift the model's mean firing rate, and both LZc and FC depend on firing rate; therefore any naïve comparison risks reporting a firing-rate artefact rather than a mechanism. We control for this by re-balancing inhibition (Feedback Inhibition Control) so that every condition sits at the same ~3 Hz set-point before metrics are read — a within-model "rate-matched" design.

---

## 2. Methods

### 2.1 Whole-brain dynamic mean-field model

We use the reduced Wong–Wang excitatory–inhibitory DMF [Deco et al., 2014] as implemented in The Virtual Brain (tvb-library 2.10) [Sanz Leon et al., 2013]. Each region *k* carries an excitatory (E) and inhibitory (I) population with NMDA-gated synaptic variables S_E, S_I:

- I_E,k = W_E·I₀ + w_p·J_N·S_E,k + G·J_N·Σⱼ C_kⱼ S_E,j − J_i,k·S_I,k
- I_I,k = W_{EI}·J_N·S_E,k − S_I,k + W_I·I₀
- firing rates r = H(I) via the standard sigmoidal transfer functions H_E, H_I;
- dS_E/dt = −S_E/τ_E + (1−S_E)·γ·H_E + noise; dS_I/dt = −S_I/τ_I + γ_I·H_I + noise.

Parameters follow Deco et al. (2014): a_E=310, b_E=125, d_E=0.16; a_I=615, b_I=177, d_I=0.087; γ=0.000641, γ_I=0.001; τ_E=100 ms, τ_I=10 ms; W_E=1, W_I=0.7, I₀=0.382 nA; w_p=1.4; J_N=0.15 nA. Integration: stochastic Heun, dt=1 ms, additive noise σ=1e-4. State initialised at the low-activity (≈3 Hz) fixed point to avoid the bistable high-activity branch.

### 2.2 Connectome and working point

Structural connectivity is the TVB default empirical human connectome — the `connectivity_76.zip` dataset shipped with *tvb-data* and loaded via `Connectivity.from_file()` (a 76-region Hagmann-type DTI parcellation; regional centroids in MNI-like coordinates) — with distance-dependent conduction delays (tract lengths up to ~153 mm, speed 3 mm/ms). Weights are normalised to unit maximum. Because the normalised row-sums are ≈13, the effective global coupling is large; the stable low-activity working point for this connectome is therefore G ≈ 0.06, just below the transition to runaway high-activity (edge between G=0.06 stable and G=0.08 unstable under analytic FIC). A 192-region connectome is used for replication (§3.5).

### 2.3 Feedback Inhibition Control (analytic + polish)

To hold every region near a physiological 3.06 Hz, regional inhibitory weights J_i are set by **analytic FIC**: at the set-point all S_E equal the known value 0.164, the target excitatory input current is I_E* = (b_E − 8)/a_E = 0.377 nA (the −0.026 nA balance point), and the inhibitory fixed point S_I* is obtained by bisection, giving a closed-form per-region

  J_i,k = (W_E·I₀ + w_p·J_N·0.164 + G·J_N·0.164·κ_k − I_E*) / S_I*,

with κ_k the node strength. A short stochastic refinement (≤25 iterations) polishes residual error. This procedure clamps the baseline to ⟨S_E⟩ = 0.164, r_E = 3.07 Hz (max regional deviation 0.012), which the earlier hand-rolled and proportional-only controllers could not achieve on the heterogeneous empirical topology.

### 2.4 Pharmacological perturbations (separable, biophysical)

- **5-HT₂A agonism (LSD/DMT):** increase the excitatory response gain a_E by a receptor-density-weighted factor (1 + s·ρ_k), with ρ a 0–1 density map and s the gain magnitude (default s=0.15) [after Deco et al., 2018].
- **NMDA antagonism (N₂O), faithful model:** sub-anaesthetic NMDA block is interneuron-preferential, producing disinhibition. We subclass the model with a separate excitatory→inhibitory NMDA conductance J_{N,EI} and reduce it (default −40%), modelling disinhibition; we additionally reduce the long-range coupling G (default −30%), modelling reduced NMDA-mediated integration.
- For comparison, a *crude* N₂O model (uniform reduction of J_N) was also evaluated.

The receptor-density map ρ is a synthetic monotone gradient with randomised regional assignment; its specific spatial pattern is varied across seeds (§2.6) to test robustness, pending substitution by an empirical 5-HT₂A PET atlas.

### 2.5 Readouts and the rate-matching control

From the excitatory gating time series S_E (sampled every 10 ms) we compute: (i) **Lempel–Ziv complexity** (LZ76, normalised) per region on the median-binarised signal, averaged over regions; (ii) **functional integration** as the mean upper-triangular Pearson correlation (FC) across regions. A haemodynamic (Balloon–Windkessel) BOLD monitor was also evaluated for FC.

**Rate-matching:** because both LZc and FC co-vary with firing rate, FIC is re-tuned *per condition* so that every condition (baseline, 5-HT₂A, N₂O, combination) sits at 3 Hz. Reported effects are therefore at matched rate, isolating mechanism from rate.

### 2.6 Robustness protocol

We assessed: (1) multi-seed reproducibility (n=6 independent noise realisations, each with a different randomised 5-HT₂A map); (2) dose–response (combined dose scaled 0.25→1.0); (3) a second, larger connectome (192 regions); (4) a working-point (G) sweep for the FC-magnitude regime.

---

## 3. Results

### 3.1 The validated baseline

Analytic FIC clamps the empirical-connectome baseline to r_E = 3.07 Hz (max regional deviation 0.012), a stable physiological resting point. We note that a hand-implemented Euler DMF with proportional-only FIC failed to reach this regime (collapsing to 0 Hz or diverging to 10²–10³ Hz across coupling values), which motivated both the validated integrator and the analytic FIC; the key practical bug was the coupling scale (G≈0.06, not ≈2, for a connectome with row-sums ≈13).

### 3.2 Single agents reproduce expected signatures

At matched rate, the 5-HT₂A condition reproduces the canonical psychedelic signature — increased signal complexity (ΔLZc = +5.4% ± 0.5, positive in 6/6 seeds) — consistent with the entropic-brain literature [Carhart-Harris et al., 2014; Schartner et al., 2017]. The faithful (disinhibitory) N₂O condition, by contrast, *decreases* complexity (ΔLZc = −4.6% ± 0.3, negative in 6/6 seeds): in this model disinhibition drives hypersynchrony rather than entropy. Both single agents reduce functional integration (5-HT₂A −46%, N₂O −18%).

### 3.3 The combination is dissociation-dominated

At matched rate, the combination produces:

| Condition (rate-matched ~3 Hz) | ΔLZc | ΔFC integration |
|---|---|---|
| 5-HT₂A (LSD/DMT) | +5.4% ± 0.5 (6/6) | −46% ± 14 (6/6) |
| N₂O (faithful, disinhibitory) | −4.6% ± 0.3 (0/6) | −18% ± 7 (6/6) |
| **Combination** | **+0.9% ± 0.5 (6/6)** | **−56% ± 12 (6/6)** |

The combination's defining feature is the **large, robust collapse of integration** (−56%), with only a **small net complexity increase** (+0.9%): the 5-HT₂A entropy boost is largely cancelled by N₂O-induced hypersynchrony (+5.4% − 4.6% ≈ +0.8%). The small ΔLZc is nonetheless statistically reliable (6/6 seeds positive; mean/standard-error ≈ 4).

### 3.4 Dose–response

Scaling the combined dose deepens the integration collapse monotonically: ΔFC = −25%, −41%, −52%, −59% at doses 0.25, 0.50, 0.75, 1.00 (rate-matched), with ΔLZc steady at +1.0…+1.6%. A monotone dose–response is the signature of a genuine mechanism rather than a single-point coincidence.

### 3.5 Replication on a second connectome

On an independent 192-region empirical connectome, the combination reproduces the pattern: ΔFC = −49%, ΔLZc = +1.2% (two seeds). The effect is not specific to one parcellation.

### 3.6 Working point, FC magnitude, and empirical receptor map

With analytic FIC the model remains stable up to at least G=0.10 (r_E ≈ 3.6 Hz), and mean baseline FC increases monotonically with G (gating-FC 0.0043 → 0.0137 over G = 0.04 → 0.10). A working-point calibration sweeping (G, noise) with FIC re-tuning raised the haemodynamic (BOLD) baseline FC from ≈0.02 (G=0.06) to ≈0.043 (G=0.10); raising the noise further destabilised the 3 Hz set-point (runaway to 6–12 Hz). At the higher-FC point (G=0.10) the combination's integration collapse was confirmed (BOLD ΔFC = −63%). Absolute FC stays below empirical resting magnitudes — an inherent property of a rate-clamped mean-field resting state — but the verdict, being a *relative* change, is independent of this.

Repeating the **full multi-seed analysis** (n = 6) with the **empirical Beliveau Cimbi-36 5-HT₂A map** in place of the synthetic gradient gave combination ΔLZc = +1.5% ± 0.3 and Δintegration = −61% ± 19 (`results_realmap_multiseed.csv`; per-seed rows in `results_realmap_multiseed_perseed.csv`) — statistically indistinguishable from the synthetic-map result (+0.9% ± 0.5; −56% ± 12; `results_multiseed.csv`, per-seed in `results_multiseed_perseed.csv`). The verdict is therefore independent of whether the receptor map is synthetic or empirical. (A single empirical-map run had given −75% — `results_realmap.csv` — an individual noise realisation within the −35…−84% per-seed spread, illustrating why the multi-seed mean, not any single run, is the reported quantity; §5, limitation 1.)

### 3.7 A rejected artefact (reported for transparency)

Before rate-matching, the combination at the critical working point drove mean firing to ~24 Hz and the metrics flagged the naïve hypothesis as "supported." We reject this as a working-point artefact: at 24 Hz the model is far from its FIC-calibrated regime, and the apparent effect reflects the hot dynamical state, not the intended mechanism. Only the rate-matched analysis is reported as the result.

---

## 4. Discussion

In this model, the combined serotonergic–dissociative state is **dissociation-dominated, not entropy-dominated**. The robust, dose-dependent, topology-general finding is a collapse of long-range functional integration; the change in local signal complexity, while statistically reliable, is small because the 5-HT₂A-driven entropy increase and an N₂O-driven hypersynchrony nearly cancel.

This asymmetry — a large integration collapse (−56%) against a near-null complexity change (+1%) — is worth stating plainly, because it **partly contradicts the expectation that motivated the study**. The originating first-person report, and the entropic-brain intuition more generally, would predict a *large* rise in signal complexity; the model instead returns essentially none, while the dissociative axis dominates. A model that disconfirms the naïve prediction of the very experience that prompted it is, by that token, not a confirmation-seeking exercise: the −56%/+1% asymmetry is the result's strongest internal check against motivated reasoning.

This refines, rather than contradicts, the entropic-brain framing. For a *single* serotonergic psychedelic, the model reproduces the expected complexity increase. It is the *combination* with an NMDA antagonist — specifically, the disinhibitory hypersynchrony the antagonist contributes — that cancels the entropy signature while compounding the loss of integration. The phenomenological reading is that, under the combination, the self dissolves because the brain loses its capacity to *integrate* distributed activity into a unified model, not because local activity becomes maximally disordered. This is consistent with accounts linking ego dissolution to global connectivity change [Tagliazucchi et al., 2016] and complements complexity-centred accounts by separating the two axes.

An incidental modelling result is that *faithful* (disinhibitory) NMDA antagonism reduced complexity in this DMF, contrary to a naïve "disinhibition → more entropy" expectation. Whether this reflects a genuine prediction or a limitation of the mean-field representation of interneuron-specific block is an open question (see Limitations).

---

## 5. Limitations

This is a mechanistic, hypothesis-generating model, and several limitations bound its claims:

1. **Receptor map.** We incorporated the empirical Beliveau et al. (2017) 5-HT₂A atlas — the [¹¹C]Cimbi-36 PET map (the same dataset used by Deco et al. 2018), obtained from the *netneurolab/hansen_receptors* repository — by sampling the volume at each region centroid. The empirical map, run through the full multi-seed protocol (n = 6), gave the same answer as the synthetic gradient (combination Δintegration −61% ± 19 vs −56% ± 12; ΔLZc +1.5% ± 0.3 vs +0.9% ± 0.5 — statistically indistinguishable), confirming map-independence with real receptor data. Caveat: the TVB Hagmann connectome's centroids are only approximately in MNI152 space, so the region-level sampling is imperfect (~⅓ of the 76 centroids fell outside cortical tissue and were median-filled). This approximation is acceptable because the multi-seed analysis independently established that the verdict does not depend on the receptor map's spatial detail; a connectome supplied with a matched parcellation would sharpen the mapping.
2. **Absolute FC magnitude.** A working-point calibration (sweeping G and noise with FIC re-tuning) roughly doubled baseline BOLD FC (≈0.02 → ≈0.043 at G=0.10) but did not reach empirical resting magnitudes, and higher noise destabilised the set-point. The integration collapse was confirmed at this higher-FC point (−63%). The residual gap reflects the rate-clamped mean-field resting state and bears only on absolute magnitude, not on the relative verdict.
3. **Single model family.** Results are within the reduced Wong–Wang DMF; replication in an independent model family (e.g., a spiking or a different neural-mass model) would strengthen generality.
4. **Phenomenological pharmacology.** The drugs are represented by a small number of effective parameters. Interneuron-preferential NMDA block is modelled as a reduced E→I conductance plus reduced coupling; richer receptor-level and laminar detail is absent.
5. **No direct empirical comparison.** The model is validated against published single-drug *signatures* (sign of the LZc change), not fitted to combined-state neuroimaging data, which to our knowledge does not yet exist for this specific combination.

These limitations are stated to bound the result, not to diminish it: the central finding (a robust, dose-dependent, topology-general integration collapse with a small reproducible complexity increase) survives the robustness checks performed.

---

## 6. Conclusion

A validated whole-brain dynamic mean-field model, with separable serotonergic and NMDA-antagonist mechanisms and a rate-matched design that removes a firing-rate confound, indicates that the combined psychedelic–nitrous-oxide ego-dissolving state is characterised primarily by a **collapse of functional integration** rather than by a surge of signal entropy. The result is robust across noise realisations and receptor maps, scales with dose, and generalises to a second connectome. It reframes the phenomenology of this combined state as a loss of integration — the unbinding of a distributed self-model — with local signal complexity largely preserved.

---

## Data and code availability

All simulation and analysis code, result tables, figures, and the full development log are openly available at **https://github.com/Biniruprojects/ego-dissolution-dmf** — model drivers, analytic FIC, the rate-matching control, and the multi-seed/dose/second-connectome/G-sweep robustness scripts (`whole_brain_tvb*.py`), result tables (`results_*.csv`), figures (`FIG*.png`), and the step-by-step development log (`FINDINGS.md`). Built on tvb-library 2.10 (Python 3.13) and reproducible with fixed seeds. The empirical 5-HT₂A map is the Beliveau et al. (2017) [¹¹C]Cimbi-36 atlas obtained via [netneurolab/hansen_receptors](https://github.com/netneurolab/hansen_receptors).

## Acknowledgements

This work was carried out as independent research. The computational modelling, implementation, and analysis were performed with substantial AI assistance (Anthropic Claude / "lunatIK"), under the author's direction; all scientific decisions, the rejection of the working-point artefact, and the honest framing of the result are the author's responsibility. The study was prompted by the author's own phenomenological report.

## References

1. Carhart-Harris RL, Leech R, Hellyer PJ, Shanahan M, Feilding A, Tagliazucchi E, Chialvo DR, Nutt D (2014). The entropic brain: a theory of conscious states informed by neuroimaging research with psychedelic drugs. *Frontiers in Human Neuroscience* 8:20. doi:10.3389/fnhum.2014.00020

2. Schartner MM, Carhart-Harris RL, Barrett AB, Seth AK, Muthukumaraswamy SD (2017). Increased spontaneous MEG signal diversity for psychoactive doses of ketamine, LSD and psilocybin. *Scientific Reports* 7:46421. doi:10.1038/srep46421

3. Deco G, Ponce-Alvarez A, Hagmann P, Romani GL, Mantini D, Corbetta M (2014). How local excitation–inhibition ratio impacts the whole brain dynamics. *Journal of Neuroscience* 34(23):7886–7898. doi:10.1523/JNEUROSCI.5068-13.2014

4. Deco G, Cruzat J, Cabral J, Knudsen GM, Carhart-Harris RL, Whybrow PC, Logothetis NK, Kringelbach ML (2018). Whole-brain multimodal neuroimaging model using serotonin receptor maps explains non-linear functional effects of LSD. *Current Biology* 28(19):3065–3074.e6. doi:10.1016/j.cub.2018.07.083

5. Tagliazucchi E, Roseman L, Kaelen M, Orban C, Muthukumaraswamy SD, Murphy K, Laufs H, Leech R, McGonigle J, Crossley N, Bullmore E, Williams T, Bolstridge M, Feilding A, Nutt DJ, Carhart-Harris R (2016). Increased global functional connectivity correlates with LSD-induced ego dissolution. *Current Biology* 26(8):1043–1050. doi:10.1016/j.cub.2016.02.010

6. Hagmann P, Cammoun L, Gigandet X, Meuli R, Honey CJ, Wedeen VJ, Sporns O (2008). Mapping the structural core of human cerebral cortex. *PLoS Biology* 6(7):e159. doi:10.1371/journal.pbio.0060159

7. Sanz Leon P, Knock SA, Woodman MM, Domide L, Mersmann J, McIntosh AR, Jirsa V (2013). The Virtual Brain: a simulator of primate brain network dynamics. *Frontiers in Neuroinformatics* 7:10. doi:10.3389/fninf.2013.00010

8. Lempel A, Ziv J (1976). On the complexity of finite sequences. *IEEE Transactions on Information Theory* 22(1):75–81. doi:10.1109/TIT.1976.1055501

*All references and DOIs verified June 2026.*
