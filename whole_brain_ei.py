"""
Spoor A - v2: E/I neural-mass whole-brain model (genuinely separable knobs).

Why v2: the v1 Hopf model (whole_brain_combo.py) had only 2 knobs (local a, global G)
and could NOT dissociate 5-HT2A gain from NMDA coupling with the right signs -> a negative,
uninformative result (see FINDINGS.md). A 2-population Wilson-Cowan / DMF-style node has
SEPARATE biophysical targets, which is the documented fix:

  * 5-HT2A agonism (LSD/DMT) -> raises the GAIN (slope) of the excitatory (pyramidal)
       response function, weighted by a 5-HT2A density map.
       (method after Deco et al. 2018, "whole-brain model using serotonin receptor maps":
        receptor density scales the neuronal response gain of the E population.)
  * NMDA block (N2O)         -> reduces long-range coupling G -> dissociation / collapsed
       integration. (sub-anaesthetic NMDA hypofunction = loss of long-range integration.)

VALIDATION GATE (must pass before the combo is trusted):
  the single 5-HT2A-up condition must REPRODUCE the published psychedelic signature --
  Lempel-Ziv complexity UP (Schartner et al. 2017: LSD/psilocybin/ketamine all raise LZc).
  If that sign is wrong, the model is not trustworthy and we do NOT report the combo.

Connectome + 5-HT2A map are still SYNTHETIC (POC scaffold). The hardening pass swaps in an
empirical connectome (HCP/TVB) + a real 5-HT2A PET atlas (Beliveau/Hansen). Structure first,
data second.
"""
import numpy as np, networkx as nx, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
N = 90                         # regions (AAL-like count)
T, dt = 180.0, 0.01            # model time / step
steps = int(T / dt)
burn = int(0.25 * steps)       # discard transient
ds = 10                        # coarse-grain factor for LZ (slow envelope, keeps LZ fast)

# --- synthetic small-world connectome (symmetric, row-normalised, no self-loops) ---
Gsw = nx.watts_strogatz_graph(N, k=8, p=0.3, seed=7)
C = nx.to_numpy_array(Gsw)
C *= rng.uniform(0.5, 1.5, C.shape)        # weight heterogeneity
C = (C + C.T) / 2
np.fill_diagonal(C, 0.0)
C /= C.sum(axis=1, keepdims=True).clip(min=1e-9)   # row-normalise (mean input ~1)

# --- synthetic 5-HT2A density map: assoc-cortex-like gradient, 0..1 ---
dens = np.linspace(0.1, 1.0, N)
rng.shuffle(dens)

# --- Wilson-Cowan E/I node parameters (canonical oscillatory working point) ---
wEE, wEI = 16.0, 12.0          # E<-E, E<-I weights
wIE, wII = 15.0, 3.0           # I<-E, I<-I weights
aE0, thE = 1.3, 4.0            # E sigmoid slope / threshold
aI0, thI = 2.0, 3.7           # I sigmoid slope / threshold
P_E, P_I = 1.25, 0.0          # external drive
tauE, tauI = 1.0, 2.0

def sig(x, slope, th):
    return 1.0 / (1.0 + np.exp(-slope * (x - th)))

def simulate(g5ht2a=0.0, nmda_block=0.0, G0=0.8, sigma=0.15):
    # 5-HT2A: density-weighted gain (slope) on the E response function.
    slopeE = aE0 * (1.0 + g5ht2a * dens)
    # NMDA block (N2O): long-range coupling down.
    G = G0 * (1.0 - nmda_block)
    E = 0.1 * rng.random(N)
    I = 0.1 * rng.random(N)
    Xe = np.empty((steps, N))
    sq = np.sqrt(dt)
    for t in range(steps):
        inpE = wEE * E - wEI * I + G * (C @ E) + P_E
        inpI = wIE * E - wII * I + P_I
        E = E + dt * (-E + sig(inpE, slopeE, thE)) / tauE + sq * sigma * rng.standard_normal(N)
        I = I + dt * (-I + sig(inpI, aI0, thI)) / tauI + sq * sigma * rng.standard_normal(N)
        np.clip(E, 0.0, 1.0, out=E)
        np.clip(I, 0.0, 1.0, out=I)
        Xe[t] = E
    return Xe[burn:]

# ---------- metrics ----------
def lz76(b):
    # Lempel-Ziv (1976) complexity of a binary sequence, normalised.
    s = ''.join('1' if v else '0' for v in b)
    n = len(s)
    if n < 2:
        return 0.0
    i, c, l = 0, 1, 1
    k, kmax = 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            kmax = max(kmax, k)
            i += 1
            if i == l:
                c += 1
                l += kmax
                if l + 1 > n:
                    break
                i = 0
                k = 1
                kmax = 1
            else:
                k = 1
    return c / (n / np.log2(n))

def metrics(Xe):
    # complexity: mean per-node LZc on median-binarised, coarse-grained signal.
    Xc = Xe[::ds]
    lzc = np.mean([lz76(Xc[:, j] > np.median(Xc[:, j])) for j in range(N)])
    # integration: mean upper-triangular functional connectivity.
    Xz = (Xe - Xe.mean(0)) / (Xe.std(0) + 1e-9)
    FC = np.corrcoef(Xz.T)
    iu = np.triu_indices(N, 1)
    return lzc, FC[iu].mean(), FC, float(Xe.mean()), float(Xe.std())

# ---------- calibrate global coupling G0 to a realistic baseline integration ----------
# The resting brain HAS substantial long-range FC. Pick G0 so baseline FC ~ target, so that
# a "collapse of integration" is actually measurable. Selection is on baseline REALISM, NOT
# on the combo outcome -> this is working-point selection (as in DMF papers), not fitting
# to the answer.
TARGET_FC = 0.30
print("[calibrate] sweeping G0 for a realistic baseline integration:")
sweep = []
for g0 in [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 11.0, 15.0]:
    Xe = simulate(g5ht2a=0.0, nmda_block=0.0, G0=g0)
    _, integ, _, _, sE = metrics(Xe)
    sweep.append((g0, integ))
    print("    G0=%5.1f  baseline integration=%.4f  E_std=%.3f" % (g0, integ, sE))
G0_use = min(sweep, key=lambda gi: abs(gi[1] - TARGET_FC))[0]
print("-> chosen G0 = %.1f (baseline FC closest to %.2f)\n" % (G0_use, TARGET_FC))

conds = {
    "Baseline":           dict(g5ht2a=0.0, nmda_block=0.00),
    "LSD/DMT (5-HT2A up)": dict(g5ht2a=1.2, nmda_block=0.00),
    "N2O (NMDA block)":   dict(g5ht2a=0.0, nmda_block=0.60),
    "COMBO (both)":       dict(g5ht2a=1.2, nmda_block=0.60),
}

rows = []
FCs = {}
for name, p in conds.items():
    Xe = simulate(G0=G0_use, **p)
    lzc, integ, FC, mE, sE = metrics(Xe)
    FCs[name] = FC
    rows.append(dict(Condition=name, LZc=round(lzc, 4), Integration=round(integ, 4),
                     E_mean=round(mE, 3), E_std=round(sE, 3)))

df = pd.DataFrame(rows)
df["dLZc_%"] = (100 * (df.LZc / df.LZc[0] - 1)).round(1)
df["dInteg_%"] = (100 * (df.Integration / df.Integration[0] - 1)).round(1)
print(df.to_string(index=False))

# ---------- validation gate ----------
lsd = df[df.Condition.str.startswith("LSD")].iloc[0]
ok = lsd["dLZc_%"] > 0
print("\n[validation] single 5-HT2A-up vs baseline: dLZc = %s %%  -> %s" % (
    lsd["dLZc_%"],
    "PASS (entropy up, matches Schartner 2017)" if ok
    else "FAIL (entropy not up -- operating point/mechanism needs work; do NOT trust combo)"))
if ok:
    combo = df[df.Condition.str.startswith("COMBO")].iloc[0]
    print("[combo]      LZc dInteg = %s %% / %s %%  -> hypothesis = high LZc + collapsed integration"
          % (combo["dLZc_%"], combo["dInteg_%"]))

df.to_csv(r"E:\BiniruProjects\psyche-sim\results_ei.csv", index=False)

# ---------- figure: the LZc x Integration plane ----------
fig, ax = plt.subplots(figsize=(6, 5))
for _, r in df.iterrows():
    ax.scatter(r.Integration, r.LZc, s=120)
    ax.annotate(r.Condition, (r.Integration, r.LZc), fontsize=8,
                xytext=(6, 6), textcoords="offset points")
ax.set_xlabel("FC integration  (dissociation <-)")
ax.set_ylabel("Lempel-Ziv complexity  (entropy ->)")
ax.set_title("E/I model v2 - where the COMBO lands")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(r"E:\BiniruProjects\psyche-sim\combo_map_ei.png", dpi=130)
print("Saved: results_ei.csv + combo_map_ei.png")
