"""
Spoor A - v3: Dynamic Mean Field (reduced Wong-Wang, Deco et al. 2014) + Feedback
Inhibition Control (FIC), on a whole-brain connectome. This is the model FINDINGS.md
pointed to from the start.

Why v3 (see FINDINGS.md): v2 (Wilson-Cowan, whole_brain_ei.py) FIXED the entropy axis --
the separable 5-HT2A gain knob reproduces the published LZc increase (+19%, validation
PASS). But a plain WC network has no realistic partially-integrated regime: as coupling
rises it jumps desynchronised (FC~0) -> frozen, skipping the band where baseline long-range
FC is substantial. So "collapsed integration" was not measurable. The Deco DMF+FIC is the
literature cure: FIC tunes each node's local inhibition J_i so excitatory firing clamps at
~3 Hz, holding the network at the edge of the transition where baseline FC is realistic AND
modulable -- exactly the regime the 5-HT2A whole-brain papers use.

Mechanisms (biophysically faithful + separable):
  * 5-HT2A agonism (LSD/DMT) -> GAIN of the excitatory response function H_E (scales slope
       a_E), density-weighted by the 5-HT2A map.  (Deco et al. 2018 serotonin-map method.)
  * NMDA block (N2O)         -> lower the NMDA conductance J_NMDA -> weakens BOTH local
       recurrent excitation (w_plus*J_NMDA*S_E) and long-range coupling (G*J_NMDA*C@S_E).
       This IS NMDA antagonism, the literal N2O target.

FIC is tuned at BASELINE and then held fixed under the acute drug perturbations (FIC is a
slow homeostatic process; an acute drug does not re-tune it).

Validation gate: single 5-HT2A-up must raise LZc (Schartner 2017) before the combo counts.

Connectome + 5-HT2A map still SYNTHETIC (POC). Hardening = empirical connectome (HCP/TVB) +
real 5-HT2A PET atlas (Beliveau/Hansen).
"""
import numpy as np, networkx as nx, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
N = 90

# --- synthetic small-world connectome (symmetric, no self-loops, scaled to [0,1]) ---
Gsw = nx.watts_strogatz_graph(N, k=8, p=0.3, seed=7)
C = nx.to_numpy_array(Gsw)
C *= rng.uniform(0.5, 1.5, C.shape)
C = (C + C.T) / 2
np.fill_diagonal(C, 0.0)
C /= C.max()

# --- synthetic 5-HT2A density map (assoc-cortex-like gradient, 0..1) ---
dens = np.linspace(0.1, 1.0, N)
rng.shuffle(dens)

# --- DMF (reduced Wong-Wang) parameters, Deco et al. 2014 ---
aE, bE, dE = 310.0, 125.0, 0.16        # excitatory transfer function
aI, bI, dI = 615.0, 177.0, 0.087       # inhibitory transfer function
gamma   = 0.641                        # NMDA kinetic (r in Hz)
tauE, tauI = 0.100, 0.010              # s (NMDA / GABA)
WE, WI  = 1.0, 0.7
I0      = 0.382                        # nA  background input
w_plus  = 1.4                          # local recurrent excitation
J_NMDA0 = 0.15                         # nA  NMDA conductance (baseline)
rE_target = 3.0                        # Hz  FIC clamps excitatory rate here
sigma   = 0.01                         # noise amplitude
dt      = 0.001                        # s
G       = 0.5                          # global coupling working point (low -> avoid runaway)

def phi(x, a, b, d):
    # f-I curve y/(1-exp(-d*y)); removable singularity at y=0 -> limit 1/d.
    y = a * x - b
    denom = 1.0 - np.exp(np.minimum(-d * y, 50.0))   # clip exponent -> no overflow when y<<0
    out = np.empty_like(y)
    small = np.abs(denom) < 1e-9
    out[~small] = y[~small] / denom[~small]
    out[small] = 1.0 / d
    return out

def run(J, gain=None, J_NMDA=J_NMDA0, T=60.0, rec_every=20, record=True, S_E0=None, S_I0=None):
    if gain is None:
        gain = np.ones(N)
    steps = int(T / dt)
    half = steps // 2                  # measure on second half (post-transient)
    S_E = (0.20 * np.ones(N)) if S_E0 is None else S_E0.copy()
    S_I = (0.15 * np.ones(N)) if S_I0 is None else S_I0.copy()
    sq = np.sqrt(dt)
    GJ = G * J_NMDA
    rE_sum = np.zeros(N); cnt = 0
    rec = []
    for t in range(steps):
        I_E = WE * I0 + w_plus * J_NMDA * S_E + GJ * (C @ S_E) - J * S_I
        I_I = WI * I0 + J_NMDA * S_E - S_I
        r_E = phi(I_E, aE * gain, bE, dE)
        r_I = phi(I_I, aI, bI, dI)
        S_E = S_E + dt * (-S_E / tauE + gamma * (1.0 - S_E) * r_E) + sq * sigma * rng.standard_normal(N)
        S_I = S_I + dt * (-S_I / tauI + r_I) + sq * sigma * rng.standard_normal(N)
        np.clip(S_E, 0.0, 1.0, out=S_E)
        np.clip(S_I, 0.0, 1.0, out=S_I)
        if t >= half:
            rE_sum += r_E; cnt += 1
            if record and (t % rec_every == 0):
                rec.append(S_E.copy())
    S_rec = np.array(rec) if record else None
    return S_rec, rE_sum / max(cnt, 1), S_E, S_I

def fic_tune(iters=150, lr=0.3, cap=0.15, T=0.5):
    # FIC approached from the ACTIVE branch: start with LOW inhibition (E firing high), then
    # raise J slowly to bring the rate DOWN to target. State is carried over between steps
    # (no reset), so the network stays on the active branch and never collapses into the
    # silent 0 Hz down-state -- the reduced Wong-Wang node is bistable, and a cold restart
    # at high J falls straight into that trap.
    J = 0.5 * np.ones(N)
    S_E = 0.30 * np.ones(N)
    S_I = 0.15 * np.ones(N)
    it = 0
    for it in range(iters):
        _, rE, S_E, S_I = run(J, T=T, record=False, S_E0=S_E, S_I0=S_I)
        err = rE - rE_target
        J = np.clip(J + np.clip(lr * err, -cap, cap), 0.0, None)
        if np.max(np.abs(err)) < 0.25:
            break
    _, rE, S_E, S_I = run(J, T=T, record=False, S_E0=S_E, S_I0=S_I)
    print("[FIC] stopped after %d iters: mean r_E = %.2f Hz  (max node dev %.2f Hz)"
          % (it + 1, rE.mean(), np.max(np.abs(rE - rE_target))))
    return J, S_E, S_I

# ---------- metrics ----------
def lz76(b):
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

def metrics(S):
    Sc = S[::5]                                   # coarse-grain for LZ
    lzc = np.mean([lz76(Sc[:, j] > np.median(Sc[:, j])) for j in range(N)])
    Sz = (S - S.mean(0)) / (S.std(0) + 1e-9)
    FC = np.corrcoef(Sz.T)
    iu = np.triu_indices(N, 1)
    return lzc, FC[iu].mean(), FC

# ---------- tune FIC at baseline, then run conditions with J held fixed ----------
print("[setup] tuning Feedback Inhibition Control at baseline (G=%.1f)..." % G)
J_fic, S_E0, S_I0 = fic_tune()

conds = {
    "Baseline":           dict(gain_s=0.0, nmda_block=0.00),
    "LSD/DMT (5-HT2A up)": dict(gain_s=0.5, nmda_block=0.00),
    "N2O (NMDA block)":   dict(gain_s=0.0, nmda_block=0.50),
    "COMBO (both)":       dict(gain_s=0.5, nmda_block=0.50),
}

rows = []
FCs = {}
for name, p in conds.items():
    gain = 1.0 + p["gain_s"] * dens
    J_NMDA = J_NMDA0 * (1.0 - p["nmda_block"])
    # start each condition from the warm baseline state (acute drug from baseline)
    S, rE, _, _ = run(J_fic, gain=gain, J_NMDA=J_NMDA, S_E0=S_E0, S_I0=S_I0)
    lzc, integ, FC = metrics(S)
    FCs[name] = FC
    rows.append(dict(Condition=name, LZc=round(lzc, 4), Integration=round(integ, 4),
                     rE_Hz=round(float(rE.mean()), 2)))

df = pd.DataFrame(rows)
df["dLZc_%"] = (100 * (df.LZc / df.LZc[0] - 1)).round(1)
df["dInteg_%"] = (100 * (df.Integration / df.Integration[0] - 1)).round(1)
print(df.to_string(index=False))

# ---------- validation gate + combo read ----------
lsd = df[df.Condition.str.startswith("LSD")].iloc[0]
ok = lsd["dLZc_%"] > 0
print("\n[validation] single 5-HT2A-up vs baseline: dLZc = %s %%  -> %s" % (
    lsd["dLZc_%"],
    "PASS (entropy up, matches Schartner 2017)" if ok
    else "FAIL (entropy not up -- do NOT trust combo)"))
base = df.iloc[0]
combo = df[df.Condition.str.startswith("COMBO")].iloc[0]
print("[baseline]   integration = %.3f  (must be substantially > 0 for 'collapse' to mean anything)"
      % base["Integration"])
print("[combo]      dLZc = %s %% , dInteg = %s %%" % (combo["dLZc_%"], combo["dInteg_%"]))
hit = (combo["dLZc_%"] > 0) and (combo["dInteg_%"] < 0) and (base["Integration"] > 0.05)
print("[hypothesis] high LZc + collapsed integration: %s" % ("SUPPORTED" if hit else "not (yet) supported"))

df.to_csv(r"E:\BiniruProjects\psyche-sim\results_dmf.csv", index=False)

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(6, 5))
for _, r in df.iterrows():
    ax.scatter(r.Integration, r.LZc, s=120)
    ax.annotate(r.Condition, (r.Integration, r.LZc), fontsize=8,
                xytext=(6, 6), textcoords="offset points")
ax.set_xlabel("FC integration  (dissociation <-)")
ax.set_ylabel("Lempel-Ziv complexity  (entropy ->)")
ax.set_title("DMF+FIC (v3) - where the COMBO lands")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(r"E:\BiniruProjects\psyche-sim\combo_map_dmf.png", dpi=130)
print("Saved: results_dmf.csv + combo_map_dmf.png")
