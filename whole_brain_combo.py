"""
Proof-of-concept: whole-brain Hopf model of a dual-mechanism psychedelic state.

Hypothesis (from lived report): the LSD/DMT + N2O combination produces a signature
that neither drug reaches alone — HIGH neural complexity (5-HT2A-driven entropy)
together with COLLAPSED integration (NMDA-block-driven dissociation). That double
signature = the brain can no longer compute the self/world boundary ("ego death /
soul leaving the body").

Model: N Stuart-Landau (Hopf) oscillators on a small-world connectome.
  dz_j/dt = (a_j + i*w_j - |z_j|^2) z_j + G * sum_k C_jk (z_k - z_j) + beta*noise

Two pharmacological knobs:
  * 5-HT2A gain  -> raises local excitability a_j, weighted by a 5-HT2A density map
                   (proxy for LSD/DMT pushing cortex toward criticality).
  * NMDA block   -> lowers global coupling G (proxy for N2O dissociation).

Readouts: Lempel-Ziv complexity (LZc), FC integration, metastability.

NOTE: connectome + 5-HT2A map are SYNTHETIC here (POC). The real preprint needs an
empirical connectome (e.g. HCP / TVB default) and a real 5-HT2A PET density map.
"""
import numpy as np, networkx as nx, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
N = 90                         # regions (AAL-like count)
T, dt = 240.0, 0.01            # model time / step
steps = int(T/dt)
burn = int(0.25*steps)         # discard transient

# --- synthetic small-world connectome (symmetric, weight-normalised) ---
G_sw = nx.watts_strogatz_graph(N, k=8, p=0.3, seed=7)
C = nx.to_numpy_array(G_sw)
C *= rng.uniform(0.5, 1.5, C.shape)      # weight heterogeneity
C = (C + C.T) / 2
C /= C.sum(axis=1, keepdims=True).clip(min=1e-9)   # row-normalise (mean input ~1)

# --- synthetic 5-HT2A density map: assoc-cortex-like gradient, 0..1 ---
dens = np.linspace(0.1, 1.0, N)
rng.shuffle(dens)

w = 2*np.pi*(0.05 + 0.01*rng.standard_normal(N))   # intrinsic freqs (BOLD-ish)

def simulate(a0=-0.05, g5ht2a=0.0, G0=0.6, nmda_block=0.0, beta=0.03):
    # Mechanism (corrected):
    #  * 5-HT2A agonism = increased neural GAIN -> amplified intrinsic fluctuations
    #    (the "entropic brain": more diverse spontaneous activity), density-weighted.
    #  * NMDA block (N2O) = reduced global coupling -> dissociation / disintegration.
    a = a0*np.ones(N)                         # stay near criticality
    beta_j = beta*(1.0 + g5ht2a*dens)         # 5-HT2A -> stronger stochastic drive
    G = G0*(1.0 - nmda_block)
    z = 0.1*(rng.standard_normal(N) + 1j*rng.standard_normal(N))
    X = np.empty((steps, N))
    sq = np.sqrt(dt)
    for t in range(steps):
        coup = G*(C @ z - z*C.sum(axis=1))           # diffusive coupling
        dz = (a + 1j*w - np.abs(z)**2)*z + coup
        z = z + dt*dz + sq*beta_j*(rng.standard_normal(N) + 1j*rng.standard_normal(N))
        X[t] = z.real
    return X[burn:], z

# ---------- metrics ----------
def lz76(b):
    # Lempel-Ziv (1976) complexity of a binary sequence, normalised.
    s = ''.join('1' if v else '0' for v in b)
    i, c, l, n = 0, 1, 1, len(s)
    k, kmax = 1, 1
    while True:
        if s[i+k-1] == s[l+k-1]:
            k += 1
            if l+k > n:
                c += 1; break
        else:
            kmax = max(kmax, k)
            i += 1
            if i == l:
                c += 1; l += kmax;
                if l+1 > n: break
                i = 0; k = 1; kmax = 1
            else:
                k = 1
    norm = n/np.log2(n)
    return c/norm

def metrics(X):
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    # complexity: mean per-node LZc on median-binarised signal.
    # coarse-grain (every 20th sample) -> capture slow-envelope dynamics, keep LZ fast.
    Xc = X[::20]
    lzc = np.mean([lz76(Xc[:, j] > np.median(Xc[:, j])) for j in range(N)])
    # integration: mean upper-triangular functional connectivity
    FC = np.corrcoef(Xz.T)
    iu = np.triu_indices(N, 1)
    integ = FC[iu].mean()
    return lzc, integ, FC

conds = {
    "Baseline":              dict(g5ht2a=0.0, nmda_block=0.00),
    "LSD/DMT (5-HT2A up)":    dict(g5ht2a=1.6, nmda_block=0.00),
    "N2O (NMDA block)":      dict(g5ht2a=0.0, nmda_block=0.65),
    "COMBO (both)":          dict(g5ht2a=1.6, nmda_block=0.65),
}

rows = []
FCs = {}
for name, p in conds.items():
    X, _ = simulate(**p)
    lzc, integ, FC = metrics(X)
    FCs[name] = FC
    rows.append(dict(Condition=name, LZc=round(lzc,4), Integration=round(integ,4)))

df = pd.DataFrame(rows)
df["dLZc_%"]   = (100*(df.LZc/df.LZc[0]-1)).round(1)
df["dInteg_%"] = (100*(df.Integration/df.Integration[0]-1)).round(1)
print(df.to_string(index=False))
df.to_csv(r"E:\BiniruProjects\psyche-sim\results.csv", index=False)

# ---------- figure: the LZc x Integration plane ----------
fig, ax = plt.subplots(figsize=(6,5))
for _, r in df.iterrows():
    ax.scatter(r.Integration, r.LZc, s=120)
    ax.annotate(r.Condition, (r.Integration, r.LZc), fontsize=8,
                xytext=(6,6), textcoords="offset points")
ax.set_xlabel("FC integration  (dissociation ←)")
ax.set_ylabel("Lempel-Ziv complexity (entropy →)")
ax.set_title("Dual-mechanism map: where the COMBO lands")
ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(r"E:\BiniruProjects\psyche-sim\combo_map.png", dpi=130)
print("\nSaved: results.csv + combo_map.png")
