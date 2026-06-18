"""
Spoor A - firming step 3: G-SWEEP toward criticality, for realistic FC magnitude.

Baseline BOLD/gating FC was low (~0.006-0.022) because the FIC-tuned working point sits below
criticality. FC peaks as G approaches the stability edge (the critical transition). This sweep
re-tunes FIC (analytic + polish) at each G, keeps only the STABLE ones (baseline ~3 Hz), and
reports baseline FC + global fluctuation amplitude. The G with the highest stable FC is the
near-critical working point. Uses the fast base reduced-Wong-Wang (numba); no drugs here.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tvb.simulator.lab import *  # noqa

conn = connectivity.Connectivity.from_file()
conn.weights = conn.weights / conn.weights.max()
conn.speed = np.array([3.0])
conn.configure()
N = conn.number_of_regions
k_strength = conn.weights.sum(axis=1)
S_E_TARGET = 0.164
NSIG = 1e-4

def s_e_to_rate(s):
    return s / (100.0 * 0.000641 * (1.0 - s))

def build_sim(J_i, G, dt=1.0):
    m = models.ReducedWongWangExcInh()
    m.G = np.array([float(G)]); m.J_N = np.array([0.15])
    m.a_e = np.full(N, 310.0); m.J_i = np.asarray(J_i, float).reshape(N)
    m.variables_of_interest = ['S_e']
    ic = np.zeros((200, 2, N, 1)); ic[:, 0, :, 0] = 0.164; ic[:, 1, :, 0] = 0.04
    sim = simulator.Simulator(
        model=m, connectivity=conn, coupling=coupling.Linear(a=np.array([1.0])),
        integrator=integrators.HeunStochastic(dt=dt, noise=noise.Additive(nsig=np.array([NSIG]))),
        monitors=[monitors.TemporalAverage(period=10.0)], initial_conditions=ic)
    sim.configure()
    return sim

def run_sim(sim, T, discard_ms=0.0):
    t, d = sim.run(simulation_length=T)[0]
    S_e = d[:, 0, :, 0]
    return S_e[int(discard_ms / 10.0):] if discard_ms > 0 else S_e

def H_I_scalar(x):
    y = 615.0 * x - 177.0
    return y / (1.0 - np.exp(min(-0.087 * y, 50.0)))

def fic_for(G, refine=20, gain=3.0, cap=0.5, T=1500.0, discard=800.0):
    gI, tI = 0.001, 10.0
    wp, JN, We, Wi, Io = 1.4, 0.15, 1.0, 0.7, 0.382
    S_E = 0.164
    def g(S_I):
        return tI * gI * H_I_scalar(Wi * Io + JN * S_E - S_I) - S_I
    lo, hi = 1e-4, 0.20; glo = g(lo)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if glo * g(mid) <= 0: hi = mid
        else: lo = mid; glo = g(lo)
    S_I = 0.5 * (lo + hi)
    I_E_target = (125.0 - 8.0) / 310.0
    J_i = np.clip((We * Io + wp * JN * S_E + G * JN * S_E * k_strength - I_E_target) / S_I, 0.001, None)
    for _ in range(refine):
        sim = build_sim(J_i, G)
        m = run_sim(sim, T, discard_ms=discard).mean(0)
        err = m - S_E_TARGET
        if np.max(np.abs(err)) < 0.012: break
        J_i = np.clip(J_i + np.clip(gain * err, -cap, cap), 0.001, None)
    return J_i

def fc_stats(S_e):
    Sz = (S_e - S_e.mean(0)) / (S_e.std(0) + 1e-9)
    FC = np.corrcoef(Sz.T); iu = np.triu_indices(N, 1)
    glob = S_e.mean(axis=1)                          # global signal
    return FC[iu].mean(), FC[iu].std(), glob.std()    # mean FC, FC spread, metastability proxy

rows = []
for G in [0.04, 0.05, 0.06, 0.065, 0.07, 0.08, 0.10]:
    J_i = fic_for(G)
    S_e = run_sim(build_sim(J_i, G), 50000.0, discard_ms=15000.0)
    rate = s_e_to_rate(S_e.mean())
    mfc, sfc, meta = fc_stats(S_e)
    stable = rate < 6.0
    rows.append(dict(G=G, rE_Hz=round(rate, 2), meanFC=round(mfc, 4),
                     FCspread=round(sfc, 3), metastab=round(meta, 4),
                     stable="yes" if stable else "RUNAWAY"))
    print("  G=%.3f  r_e=%.2f Hz  meanFC=%.4f  metastab=%.4f  %s"
          % (G, rate, mfc, meta, "stable" if stable else "RUNAWAY"), flush=True)

df = pd.DataFrame(rows)
print("\n=== G-SWEEP (criticality / FC magnitude) ===")
print(df.to_string(index=False))
stable_df = df[df.stable == "yes"]
if len(stable_df):
    best = stable_df.loc[stable_df.meanFC.idxmax()]
    print("\n[working point] highest stable FC at G=%.3f -> meanFC=%.4f (r_e=%.2f Hz)"
          % (best.G, best.meanFC, best.rE_Hz))
    print("  (baseline FC at the original G=0.06 was ~0.006; this is the realistic-FC working point)")
df.to_csv(r"E:\BiniruProjects\psyche-sim\results_gsweep.csv", index=False)
print("Saved: results_gsweep.csv")
