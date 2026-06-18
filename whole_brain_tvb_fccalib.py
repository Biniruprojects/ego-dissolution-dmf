"""
Spoor A - firming step 3b: FC-MAGNITUDE CALIBRATION. Find a near-critical working point where
baseline BOLD functional connectivity reaches the realistic resting range (~0.1-0.4), by jointly
raising global coupling G (toward the stability edge) and noise (larger global fluctuations) --
both now feasible because the analytic FIC keeps the 3 Hz set-point stable. Reports baseline
BOLD-FC + rate for each (G, noise), then re-runs the rate-matched COMBO at the best stable point
to confirm the dissociation verdict survives at realistic FC magnitude.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tvb.simulator.lab import *  # noqa
from tvb.basic.neotraits.api import NArray

conn = connectivity.Connectivity.from_file()
conn.weights = conn.weights / conn.weights.max()
conn.speed = np.array([3.0])
conn.configure()
N = conn.number_of_regions
k_strength = conn.weights.sum(axis=1)
S_E_TARGET = 0.164

def s_e_to_rate(s):
    return s / (100.0 * 0.000641 * (1.0 - s))

def H_I_scalar(x):
    y = 615.0 * x - 177.0
    return y / (1.0 - np.exp(min(-0.087 * y, 50.0)))

def analytic_Ji(G, a_e, J_N_EI=0.15):
    a_e = np.asarray(a_e, float).reshape(N)
    gI, tI = 0.001, 10.0
    wp, JN, We, Wi, Io = 1.4, 0.15, 1.0, 0.7, 0.382
    S_E = 0.164
    def g(S_I):
        return tI * gI * H_I_scalar(Wi * Io + J_N_EI * S_E - S_I) - S_I
    lo, hi = 1e-4, 0.20; glo = g(lo)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if glo * g(mid) <= 0: hi = mid
        else: lo = mid; glo = g(lo)
    S_I = 0.5 * (lo + hi)
    return np.clip((We * Io + wp * JN * S_E + G * JN * S_E * k_strength - (125.0 - 8.0) / a_e) / S_I, 0.001, None)

class N2OWongWang(models.ReducedWongWangExcInh):
    J_N_EI = NArray(default=np.array([0.15]))
    def dfun(self, x, c, local_coupling=0.0, **kwargs):
        S = x; ae = np.reshape(self.a_e, (-1, 1)); ji = np.reshape(self.J_i, (-1, 1))
        coup = self.G * self.J_N * (c[0] + local_coupling * S[0]); JNSe = self.J_N * S[0]
        x_e = self.w_p * JNSe - ji * S[1] + self.W_e * self.I_o + coup + self.I_ext
        x_e = ae * x_e - self.b_e; H_e = x_e / (1.0 - np.exp(-self.d_e * x_e))
        dS_e = -(S[0] / self.tau_e) + (1.0 - S[0]) * H_e * self.gamma_e
        x_i = self.J_N_EI * S[0] - S[1] + self.W_i * self.I_o + self.lamda * coup
        x_i = self.a_i * x_i - self.b_i; H_i = x_i / (1.0 - np.exp(-self.d_i * x_i))
        dS_i = -(S[1] / self.tau_i) + H_i * self.gamma_i
        return np.array([dS_e, dS_i])

def build(a_e, J_N_EI, J_i, G, nsig, use_bold=False, faithful=False):
    m = N2OWongWang() if faithful else models.ReducedWongWangExcInh()
    m.G = np.array([float(G)]); m.J_N = np.array([0.15])
    if faithful:
        m.J_N_EI = np.array([float(J_N_EI)])
    m.a_e = np.asarray(a_e, float).reshape(N); m.J_i = np.asarray(J_i, float).reshape(N)
    m.variables_of_interest = ['S_e']
    ic = np.zeros((200, 2, N, 1)); ic[:, 0, :, 0] = 0.164; ic[:, 1, :, 0] = 0.04
    mons = [monitors.TemporalAverage(period=10.0)]
    if use_bold:
        mons.append(monitors.Bold(period=2000.0))
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coupling.Linear(a=np.array([1.0])),
        integrator=integrators.HeunStochastic(dt=1.0, noise=noise.Additive(nsig=np.array([nsig]))),
        monitors=mons, initial_conditions=ic)
    sim.configure(); return sim

def polish(a_e, J_N_EI, G, nsig, faithful=False, refine=20):
    J_i = analytic_Ji(G, a_e, J_N_EI)
    for _ in range(refine):
        sim = build(a_e, J_N_EI, J_i, G, nsig, faithful=faithful)
        m = sim.run(simulation_length=1800.0)[0][1][:, 0, :, 0][80:].mean(0)
        err = m - S_E_TARGET
        if np.max(np.abs(err)) < 0.02: break
        J_i = np.clip(J_i + np.clip(3.0 * err, -0.5, 0.5), 0.001, None)
    return J_i

def bold_fc(sim, T=90000.0, discard_bold=15000.0):
    out = sim.run(simulation_length=T)
    Se = out[0][1][:, 0, :, 0]
    B = out[1][1][:, 0, :, 0][int(discard_bold / 2000.0):]
    Bz = (B - B.mean(0)) / (B.std(0) + 1e-9)
    FC = np.corrcoef(Bz.T); iu = np.triu_indices(N, 1)
    return float(FC[iu].mean()), s_e_to_rate(Se[1500:].mean())

# --- phase 1: baseline BOLD-FC over (G, noise) ---
print("[phase 1] baseline BOLD-FC vs (G, noise):", flush=True)
grid = [(0.06, 1e-4), (0.10, 1e-4), (0.10, 1e-3), (0.14, 1e-3), (0.10, 3e-3), (0.14, 3e-3)]
rows = []
best = None
for G, nz in grid:
    Ji = polish(np.full(N, 310.0), 0.15, G, nz)
    fc, rate = bold_fc(build(np.full(N, 310.0), 0.15, Ji, G, nz, use_bold=True))
    stable = rate < 6.0
    rows.append(dict(G=G, noise=nz, BOLD_FC=round(fc, 4), rE_Hz=round(rate, 2), stable="yes" if stable else "RUNAWAY"))
    print("   G=%.2f noise=%.0e  BOLD_FC=%.4f  r_e=%.2f Hz  %s" % (G, nz, fc, rate, "stable" if stable else "RUNAWAY"), flush=True)
    if stable and (best is None or fc > best[2]):
        best = (G, nz, fc)
pd.DataFrame(rows).to_csv(r"E:\BiniruProjects\psyche-sim\results_fccalib.csv", index=False)

# --- phase 2: confirm verdict at the best realistic-FC working point ---
if best:
    Gb, nzb, fcb = best
    print("\n[phase 2] best stable point G=%.2f noise=%.0e (baseline BOLD-FC=%.3f) -> rate-matched combo:" % (Gb, nzb, fcb), flush=True)
    dens = np.linspace(0.1, 1.0, N); np.random.default_rng(1).shuffle(dens)
    JNEI = 0.15 * 0.6; Gn = Gb * 0.7
    # baseline
    Jb = polish(np.full(N, 310.0), 0.15, Gb, nzb, faithful=True)
    fcb2, _ = bold_fc(build(np.full(N, 310.0), 0.15, Jb, Gb, nzb, use_bold=True, faithful=True))
    # combo (rate-matched at its own params)
    a_e = 310.0 * (1.0 + 0.15 * dens)
    Jc = polish(a_e, JNEI, Gn, nzb, faithful=True)
    fcc, rc = bold_fc(build(a_e, JNEI, Jc, Gn, nzb, use_bold=True, faithful=True))
    dfc = 100 * (fcc / fcb2 - 1)
    print("   baseline BOLD-FC=%.3f  COMBO BOLD-FC=%.3f  (r_e=%.2f Hz)  dFC=%+.0f%%" % (fcb2, fcc, rc, dfc), flush=True)
    print("   -> integration collapse at realistic-FC working point: %s" % ("CONFIRMED" if dfc < -10 else "weaker"))
print("Saved: results_fccalib.csv")
