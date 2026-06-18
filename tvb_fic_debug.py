"""Diagnose the analytic FIC: are k_i / J_i sane, is the IC applied, is the low FP stable?"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from tvb.simulator.lab import *  # noqa

conn = connectivity.Connectivity.from_file()
conn.weights = conn.weights / conn.weights.max()
conn.speed = np.array([3.0])
conn.configure()
N = conn.number_of_regions
k = conn.weights.sum(1)
print("k (node strength)  min=%.2f mean=%.2f max=%.2f" % (k.min(), k.mean(), k.max()))

def H_I(x):
    y = 615 * x - 177
    return y / (1 - np.exp(min(-0.087 * y, 50.0)))
S_E = 0.164
IEt = (125 - 8) / 310.0
def g(SI):
    return 10 * 0.001 * H_I(0.7 * 0.382 + 0.15 * S_E - SI) - SI
lo, hi = 1e-4, 0.2
glo = g(lo)
for _ in range(100):
    mid = .5 * (lo + hi)
    if glo * g(mid) <= 0:
        hi = mid
    else:
        lo = mid; glo = g(lo)
SI = .5 * (lo + hi)
print("S_I*=%.4f" % SI)
print("G-sweep: largest G where the low (3 Hz) fixed point stays STABLE = working point")
for GVAL in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15]:
    Ji = (1 * 0.382 + 1.4 * 0.15 * S_E + GVAL * 0.15 * S_E * k - IEt) / SI
    Ji = np.clip(Ji, 0.001, None)
    m = models.ReducedWongWangExcInh()
    m.G = np.array([GVAL]); m.J_N = np.array([0.15]); m.J_i = Ji
    ic = np.zeros((200, 2, N, 1)); ic[:, 0, :, 0] = 0.164; ic[:, 1, :, 0] = SI
    sim = simulator.Simulator(
        model=m, connectivity=conn,
        coupling=coupling.Linear(a=np.array([1.0])),
        integrator=integrators.HeunStochastic(dt=1.0, noise=noise.Additive(nsig=np.array([1e-4]))),
        monitors=[monitors.TemporalAverage(period=10.0)],
        initial_conditions=ic)
    sim.configure()
    (t, d), = sim.run(simulation_length=4000.0)
    Se = d[:, 0, :, 0]
    tail = Se[len(Se) // 2:]
    rate = (tail.mean()) / (100.0 * 0.000641 * (1.0 - tail.mean()))
    print("   G=%.2f  Ji(mean %.1f, max %.1f)  final <S_e>=%.3f  r_e~%.1f Hz  %s"
          % (GVAL, Ji.mean(), Ji.max(), tail.mean(), rate,
             "STABLE" if tail.mean() < 0.30 else "runaway"))
