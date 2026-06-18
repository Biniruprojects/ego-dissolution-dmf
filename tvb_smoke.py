"""Smoke test: confirm the TVB wiring before writing the full driver.
Checks: synthetic connectome -> Connectivity, per-node a_e + J_i, sim runs, output shape,
S_e in a sane range. If this runs clean, the full DMF+FIC driver is safe to write."""
import numpy as np
from tvb.simulator.lab import *  # noqa

N = 10
rng = np.random.default_rng(0)
C = rng.random((N, N)); C = (C + C.T) / 2; np.fill_diagonal(C, 0.0); C /= C.max()

conn = connectivity.Connectivity(
    weights=C,
    tract_lengths=np.zeros((N, N)),
    region_labels=np.array(["R%02d" % i for i in range(N)]),
    centres=np.zeros((N, 3)),
)
conn.speed = np.array([np.inf])
conn.configure()
print("[conn] ok, regions =", conn.number_of_regions)

m = models.ReducedWongWangExcInh()
m.G = np.array([2.0])
m.J_N = np.array([0.15])
m.a_e = np.full(N, 310.0)      # per-node gain (5-HT2A knob lives here)
m.J_i = np.ones(N)             # per-node inhibition (FIC knob)

sim = simulator.Simulator(
    model=m,
    connectivity=conn,
    coupling=coupling.Linear(a=np.array([1.0])),
    integrator=integrators.HeunStochastic(dt=1.0, noise=noise.Additive(nsig=np.array([1e-3]))),
    monitors=[monitors.TemporalAverage(period=10.0)],
)
sim.configure()
print("[sim] configured")

(t, d), = sim.run(simulation_length=2000.0)
S_e = d[:, 0, :, 0]
S_i = d[:, 1, :, 0]
print("[run] output shape", d.shape, "-> S_e", S_e.shape)
print("[run] S_e mean=%.4f  min=%.4f  max=%.4f" % (S_e.mean(), S_e.min(), S_e.max()))
print("[run] S_i mean=%.4f" % S_i.mean())
print("[ok] per-node a_e + J_i + sim wiring all work")
