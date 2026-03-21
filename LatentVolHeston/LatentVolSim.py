import numpy as np
from LatentSpaceFeatures import LatentSpaceVol
from LatentSpaceModel import NMVMDistribution

class LatentVolSimulation:
  def __init__(self, model, n_paths, n_steps, T, mu=0.3):
    self.model = model
    mu_est = self.model.params.mu
    self.r = self.model.params.r_f
    self.n_paths = n_paths
    self.n_steps = n_steps
    self.dt = self.model.params.T/self.n_steps
    self.mu = mu_est if mu_est != 0 and mu_est != np.inf else mu
    self.price_paths = np.zeros((n_paths, n_steps+1))

  def simulate(self, S0, plot=False, Q=False):
    self.v_t = model.sample(self.n_paths, self.n_steps, Q=True)
    self.Z = np.random.normal(size=(self.n_paths, self.n_steps))
    print("v_t stats:", self.v_t.min(), self.v_t.max())
    self.price_paths[:, 0] = S0
    drift = self.mu if not Q else self.r
    for t in range(self.n_steps):
      x = np.log(self.price_paths[:, t])
      x_next = x + (drift - 0.5*self.v_t[:, t])*self.dt + np.sqrt(self.v_t[:, t])*self.Z[:, t] * np.sqrt(self.dt)
      self.price_paths[:, t+1] = np.exp(x_next)

  def get_V (self, K, S0=None, plot=True):
    if not S0:
      S0 = self.model.params.S0

    self.simulate(S0, Q=True)
    S_T = self.price_paths[:, -1]

    if plot:
      plt.plot(self.price_paths.T)
      plt.show()

    erT = np.exp(-self.r * self.nmodel.params.T)
    V_0 = erT * np.mean(np.maximum(S_T-K, 0))

    return V_0


