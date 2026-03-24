import numpy as np
from LatentSpaceFeatures import LatentSpaceVol
from LatentSpaceModel import NMVMDistribution

class LatentVolSimulation:
  def __init__(self, model, n_paths, n_steps, mu=0.3, T=1):
    self.model = model
    mu_est = self.model.params.mu
    self.r = self.model.params.r_f
    self.n_paths = n_paths
    self.n_steps = n_steps
    self.dt = T/self.n_steps
    self.mu = mu_est if mu_est != 0 and mu_est != np.inf else mu
    self.price_paths = np.zeros((n_paths, n_steps+1))

  def simulate(self, S0, plot=False, Q=False, v_max = 1.5, alpha = 0.9):
    self.v_t = self.model.sample(self.n_paths, self.n_steps, Q=Q)
    
    print("v_t stats:", self.v_t.mean(), np.median(self.v_t))
    self.Z = np.random.normal(size=(self.n_paths, self.n_steps))

    self.price_paths[:, 0] = S0
    drift = self.r if Q else self.mu
    alpha = 0.9
    for t in range(self.n_steps):
      v_t = alpha * self.v_t[:, t-1] + (1 - alpha) * self.v_t[:, t] 
      
      v_t = np.clip(self.v_t[:, t], 1e-6, v_max)

      x = np.log(self.price_paths[:, t])
      x_next = x + (drift - 0.5*v_t)*self.dt + np.sqrt(v_t)*self.Z[:, t] * np.sqrt(self.dt)
      self.price_paths[:, t+1] = np.exp(x_next)

    return self.price_paths

  def get_V (self, K, S0=None, plot=True):
    if not S0:
      S0 = self.model.params.S0

    paths = self.simulate(S0, Q=True)
    S_T = paths[:, -1]

    if plot:
      plt.plot(self.price_paths.T)
      plt.show()

    erT = np.exp(-self.r)
    V_0 = erT * np.mean(np.maximum(S_T-K, 0))

    return V_0

    erT = np.exp(-self.r * self.nmodel.params.T)
    V_0 = erT * np.mean(np.maximum(S_T-K, 0))

    return V_0
