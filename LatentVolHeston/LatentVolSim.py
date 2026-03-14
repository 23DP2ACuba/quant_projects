import numpy as np
from LatentSpaceFeatures import LatentSpaceVol
from LatentSpaceModel import NMVMDistribution

class LatentVolSim:
  def __init__(self, model, n_paths, n_steps, T, mu):
    self.model = model
    self.n_paths = n_paths
    self.n_steps = n_steps
    self.dt = 1/T
    self.Z = np.random.normal(size=(n_paths, n_steps))
    self.mu = mu
    self.price_paths = np.zeros((n_paths, n_steps+1))
    self.v_t = model.sample(n_paths, n_steps)

  def simulate(self, S0, plot=False):
    self.price_paths[:, 0] = S0
    for t in range(2):
      x = np.log(self.price_paths[:, t])
      x_next = x + (self.mu - 0.5*self.v_t[:, t])*self.dt + np.sqrt(self.v_t[:, t])*self.Z[:, t] * np.sqrt(self.dt)
      print((self.mu - 0.5*self.v_t[:, t]))
      self.price_paths[:, t+1] = np.exp(x_next)
    return self.price_paths
