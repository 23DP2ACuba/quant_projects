import numpy as np
from LatentSpaceFeatures import LatentSpaceVol
from LatentSpaceModel import NMVMDistribution

class LatentVolHeston:
    def __init__(self, encoder, decoder, nmvm_dist, mu=0.0, dt=1/252):
        self.encoder = encoder
        self.decoder = decoder
        self.nmvm_dist = nmvm_dist
        self.mu = mu  
        self.dt = dt

    def simulate():
        return 