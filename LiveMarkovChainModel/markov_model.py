from utils import Theme
import numpy as np
class MarkovRegime(Theme):
    def __init__(self, states=3):
        self.n_states = states
        self.current_state = 0
        self.colors = self.REG_CLRS
        self.bg_colors = self.REG_BG_CLRS
        self.state_probs = np.array([1/3, 1/3, 1/3])
        self.transition_mtx = np.array([
            [.9, .08, .02],
            [.1, .8, .1],
            [.02, .08, .9]
        ])
        self.emission_means = np.array([.0005, .0002,.005])
        self.emission_stds = np.array([.0005, .002, .003])
        
    
    def gaussian_likelyhood(self, vol, regime):
        pass
    
    def calibrate_model(self, hist_bars):
        if len(hist_bars) < 20:
            return
        
        vols = np.array(
            [
                (b["h"]-b["l"]) 
                / b["c"] if b["c"] > 0 else 0  
                for b in hist_bars
            ]
        )[vols>0]

        if len(vols) < 20:
            return
        
        p33, p67 = np.percentile(vols, 33), np.percentile(vols, 67)
        regime_assignments = np.zeros(len(vols), dtype=int)
        regime_assignments[vols >= p33] = 1
        regime_assignments[vols >= p67] = 2
        
        for reg in range(self.n_states):
            regime_vols = vols[regime_assignments == reg]
            if len(regime_vols) >= 3:
                self.emission_means[reg] = np.mean(regime_vols)
                self.emission_stds[reg] = max(np.std(regime_vols), 1e-6)
        
        sorted_idx = np.argsort(self.emission_means)
        self.emission_means = self.emission_means[sorted_idx]
        self.emission_stds = self.emission_stds[sorted_idx]
        
        transition_counts = np.zeros((self.n_states, self.n_states))
        for t in range(1, len(regime_assignments)):
            prev = regime_assignments[t-1]
            curr = regime_assignments[t]
            
            transition_counts[prev, curr] += 1
        
        a = 0.1
        for  i in range(self.n_states):
            row_sum = transition_counts[i].sum()
            if row_sum > 0:
                self.transition_mtx[i] = (transition_counts[i] + a) / row_sum + a*self.n_states

        self.state_probs = np.array([1/3, 1/3, 1/3])
        print(f"Calibrated emission means: {self.emission_means}")
        print(f"Calibrated emission vars: {self.emission_stds}")
            
            
            
    def get_regime(self):
        pass