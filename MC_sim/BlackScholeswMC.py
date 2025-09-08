import numpy as np

class OptionPricing:
    def __init__(self, S0, E, T, rf, sigma, i):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma=sigma
        self.i = i
        
    def call_option(self):
        option_data = np.zeros([self.i, 2])
        rand = np.random.normal(0, 1, [1, self.i])
        sp = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma**2) 
                              + self.sigma*np.sqrt(self.T)*rand)
        option_data[:, 1] = sp - self.E

        aver = np.sum(np.amax(option_data, axis=1)) / float(self.i)
        
        return np.exp(-1*self.rf*self.T)*aver
    
    def put_option(self):
        option_data = np.zeros([self.i, 2])
        rand = np.random.normal(0, 1, [1, self.i])
        sp = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma**2) 
                              + self.sigma*np.sqrt(self.T)*rand)
        option_data[:, 1] = self.E - sp

        aver = np.sum(np.amax(option_data, axis=1)) / float(self.i)
        
        return np.exp(-1*self.rf*self.T)*aver

if __name__ == "__main__":
    model = OptionPricing(100, 100, 1, 0.05, 0.2, 10000)
    print(f"val of the call option: {model.call_option()})")
    print(f"val of the call option: {model.put_option()})")