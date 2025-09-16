import numpy as np
import scipy.stats as si
#from ipywidgets import interact

class GreekCalculator:
    def __init__(self, S, K, T, r, sigma, put = False):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.coef = -1 if put else 1
        
    def get_black_scholes(self):
        '''European opton price calcuation'''
        self.d1 = (np.log(self.S / self.K) + (self.r + self.sigma**2 / 2)\
            *self.T) / (self.sigma * np.sqrt(self.T))
        self.d2  = self.d1 - self.sigma * np.sqrt(self.T)
        
        self.call_price = (self.S * si.norm.cdf(self.d1*self.coef)*self.coef) - self.K \
            * np.exp(-self.r * self.T) * si.norm.cdf(self.d2*self.coef)* self.coef
        
        return self.call_price, self.d1, self.d2

    def get_delta(self):
        '''Delta: Sensetivity to price changes'''
        self.delta = si.norm.cdf(self.d1)
        return self.delta + self.coef if self.coef < 0 else self.delta

    def get_gamma(self):
        '''Gamma: Sensetivity of Delta to price changes'''
        self.gamma = si.norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
        return self.gamma
        
    def get_theta(self):
        '''THeta: Time decay of option price'''
        term1 = -(self.S * si.norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        term2 = self.r * self.K * np.exp(-self.r*self.T) * si.norm.cdf(self.d2*self.coef)
        
        self.theta = term1 - term2*self.coef
        return self.theta
        
    def get_vega(self):
        '''Vega: Sensitivity to volatility'''
        self.vega = self.S * si.norm.pdf(self.d1) * np.sqrt(self.T)
        
        return self.vega
        
    def get_rho(self):
        '''Rho: Sensitivity to interest rate'''
        self.rho = self.K * self.T * si.norm.cdf(self.d2*self.coef) * np.exp(-self.r*self.T) * self.coef
        return self.rho
    
    def get_option_greeks(self):
        price, d1, d2 = self.get_black_scholes()

        print(f"Call option Price: ${price:.2f}")
        #print(f"d1; d2: {d1};{d2}")
        print(f"Sensetivity to price changes Delta: {self.get_delta():.3f}")
        print(f"Sensetivity of Delta to price changes Gamma: {self.get_gamma():.3f}")
        print(f"Price decay over time Theta: {self.get_theta():.3f}")
        print(f"Sensetivity to volatiliy Vega: {self.get_vega():.3f}")
        print(f"ensetivity to interest rate Rho: {self.get_rho():.3f}")
    
if __name__ == '__main__':
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    
    c = GreekCalculator(S=S, K=K, T=T, r=r, sigma=sigma, put=True)
    c.get_option_greeks()
    
    
    
    
    
        
