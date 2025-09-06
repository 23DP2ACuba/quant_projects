from scipy import stats
import numpy as np

S = 100
E = 100
T = 1
rf = 0.05
sigma = 0.2

def call_option_price(S, E, T, rf, sigma):
    d1 = (np.log(S/E)+(rf+sigma**2/2.0)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    print(f"d1: {d1}\nd2: {d2}")

    return S*stats.norm.cdf(d1)-E*np.exp(-rf*T)*stats.norm.cdf(d2)


def put_option_price(S, E, T, rf, sigma):
    d1 = (np.log(S/E)+(rf+sigma**2/2.0)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    print(f"d1: {d1}\nd2: {d2}")

    return -S*stats.norm.cdf(-d1)+E*np.exp(-rf*T)*stats.norm.cdf(-d2)



call_opt_price = call_option_price(S=S, E=E, T=T, rf=rf, sigma=sigma)
put_opt_price = put_option_price(S=S, E=E, T=T, rf=rf, sigma=sigma)

print(f"call opt price: {call_opt_price}, \
      \n put opt price: {put_opt_price}")
