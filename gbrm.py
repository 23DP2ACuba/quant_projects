import matplotlib.pyplot as plt
import numpy as np

def simulate_grw(S0, T=2, N=1000, mu=0.1, sigma=0.05):
    dt = T/N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)

    W = np.cumsum(W) * np.sqrt(dt)

    X = (mu - 0.5 * (sigma**2)) * t + sigma * W
    S = S0 * np.exp(X)

    return t, S

def plot(t, S):
    plt.plot(t, S)
    plt.xlabel("t")
    plt.ylabel("S(t)")
    plt.title("Geometric brownian motion")
    plt.show()



if __name__ == "__main__":
    t, S = simulate_grw(55)
    plot(t, S)
