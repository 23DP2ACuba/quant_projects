from numpy import random as npr
import matplotlib.pyplot as plt
import numpy as np

def wiener_process(dt=0.1, x0=0, n=1000):
    W = np.zeros(n+1)
    t = np.linspace(x0, n, n+1)
    W[1:n+1] = np.cumsum(npr.normal(0, np.sqrt(dt), n))

    return t, W

def plt_w(t, W):
    plt.plot(t, W)
    plt.title("Wiener Process")
    plt.xlabel("t")
    plt.xlabel("S(t)")
    plt.show()

if __name__ == "__main__":
    t, w = wiener_process()
    plt_w(t, w)
