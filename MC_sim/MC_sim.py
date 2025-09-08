import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUM_SIMMULATIONS = 100

def stock_mc(S0, mu, sigma, N=252, t=1):
    results = []
    for _ in range(NUM_SIMMULATIONS):
        price = [S0]
        for _ in range(N):
            St = price[-1] * np.exp((mu- 0.5 * sigma ** 2) * t + sigma * np.random.normal())
            price.append(St)

        results.append(price)

    data = pd.DataFrame(results)
    data = data.T

    data["Mean"] = data.mean(axis=1)

    print(f"Price in {N} days: {data.Mean.tail()}")
    plt.plot(data)
    plt.plot(data["Mean"])
    plt.show()

if __name__ == "__main__":
    stock_mc(50, 0.0002, 0.01)