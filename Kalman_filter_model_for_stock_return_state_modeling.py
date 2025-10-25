import yfinance as yf
import numpy as np
from numpy.linalg import inv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
np.random.seed(42)

# -------------------------------
# Parameters
# -------------------------------
ticker = "TSLA"
market_ticker = "^GSPC"
start = "2020-01-01"
end = "2025-01-01"
W = 30               
K = 3                 
state_dim = 1         
max_alloc = 1.0       
reestimation_period = 21
gamma = 5             

# -------------------------------
# Download data
# -------------------------------
def download(ticker, start, end):
    return yf.Ticker(ticker).history(start=start, end=end)[["Close"]]

data = download(ticker, start, end)
market = download(ticker=market_ticker, start=start, end=end)

r_st = np.log(data["Close"]/data["Close"].shift()).dropna().values
r_mt = np.log(market["Close"]/market["Close"].shift()).dropna().values
T = len(r_st)

# -------------------------------
# Weighted KMeans
# -------------------------------
def weighted_km(X, weights, n_clusters=3, n_init=10):
    weights = np.nan_to_num(weights, nan=1e-6, posinf=1e6, neginf=1e-6)
    weights = np.clip(weights, 1e-3, None)
    int_weights = np.round(weights / weights.min()).astype(int)
    X_rep = np.repeat(X, int_weights, axis=0)
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=42).fit(X_rep)
    return km

# -------------------------------
# Kalman filter
# -------------------------------
def kalman_filter(y, F, B, H, Q, R, u, x0=None, P0=None):
    y = np.asarray(y).flatten()
    T = len(y)
    n = F.shape[0]

    x_pred = np.zeros((T, n))
    P_pred = np.zeros((T, n, n))
    x_filt = np.zeros((T, n))
    P_filt = np.zeros((T, n, n))

    x_prev = np.zeros((n,1)) if x0 is None else np.asarray(x0).reshape(n,1)
    P_prev = np.eye(n) if P0 is None else np.asarray(P0).reshape(n,n)

    for t in range(T):
        u_t = np.asarray(u[t]).reshape(-1,1)
        x_t_pred = F @ x_prev + B @ u_t
        P_t_pred = F @ P_prev @ F.T + Q

        y_t = y[t] - (H @ x_t_pred).item()
        S_t = H @ P_t_pred @ H.T + R
        K_t = P_t_pred @ H.T @ inv(S_t)

        x_t_filt = x_t_pred + K_t * y_t
        P_t_filt = (np.eye(n) - K_t @ H) @ P_t_pred

        x_pred[t,:] = x_t_pred.flatten()
        P_pred[t,:,:] = P_t_pred.reshape(n,n)
        x_filt[t,:] = x_t_filt.flatten()
        P_filt[t,:,:] = P_t_filt.reshape(n,n)

        x_prev = x_t_filt
        P_prev = P_t_filt

    return x_pred, P_pred, x_filt, P_filt

# -------------------------------
# EM estimation
# -------------------------------
def EM_estimate(y, u, F_init, B_init, H, Q_init, R_init, n_iter=5):
    y = np.array(y).flatten()
    u = np.array(u)
    n = F_init.shape[0]
    F, B, Q, R = F_init.copy(), B_init.copy(), Q_init.copy(), R_init

    for _ in range(n_iter):
        x_pred, P_pred, x_filt, P_filt = kalman_filter(y, F, B, H, Q, R, [u_i for u_i in u])
        x_prev = x_filt[:-1,0].reshape(-1,1)
        x_next = x_filt[1:,0].reshape(-1,1)
        u_prev = u[1:]

        A = np.hstack([x_prev, u_prev])
        coef = np.linalg.lstsq(A, x_next, rcond=None)[0]
        F[0,0], B[0,0] = coef.flatten()

        res = x_next.flatten() - (F[0,0]*x_prev.flatten() + B[0,0]*u_prev.flatten())
        Q[0,0] = np.var(res)
        y_pred = H @ x_filt.T
        R = np.var(y - y_pred.flatten())

    return F, B, Q, R

# -------------------------------
# Backtest loop
# -------------------------------
x0 = np.zeros((state_dim,1))
P0 = np.eye(state_dim)

w_alloc = []
portfolio_ret = []

F = np.array([[1.0]])
B = np.array([[0.0]])
H = np.array([[1.0]])
Q = np.array([[1e-6]])
R = np.array([[np.var(r_st)]])

for t in range(W, T-1):
    x_window = np.column_stack([r_st[t-W:t], r_mt[t-W:t]])
    wts = np.abs(r_mt[t-W:t])
    km = weighted_km(x_window, wts, n_clusters=K)

    cluster_id = km.predict(np.array([[r_st[t], r_mt[t]]]))[0]
    u_t = np.array([[cluster_id / (K-1)]])

    if (t-W) % reestimation_period == 0:
        y_train = r_st[t-W:t]
        cluster_train = km.predict(x_window)
        u_train = cluster_train.reshape(-1,1) / (K-1)
        F, B, Q, R = EM_estimate(y_train, u_train, F, B, H, Q, R, n_iter=5)

    x_pred, P_pred, x_filt, P_filt = kalman_filter(
        np.array([r_st[t]]).reshape(-1,1), F, B, H, Q, R, [u_t], x0, P0
    )
    mu_pred = x_pred[-1,0]
    sigma_pred = np.sqrt(P_pred[-1,0,0] + R[0,0])

    w_t = mu_pred / (gamma * sigma_pred**2)
    w_t = np.clip(w_t, 0, max_alloc)
    w_alloc.append(w_t)

    ret = w_t * r_st[t+1]
    portfolio_ret.append(ret)

    x0 = x_filt[-1].reshape(-1,1)
    P0 = P_filt[-1]

# -------------------------------
# Results
# -------------------------------
w_alloc = np.array(w_alloc)
portfolio_ret = np.array(portfolio_ret)
cum_ret = np.cumprod(1 + portfolio_ret) - 1

plt.figure(figsize=(10,5))
plt.plot(cum_ret, label='Portfolio (Kalman + KMeans)')
plt.plot(np.cumprod(1 + r_st[W+1:]) - 1, label=ticker, alpha=0.6)
plt.legend()
plt.title('Cumulative Returns')
plt.show()
