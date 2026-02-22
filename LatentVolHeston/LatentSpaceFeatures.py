@dataclass
class Params:
  ticker: str
  start: str
  end: str
  vol_window: int = 20
  T: float = 1/252
  garch_window: int = 60
  tcm_window: int = 150
  q: float = 0.90
  pct: float = 0.95
  bounds = [(-1, 2), (1e-6, None)]

class LatentSpaceVol:
  def __init__(self, feature_params, **kwargs):
    self.params = feature_params
    
  def get_price_data(self):
    self.data = yf.Ticker(self.params.ticker).history(start=self.params.start, end=self.params.end, auto_adjust=False)[["Close"]]

  def get_tail_data(self, returns):
    L = np.abs(returns[returns < 0])

    if len(L) < 10:
      return np.nan, np.nan, np.nan, np.nan, np.nan

    u = np.percentile(L, 100 * self.params.pct)
    P_u = np.mean(L > u)
    Y = L[L > u] - u

    xi, loc, beta = genpareto.fit(Y, floc=0)

    VaR_q = u + beta / xi * (((1 - self.params.q) / P_u) ** (-xi) - 1)
    CTE_q = (VaR_q + (beta - xi * u)) / (1 - xi)

    moments = self.get_tail_moments_gpd((xi, beta))
    tv, skewness, kurtosis = moments[1:]

    return xi, CTE_q, tv, skewness, kurtosis

  def get_tail_moments_gpd(self, params):
    xi, beta = params
    moments = []
    order = 4
    eps = 1e-6

    for k in range(1, order+1):
      if xi >= 1 / k:
        moments.append(eps)
        continue

      moment = np.exp(
        math.lgamma(k + 1) + k * np.log(beta) - sum(np.log(1 - j * xi) for j in range(1, k + 1))
      )
      moments.append(round(moment, 8))

    return moments

  def gpd_neg_log_likelihood(self, params: tuple, Y: np.ndarray):
    xi, beta = params
    eps = 1e-6

    if beta <= 0:
      return eps
    if abs(xi) < eps:
      nll = np.sum(np.log(beta) + Y/beta)

    else:
      t = 1 + xi * Y / beta
      if np.any(t <= 0):
        return eps
      nll = np.sum(np.log(beta) + (1/xi+ 1) * np.log(t))

    return nll

  def get_garch_params(self, r):
    alphas, betas = [], []
    steps = r.shape[0]
    for i in range(self.params.garch_window, r.shape[0]):
      train_data = r[i - self.params.garch_window:i]
      model = arch_model(train_data, rescale=False, p=1, q=1)
      model_fit= model.fit(disp="off")

      params = model_fit.params
      alphas.append(params["alpha[1]"])
      betas.append(params["beta[1]"])

    filler = [np.nan] * (self.params.garch_window+1)

    return  np.array(filler+alphas), np.array(filler+betas)

  def normalize(self, df):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_scaled

  def generate_features(self):
    eps = 1e-5
    r = self.data["Close"].pct_change()
    r2 = r**2
    r.dropna(inplace=True)

    self.data["RealizedVOl"] = np.sqrt(r2.rolling(window=self.params.vol_window).mean())

    self.data["StdDev"] = r.rolling(window=self.params.vol_window).std()

    self.data["MAD"] = r.rolling(window=self.params.vol_window).apply(lambda x: np.mean(np.abs(x-np.mean(x))))
    print("caculating tcm:")
    step = np.floor(len(r)/15)
    tcm_cols = ["Xi", "CTE", "TV", "Skew", "Kurt"]
    tcm = pd.DataFrame(index=self.data.index, columns=tcm_cols)

    for i in range(self.params.tcm_window, len(r)):
      window = r.iloc[i - self.params.tcm_window:i].dropna()
      tcm.iloc[i] = self.get_tail_data(window)
      if i % step == 0:
        print("|", end="")


    self.data = pd.concat([self.data, tcm], axis=1)
    self.data["Xi"] = pd.to_numeric(self.data["Xi"], errors="coerce")
    self.data["CTE"] = pd.to_numeric(self.data["CTE"], errors="coerce")
    self.data["Xi"] = np.tanh(self.data["Xi"].values)
    self.data["CTE"] = np.log(self.data["CTE"])
    print(f"\ncaculating garch parameters:")

    alphas, betas = self.get_garch_params(r)

    self.data["Presistance"], self.data["Alpha"] = (alphas+betas), alphas
    self.data.drop(["Close"], axis=1, inplace=True)
    self.data.dropna(inplace=True)


    self.df_normalized = self.normalize(self.data)
