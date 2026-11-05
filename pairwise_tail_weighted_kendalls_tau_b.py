class TailWeightedKendall:
  _BLOCK_ELEMS = 2e7

  def __init__(self, params, **kwargs):
    self.p = params

  def empirical_uniforms(self, losses, T):
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 2:
      raise ValueError("losses must be two-dimensional.")

    k_ranks = rankdata(
        losses, method="average", axis=0
    )
    return k_ranks / (T + 1.0)

  def tail_weight(self, u):
    u = np.asarray(u, dtype=float)
    tail_depth = np.maximum(
        (u - self.p.tail_threshold) / (1 - self.p.tail_threshold),
        0.0
    ) ** self.p.tail_power

    return tail_depth
    

  def _get_tau(self, U, w):
    U = np.asarray(U, dtype=float)
    T_dim, N = U.shape
    z = (w <= 0).astype(float)

    G = np.zeros((N, N))
    D_and = np.zeros((N, N))
    D_or = np.zeros((N, N))
    Gw_and = np.zeros((N, N))
    Gw_or = np.zeros((N, N))
    zeros = np.zeros(N)

    block_size = max(1, int(self._BLOCK_ELEMS // (T_dim * N)))
    for a in range(0, T_dim, block_size):
      b = min(a + block_size, T_dim)
      S = np.sign(U[a:b, None, :] - U[None, :, :])
      S2 = S ** 2
      S_flat = S.reshape(-1, N)

      G += S_flat.T @ S_flat
      zeros += (S_flat == 0).sum(axis=0)

      W_pairwise = w[a:b, None, :] * w[None, :, :]
      A = S * W_pairwise

      A_flat = A.reshape(-1, N)
      Gw_and += A_flat.T @ A_flat
      S2W_flat = (S2 * W_pairwise).reshape(-1, N)
      D_and += 0.5 * (S2W_flat.T @ W_pairwise.reshape(-1, N))

      h_t = w[a:b, None, :]
      h_s = w[None, :, :]

      def wO(h): 
        return h[..., :, None] + h[..., None, :] - h[..., :, None] * h[..., None, :]

      W = wO(h_s) * wO(h_t)
      mask = np.arange(w.shape[0])[None, :] < np.arange(a, b)[:, None]

      Gw_or += np.einsum('ts,tsij,tsi,tsj->ij', mask, W, S, S)
      D_or += np.einsum(
          'ts,tsij,tsi->ij',
          mask, W, S2
      )

    return G, Gw_and, Gw_or, zeros, D_and, D_or

  def _adjust_tau_or(self, tau_or_raw):
    m = (1 - self.p.tail_threshold) / (1 + self.p.tail_power)
    r = (1 - self.p.tail_threshold) / (2 + self.p.tail_power)
    
    tau_or_0 = -2 * ((1-2*r)/(2-m))**2

    return np.where(
        tau_or_raw >= tau_or_0,
        (tau_or_raw - tau_or_0) / (1.0 - tau_or_0), 
        (tau_or_raw - tau_or_0) / (1.0 + tau_or_0)
    )


  def _tau_matrix(self, losses, U):
    T_dim, N = losses.shape
    h = self.tail_weight(U)
    n0 = T_dim * (T_dim - 1) / 2.0

    G, Gw_and, Gw_or, zeros, D_and, D_or_u = self._get_tau(losses, h)
    ties = (zeros - T_dim) / 2.0 

    den_g = np.sqrt(np.outer(n0 - ties, n0 - ties))
    tau_glob= np.divide(G / 2.0, den_g, out=np.zeros((N, N)), where=den_g > 0)

    w2 = h ** 2
    W_prod = h.T @ h
    W2_prod = w2.T @ w2
    T_and = 0.5 * (W_prod ** 2 - W2_prod)
    D_v = D_and.T
    den_and = np.sqrt(D_and * D_v)

    tau_and = np.divide(
        0.5*Gw_and, den_and, out=np.full((N, N), np.nan), where=den_and > 0
    )

    ess_and = np.divide(
        W_prod ** 2, 
        W2_prod, 
        out=np.zeros((N, N)), 
        where=W2_prod > 0
    )

    D_or_v = D_or_u.T

    den_or = np.sqrt(
        np.maximum(D_or_u, 0)
        * np.maximum(D_or_v, 0)
    )

    tau_or_raw = np.divide(
        Gw_or, den_or, out=np.full_like(Gw_or, np.nan), where=den_or > 0
    ) 
        
    s1 = h.sum(axis=0)
    h2 = h**2
    s2 = (h2).sum(axis=0)

    sum_w = s1[:, None] + s1[None, :] - h.T @ h
    sum_w2 = (
        s2[:, None] + s2[None, :] 
        + h2.T @ h2 + 2*h.T @ h
        - 2*h2.T @ h - 2*(h2.T @ h).T
    )

    ess_or = np.divide(
        sum_w**2,
        sum_w2,
        out=np.zeros_like(sum_w),
        where=sum_w2 > 0
    )
    tau_or = self._adjust_tau_or(tau_or_raw)

    return tau_glob, tau_and, tau_or, ess_and, ess_or

def _get_rel_score(self, ess):
  score = (
    (ess - self.p.min_effective_sample)
    / (ess + self.p.shrink_target_ess - self.p.min_effective_sample), 0.0, 1
  )

  score = np.clip(score, 0.0, 1.0)
  return score

def fit_kendall(self, returns, assets=None, tau_prev=None, return_tau=False):
  if assets is None:
    assets = list(returns.columns)

  losses = -returns
  n_steps, n_assets = losses.shape

  U = self.empirical_uniforms(losses, T=n_steps)
  tau_glob, tau_and, tau_or, ess_and, ess_or = self._tau_matrix(losses=losses, U=U)
  
  
  reliability_score_and = self._get_rel_score(ess_and)
  reliability_score_or = self._get_rel_score(ess_or)

  eff_comp_and = self.p.max_tail_inf * self.p.and_pref * reliability_score_and
  eff_comp_or = self.p.max_tail_inf * (1 - self.p.and_pref) * reliability_score_or
  eff_comp_glob = 1 - eff_comp_and - eff_comp_or

  tau_final = (
      eff_comp_and * tau_and 
      + eff_comp_or * tau_or 
      + eff_comp_glob * tau_glob
  )

  return tau_final
