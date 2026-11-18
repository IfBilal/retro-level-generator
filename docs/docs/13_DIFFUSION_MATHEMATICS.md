# 13 — Diffusion Mathematics (DDPM · DDIM · CFG · RePaint · Losses · Optimisation)

> **Purpose.** Every equation the code implements, derived once, with the numeric values verified for *this* project (T = 1000, 13-channel ±1 one-hot tiles). `src/text2mario/diffusion/scheduler.py`, `src/text2mario/train/train.py` and `src/text2mario/generate/*` must implement exactly these formulas. Function names are given in **bold**; each one has a test in `tests/test_scheduler.py` (doc 24).
> **Ledger rows used:** D7 (representation), M1 (ε-prediction, T = 1000), M2 (schedules), M4 (time embedding), M5–M7 (attention, text), M9 (losses), M10 (AdamW + LR), M11 (EMA), M13 (DDIM/CFG/clipping), G1 (RePaint), H5 (bf16).
> **Verification.** All numbers in the tables below were computed on 2026-11-17 with float64 PyTorch (scripts in `reference/`; re-run them in `tests/test_scheduler.py::test_reference_values`).

---

## 0. Notation and conventions (read this first — off-by-one bugs live here)

| Symbol | Meaning | Code |
|---|---|---|
| $T = 1000$ | number of diffusion steps | `cfg.diffusion.T` |
| $t \in \{0,\dots,T-1\}$ | **code index** of a step (0-based). The DDPM paper's $t' = t+1$. | `t` (LongTensor) |
| $\beta_t$ | variance added at step $t$ | `sched.betas[t]` |
| $\alpha_t = 1-\beta_t$ | | `sched.alphas[t]` |
| $\bar\alpha_t = \prod_{i=0}^{t}\alpha_i$ | cumulative signal retention; **convention $\bar\alpha_{-1} := 1$** | `sched.alphas_cumprod[t]`, `sched.alphas_cumprod_prev[t]` |
| $\mathrm{SNR}_t = \bar\alpha_t/(1-\bar\alpha_t)$ | signal-to-noise ratio | `sched.snr[t]` |
| $x_0 \in \{-1,+1\}^{13\times H\times W}$ | clean level, one-hot scaled: $x_0 = 2\,\mathrm{onehot}(y)-1$ (D7) | `x0` |
| $y \in \{0..12\}^{H\times W}$ | tile IDs (ledger F6) | `y` |
| $c$ | caption (text condition); $\varnothing$ = null caption `""` | |
| $\epsilon \sim \mathcal N(0,I)$ | standard Gaussian noise, same shape as $x_0$ | `eps` |
| $\epsilon_\theta(x_t,t,c)$ | network output (ε-prediction, M1) | `model(x_t, t, ctx, mask)` |
| $D = 13\cdot H\cdot W$ | number of scalar elements per sample ($D = 3328$ for 16×16) | |

**Every element of $x_0$ is $\pm 1$, so $\lVert x_0\rVert^2 = D$ exactly and $\mathbb E[x_{0,i}^2] = 1$.** Consequence: the *nominal* $\mathrm{SNR}_t$ is also the *true* per-element signal-to-noise ratio. (MarioDiffusion used {0, 1} one-hot, where $\mathbb E[x_{0,i}^2] = 1/13$, which silently shifts every log-SNR by $-\ln 13 \approx -2.56$; our ±1 choice from the PDF avoids that.)

**Every pixel's 13-vector has exactly one +1 and twelve −1**, so $\sum_c x_{0,c,i,j} = -11$ at every pixel. This linear constraint is used in §1.

All schedule buffers are built in **float64** at start-up and cast to float32 at the point of use (§14).

---

## 1. The data geometry: tiles are vertices, the ideal denoiser outputs probabilities

For one pixel, the 13 possible values of $x_0$ are the vectors $v_k = 2e_k - \mathbf 1 \in \mathbb R^{13}$, $k = 0..12$. All pairs are equidistant: $\lVert v_k - v_l\rVert^2 = 8$ for $k \ne l$.

**Posterior-mean identity.** For any noisy observation $x_t$ and any pixel,
$$
\mathbb E[x_{0}\mid x_t] = \sum_k p(y{=}k\mid x_t)\, v_k \quad\Longrightarrow\quad \mathbb E[x_{0,c}\mid x_t] = 2\,p(y{=}c\mid x_t) - 1 .
$$
So the MMSE estimate $\hat x_0$ at each pixel is an **affine image of the class posterior**:

1. $\hat x_{0,c}\in[-1,1]$ for the ideal estimator → **clipping $\hat x_0$ to $[-1,1]$ (M13) is principled, not a hack.**
2. $\sum_c \hat x_{0,c} = 2\cdot 1 - 13 = -11$ for the ideal estimator → a **free diagnostic**: the learned model's implied $\hat x_0$ should satisfy $\lvert\sum_c \hat x_{0,c} + 11\rvert \approx 0$. (Verified on the Bayes-optimal per-pixel estimator: deviation < 1e-13.) Log the mean absolute deviation per t-bucket in validation (doc 16 §10, `simplex_dev`).
3. $\arg\max_c \hat x_{0,c} = \arg\max_c p(y{=}c\mid x_t)$ → **argmax decoding of $\hat x_0$ is the MAP tile** (D7).

**When is a pixel's class decided?** Two vertices are $\sqrt 8 = 2.83$ apart. After scaling by $\sqrt{\bar\alpha_t}$ and adding noise of std $\sqrt{1-\bar\alpha_t}$, the separation-to-noise ratio is $2\sqrt2\,\sqrt{\mathrm{SNR}_t}$, which falls to 1 at $\mathrm{SNR}_t = 1/8$. Below that, a pixel's identity can only be inferred from **context** (neighbouring tiles, the caption). Verified crossing points: linear reaches SNR < 1/8 at **t = 464** (53.6 % of steps lie beyond it); cosine at **t = 781** (21.9 % beyond). See §3 for what this means.

---

## 2. Forward (noising) process

Markov chain:
$$
q(x_t\mid x_{t-1}) = \mathcal N\!\big(\sqrt{\alpha_t}\,x_{t-1},\ \beta_t I\big),\qquad t = 0..T-1,\ \ x_{-1}\equiv x_0 .
$$

**Closed form (derivation by induction).** Write $x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t$. Substituting $x_{t-1} = \sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-1}$:
$$
x_t = \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \underbrace{\sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t}_{\text{sum of independent Gaussians, variance } \alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t \,=\, 1-\alpha_t\alpha_{t-1}} .
$$
Repeating down to $x_0$:
$$
\boxed{\,q(x_t\mid x_0) = \mathcal N\!\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big)\quad\Leftrightarrow\quad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon\,}
$$
→ **`q_sample(x0, t, eps)`**. Shapes: `x0, eps: (B,13,H,W)`; `t: (B,)`; coefficients gathered and reshaped to `(B,1,1,1)`.

**Moments used by Gate G-P2.** Element-wise $\mathbb E[x_t] = \sqrt{\bar\alpha_t}\,\mathbb E[x_0]$ and $\mathrm{Var}[x_t] = \bar\alpha_t\mathrm{Var}[x_0] + (1-\bar\alpha_t)$. At $t = 999$: linear $\sqrt{\bar\alpha} = 6.35\times10^{-3}$, cosine $4.93\times10^{-5}$, so $x_{999}$ has mean ≈ 0 (|mean| < 0.02 in the gate) and std ≈ 1 (1 ± 0.02). Verified with random one-hot batches: mean −0.0046 / −0.0021, std 1.0013 / 0.9988.

---

## 3. Noise schedules (M2)

### 3.1 Linear (Ho et al. 2020)
$$
\beta_t = \beta_{\min} + \frac{t}{T-1}(\beta_{\max}-\beta_{\min}),\qquad \beta_{\min} = 10^{-4},\ \beta_{\max} = 0.02 .
$$
Code: `torch.linspace(1e-4, 0.02, T, dtype=torch.float64)`.

### 3.2 Cosine (Nichol & Dhariwal 2021)
$$
f(u) = \cos^2\!\Big(\frac{u/T + s}{1+s}\cdot\frac{\pi}{2}\Big),\qquad \bar\alpha^{\text{raw}}(u) = \frac{f(u)}{f(0)},\qquad \beta_t = \min\!\Big(1 - \frac{\bar\alpha^{\text{raw}}(t+1)}{\bar\alpha^{\text{raw}}(t)},\ 0.999\Big),\quad s = 0.008 .
$$
Then **recompute** $\bar\alpha_t = \prod_{i\le t}(1-\beta_i)$ from the clipped betas (this is what the code must store; it differs from $\bar\alpha^{\text{raw}}$ only at the last step). Exactly one beta is clipped (the last: $\beta_{999} = 0.999$). Unclipped $\bar\alpha^{\text{raw}}(T) = 3.7\times10^{-33}$, so the clip prevents $\beta = 1$ and the division by zero it would cause in the posterior coefficients.

### 3.3 Verified values

| t | linear β | linear ᾱ | linear SNR | linear log-SNR | cosine β | cosine ᾱ | cosine SNR | cosine log-SNR |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.000e-4 | 0.99990 | 9.999e3 | +9.21 | 4.128e-5 | 0.999959 | 2.422e4 | +10.10 |
| 100 | 2.092e-3 | 0.89514 | 8.537 | +2.14 | 5.319e-4 | 0.97158 | 34.18 | +3.53 |
| 250 | 5.080e-3 | 0.52142 | 1.090 | +0.09 | 1.327e-3 | 0.84589 | 5.489 | +1.70 |
| 500 | 1.006e-2 | 0.077797 | 8.436e-2 | −2.47 | 3.156e-3 | 0.49229 | 0.9696 | −0.03 |
| 750 | 1.504e-2 | 3.300e-3 | 3.311e-3 | −5.71 | 7.579e-3 | 0.14318 | 0.1671 | −1.79 |
| 900 | 1.803e-2 | 2.702e-4 | 2.703e-4 | −8.22 | 1.974e-2 | 2.362e-2 | 2.419e-2 | −3.72 |
| 990 | 1.982e-2 | 4.837e-5 | 4.837e-5 | −9.94 | 0.1900 | 1.967e-4 | 1.968e-4 | −8.53 |
| 999 | 2.000e-2 | **4.036e-5** | 4.036e-5 | −10.12 | 0.9990 | **2.429e-9** | 2.429e-9 | −19.84 |

Derived quantities (verified):

| Quantity | Linear | Cosine |
|---|---|---|
| First t with SNR < 1 | 259 | 496 |
| Fraction of steps with SNR > 1 | 25.9 % | 49.6 % |
| Fraction of steps with SNR > 10 | 9.3 % | 18.8 % |
| Fraction of steps with SNR < 0.01 | 32.6 % | 6.4 % |
| First t with SNR < 1/8 (§1 class-decision threshold) | 464 | 781 |
| $\bar\alpha_{T-1}$ (gate G-P2 bound) | 4.04e-5 (< 1e-4 ✓) | 2.43e-9 (< 1e-8 ✓) |

**Interpretation for a 16×16, 13-channel grid.** With uniform t-sampling, the linear schedule spends about a third of all training steps at SNR < 0.01, where $x_t$ is essentially pure noise and the network can only learn the marginal. The cosine schedule spreads steps much more evenly across the informative range of log-SNR. This is the quantitative basis for M2's "tie → cosine". Both are still trained in Phase 3, because the decision is made on measured dev metrics, not theory.

**Terminal SNR.** Sampling starts from $x_{T-1}\sim\mathcal N(0,I)$, which ignores the residual signal $\sqrt{\bar\alpha_{T-1}}x_0$. That residual is 0.0064 per element for linear (negligible against unit noise) and 5e-5 for cosine. No zero-terminal-SNR rescaling is needed.

---

## 4. The forward posterior $q(x_{t-1}\mid x_t, x_0)$

By Bayes, using the Markov property, $q(x_{t-1}\mid x_t,x_0) \propto q(x_t\mid x_{t-1})\,q(x_{t-1}\mid x_0)$. The exponent is
$$
-\tfrac12\Big[\tfrac{(x_t-\sqrt{\alpha_t}x_{t-1})^2}{\beta_t} + \tfrac{(x_{t-1}-\sqrt{\bar\alpha_{t-1}}x_0)^2}{1-\bar\alpha_{t-1}}\Big].
$$
**Completing the square in $x_{t-1}$.** The quadratic coefficient is $\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar\alpha_{t-1}} = \frac{\alpha_t(1-\bar\alpha_{t-1}) + \beta_t}{\beta_t(1-\bar\alpha_{t-1})} = \frac{1-\bar\alpha_t}{\beta_t(1-\bar\alpha_{t-1})}$ (using $\alpha_t\bar\alpha_{t-1} = \bar\alpha_t$ and $\alpha_t+\beta_t = 1$). Hence
$$
\boxed{\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t},\qquad
\boxed{\tilde\mu_t(x_t,x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t}.
$$
Substituting $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$:
$$
\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\Big).
$$
Buffers: `posterior_mean_coef1 = sqrt(ab_prev)*beta/(1-ab)`, `posterior_mean_coef2 = sqrt(alpha)*(1-ab_prev)/(1-ab)`, `posterior_variance = beta_tilde`, `posterior_log_variance_clipped = log(max(beta_tilde, beta_tilde[1]))`. The clipping matters because $\tilde\beta_0 = 0$ (as $\bar\alpha_{-1} = 1$), so log(0) is avoided. Verified: linear $\tilde\beta_1 = 5.45\times10^{-5}$, cosine $\tilde\beta_1 = 2.18\times10^{-5}$; at t = 999, $\tilde\beta \approx \beta$ (0.0200 / 0.99900).

→ **`q_posterior_mean_variance(x0, x_t, t)`**.

---

## 5. Reverse model, the ELBO, and the simplified ε-loss

Model: $p_\theta(x_{t-1}\mid x_t) = \mathcal N(\mu_\theta(x_t,t,c),\ \sigma_t^2 I)$ with fixed $\sigma_t^2\in\{\tilde\beta_t, \beta_t\}$.

**Variational bound.**
$$
-\log p_\theta(x_0) \le \underbrace{D_{KL}\big(q(x_{T-1}\mid x_0)\,\|\,p(x_{T-1})\big)}_{L_T\ \text{(no params)}} + \sum_{t\ge1}\underbrace{\mathbb E_q\,D_{KL}\big(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)}_{L_{t-1}} + \underbrace{\mathbb E_q[-\log p_\theta(x_0\mid x_1)]}_{L_0}.
$$
For two Gaussians with equal isotropic variance, $L_{t-1} = \frac{1}{2\sigma_t^2}\lVert\tilde\mu_t - \mu_\theta\rVert^2 + C$. Parameterising
$$
\mu_\theta(x_t,t,c) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t,c)\Big)
$$
gives
$$
L_{t-1} = \frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}\;\mathbb E\,\lVert\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t,\ c)\rVert^2 + C .
$$
Dropping the t-dependent weight (Ho et al. 2020) gives the **simplified loss**, our baseline (M9):
$$
\boxed{\mathcal L_{\text{simple}} = \mathbb E_{t\sim\mathcal U\{0..T-1\},\ x_0,\ c,\ \epsilon}\Big[\tfrac1D\big\lVert \epsilon - \epsilon_\theta(x_t,t,c)\big\rVert^2\Big]}
$$
(`F.mse_loss(eps_hat.float(), eps)` with mean reduction over all $B\cdot D$ elements).

**What ε-MSE weights in x₀-space.** From $x_0 = (x_t-\sqrt{1-\bar\alpha_t}\epsilon)/\sqrt{\bar\alpha_t}$:
$$
\hat x_0 - x_0 = -\sqrt{\tfrac{1-\bar\alpha_t}{\bar\alpha_t}}\,(\hat\epsilon - \epsilon)\quad\Rightarrow\quad \lVert\epsilon-\hat\epsilon\rVert^2 = \mathrm{SNR}_t\,\lVert x_0-\hat x_0\rVert^2 .
$$
Uniform-t ε-MSE is therefore x₀-MSE weighted by SNR. Errors at high noise count almost nothing, which is why per-t-bucket monitoring is mandatory (doc 16 §10).

**Reference curve: the context-free floor (verified, Monte Carlo).** The best ε-MSE achievable by a denoiser that sees each pixel *alone* (Bayes-optimal under the measured tile marginal F13, no spatial context) is $\mathrm{SNR}_t\cdot\mathrm{MMSE}_{x_0}(t)/13$:

| t-bucket | 0–99 | 100–199 | 200–299 | 300–399 | 400–499 | 500–599 | 600–699 | 700–799 | 800–899 | 900–999 | **mean** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear | 0.0000 | 0.0091 | 0.0299 | 0.0218 | 0.0101 | 0.0037 | 0.0011 | 0.0003 | 0.0001 | 0.0000 | **0.0076** |
| cosine | 0.0000 | 0.0000 | 0.0026 | 0.0176 | 0.0303 | 0.0290 | 0.0205 | 0.0112 | 0.0043 | 0.0007 | **0.0116** |

Consequences, used in docs 16, 19 and 25:
- The ε-MSE scale for this data is **small**: 12 of 13 channels are almost always −1, so most elements are easy. A healthy model's dev loss should end **clearly below** these floors (it can use spatial context). Zero-init gives an initial loss of ≈ 1.0 (§12.6).
- The PDF's "stop if MSE plateaus above 0.15" rule (audit #26) is mis-calibrated by more than an order of magnitude. **Replacement rule:** if, after 5k steps, the dev uniform-t mean ε-MSE is not below the context-free floor of its schedule, or any mid-t bucket stays above its floor value, the model has not learned spatial structure. Stop and debug (data scaling, tile order, conditioning wiring); do not train longer.
- The loss is near 0 at low t (the class is recoverable by rounding) and near 0 at high t (SNR → 0). It **peaks in the middle**, which is where the learning signal lives.

---

## 6. Parameterisations and conversions

With $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$:

| Want | From ε̂ | From x̂₀ | From v̂ |
|---|---|---|---|
| $\hat x_0$ | $(x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$ | — | $\sqrt{\bar\alpha_t}x_t - \sqrt{1-\bar\alpha_t}\,\hat v$ |
| $\hat\epsilon$ | — | $(x_t - \sqrt{\bar\alpha_t}\hat x_0)/\sqrt{1-\bar\alpha_t}$ | $\sqrt{1-\bar\alpha_t}\,x_t + \sqrt{\bar\alpha_t}\,\hat v$ |
| $v$ (definition) | $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$ | | |

→ **`predict_x0_from_eps(x_t, t, eps)`**, **`predict_eps_from_x0(x_t, t, x0)`**.

**Why ε (M1).** It matches the PDF and the paper, and it is the best-studied parameterisation for DDIM and CFG. Its weakness, the $1/\sqrt{\bar\alpha_t}$ blow-up of $\hat x_0$ at high t (157× for linear, about 20,000× for cosine at t = 999), is handled by clipping and fp32 arithmetic (§14). v-prediction would remove the blow-up. It is documented here as a fallback only (doc 25), not an experiment, to keep the ablation budget focused.

---

## 7. Sampling

### 7.1 DDPM ancestral step (reference sampler; Phase 2 test target)
$$
x_{t-1} = \mu_\theta(x_t,t,c) + \sigma_t z,\quad z\sim\mathcal N(0,I),\quad \sigma_t^2 = \tilde\beta_t\ (\text{default}),\qquad \text{and at } t = 0:\ \text{output } \hat x_0 \ (\text{no noise}).
$$
→ **`ddpm_step(eps_hat, t, x_t, generator)`**. The full 1000-step DDPM sampler is used only in tests and as a quality reference (too slow for the app).

### 7.2 DDIM (Song et al. 2021) — the production sampler (M13)
DDIM defines a family of **non-Markovian** forward processes with the same marginals $q(x_t\mid x_0)$, so the same trained ε-network applies. For any pair of timesteps $t > s$ (with $\bar\alpha_s > \bar\alpha_t$):
$$
x_s = \sqrt{\bar\alpha_s}\,\hat x_0 + \sqrt{1-\bar\alpha_s-\sigma^2}\;\hat\epsilon + \sigma z,\qquad
\sigma = \eta\,\sqrt{\frac{1-\bar\alpha_s}{1-\bar\alpha_t}}\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_s}},
$$
where $\hat x_0 = \mathrm{clip}\big(\text{predict\_x0\_from\_eps}(x_t,t,\hat\epsilon), -1, 1\big)$ and, **when clipping changed $\hat x_0$, $\hat\epsilon$ is recomputed from the clipped $\hat x_0$** (`predict_eps_from_x0`), so that the update stays self-consistent.

- $\eta = 0$ (our default): deterministic. The map $x_{T-1}\mapsto x_0$ is a discretised probability-flow ODE, which makes seeds reproducible and supports the "same seed, different w" comparisons.
- $\eta = 1$ and $s = t-1$: recovers DDPM with $\sigma_t^2 = \tilde\beta_t$.
- Last step: $s = -1$ with $\bar\alpha_{-1} = 1$ returns $x_{-1} = \hat x_0$, which is then decoded by argmax (§1).

**Why this works for marginals (one-line proof).** If $x_t\sim q(x_t\mid x_0)$ and $\hat x_0 = x_0$, $\hat\epsilon = \epsilon$, then $x_s = \sqrt{\bar\alpha_s}x_0 + \sqrt{1-\bar\alpha_s-\sigma^2}\,\epsilon + \sigma z$ has mean $\sqrt{\bar\alpha_s}x_0$ and variance $1-\bar\alpha_s-\sigma^2+\sigma^2 = 1-\bar\alpha_s$, i.e. $x_s\sim q(x_s\mid x_0)$. ∎

**Oracle test (Gate G-P2).** Feed the *true* noise $\epsilon^\star(x_t) = (x_t - \sqrt{\bar\alpha_t}x_0)/\sqrt{1-\bar\alpha_t}$ at every step. η = 0 DDIM must return $x_0$ exactly. Verified: float64 MSE = 0.0 and argmax agreement 100 % for both schedules (threshold MSE < 1e-6).

### 7.3 Strided timesteps
$$
\tau_i = \mathrm{round}\Big((T-1)\,\big(1 - \tfrac{i}{S-1}\big)\Big),\quad i = 0..S-1 \qquad(\texttt{torch.linspace(T-1, 0, S).round().long()}).
$$
The sequence starts at 999 (pure noise, matching training) and ends at 0. Each step maps $\tau_i\to\tau_{i+1}$; after $\tau_{S-1} = 0$ a final step to $\bar\alpha_{-1} = 1$ returns $\hat x_0$. Verified lists:

- **S = 50 (default):** 999, 979, 958, 938, 917, 897, 877, 856, 836, 816, 795, 775, 754, 734, 714, 693, 673, 652, 632, 612, 591, 571, 550, 530, 510, 489, 469, 449, 428, 408, 387, 367, 347, 326, 306, 285, 265, 245, 224, 204, 183, 163, 143, 122, 102, 82, 61, 41, 20, 0
- **S = 25 (CPU fast mode, H10):** 999, 957, 916, 874, 832, 791, 749, 708, 666, 624, 583, 541, 500, 458, 416, 375, 333, 291, 250, 208, 166, 125, 83, 42, 0

→ **`ddim_timesteps(S)`**, **`ddim_step(eps_hat, t, s, x_t, eta, generator, clip=True)`**.

### 7.4 Optional: projection instead of clipping
Clipping enforces only the box constraint. The ideal $\hat x_0$ also satisfies $\sum_c\hat x_{0,c} = -11$ (§1). A Euclidean projection onto $\{x: x_c\in[-1,1],\ \sum_c x_c = -11\}$ (equivalently, projecting $(\hat x_0+1)/2$ onto the probability simplex) is a stricter, still principled alternative. It is **not** the default, and is listed as a CFG-instability remedy in doc 25.

---

## 8. Classifier-free guidance (M13)

**Derivation.** We want samples from the sharpened conditional $\tilde p_w(x\mid c)\propto p(x)\,p(c\mid x)^w$. Its score is
$$
\nabla_x\log\tilde p_w(x\mid c) = \nabla_x\log p(x) + w\,\nabla_x\log p(c\mid x) = \nabla_x\log p(x) + w\big(\nabla_x\log p(x\mid c) - \nabla_x\log p(x)\big),
$$
using Bayes: $\nabla_x\log p(c\mid x) = \nabla_x\log p(x\mid c) - \nabla_x\log p(x)$. For the noised marginals, $\epsilon_\theta(x_t,t,\cdot)\approx -\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_t(x_t\mid\cdot)$, so
$$
\boxed{\hat\epsilon = \epsilon_\theta(x_t,t,\varnothing) + w\,\big(\epsilon_\theta(x_t,t,c) - \epsilon_\theta(x_t,t,\varnothing)\big)}
$$
**Convention (fixed project-wide): $w = 1$ ⇒ plain conditional, $w = 0$ ⇒ unconditional, $w > 1$ ⇒ guidance.** (Some papers write $1+w$; never mix the two.) One network provides both terms, because training replaces $c$ by $\varnothing$ = `""` with probability 0.10 (D8d).

**Implementation.** One forward pass on a doubled batch: inputs `cat([x_t, x_t])`, `cat([t, t])`, contexts `[ctx_c ; ctx_null]` padded to a common length, with masks. Then `eps_c, eps_u = out.chunk(2)`. → **`guided_eps(model, x_t, t, ctx_c, mask_c, ctx_u, mask_u, w)`**. Tests: w = 1 equals the conditional forward, w = 0 equals the unconditional forward, and the batched result equals two separate forwards (doc 15 §9).

**Interplay with clipping.** For large w, $\hat\epsilon$ extrapolates beyond the data manifold, so $\hat x_0$ leaves $[-1,1]$. Clipping (§7.2) contains this. The **clip rate** (fraction of $\hat x_0$ elements clipped, averaged over steps) is logged as a w-diagnostic (doc 15 §7).

---

## 9. Inpainting / outpainting: RePaint known-region replacement (G1)

Let $m\in\{0,1\}^{H\times W}$ mark known columns (1 = known, broadcast over channels), with known clean content $x_0^{\text{kn}}$. At each sampling step $t\to s$:

1. **Known part, freshly noised to the current level:** $x_t^{\text{kn}} = \sqrt{\bar\alpha_t}\,x_0^{\text{kn}} + \sqrt{1-\bar\alpha_t}\,z$, with $z$ **fresh** each step. (This resolves the PDF's ambiguous "noised to timestep t".)
2. **Merge:** $x_t \leftarrow m\odot x_t^{\text{kn}} + (1-m)\odot x_t$.
3. **Denoise:** $x_s = \text{ddim\_step}(\hat\epsilon(x_t), t, s, x_t)$ with CFG.
4. **Resampling (harmonisation), r times** for steps in the first 60 % of the schedule (G1: r = 2). Jump back from $s$ to $t$ with the forward kernel between the two strided levels:
$$
\boxed{x_t = \sqrt{\tfrac{\bar\alpha_t}{\bar\alpha_s}}\;x_s + \sqrt{1-\tfrac{\bar\alpha_t}{\bar\alpha_s}}\;z}
$$
then repeat steps 1–3. **Proof that this is the right kernel:** if $x_s\sim\mathcal N(\sqrt{\bar\alpha_s}x_0,(1-\bar\alpha_s)I)$, the result has mean $\sqrt{\bar\alpha_t}x_0$ and variance $\frac{\bar\alpha_t}{\bar\alpha_s}(1-\bar\alpha_s) + 1 - \frac{\bar\alpha_t}{\bar\alpha_s} = 1-\bar\alpha_t$, which is exactly $q(x_t\mid x_0)$. Verified by Monte Carlo (200k samples; for jumps 0→20, 500→520, 959→979 the empirical mean and variance match theory within sampling error, e.g. var 0.5395 vs 0.5388).

The jump re-injects noise, so the unknown region "forgets" a little and is re-denoised while conditioned on a fresh, consistent version of the known region. That is what removes seams. Cost per window: $S + r\cdot\lceil0.6S\rceil$ guided passes (50 + 2·30 = 110 = 2.2 × 50, consistent with H10). → **`renoise(x_s, s, t, generator)`**. The full procedure (window schedule, scene prompts, framing) is in doc 20.

---

## 10. Losses (M9)

### 10.1 Baseline
$\mathcal L = \mathcal L_{\text{simple}}$ (§5).

### 10.2 Channel-weighted MSE (rollback, also experiment X9)
$$
\mathcal L_{\text{cw}} = \frac{1}{B\,D}\sum_{b,c,i,j} w_c\,(\epsilon - \hat\epsilon)^2_{b,c,i,j},\qquad w_c = \frac{\tilde w_c}{\frac{1}{13}\sum_{c'}\tilde w_{c'}},
$$
with raw weights $\tilde w$: sky (ID 0) = 0.5; ground (8) and breakable (7) = 1.0; all ten rarer tiles (IDs 1–6, 9–12) = 2.0 (ledger M9; IDs per F6). Normalising to mean 1 keeps the loss scale, and so the effective learning rate, comparable to the baseline. Without it the comparison would be confounded with an LR change (raw mean = 22.5/13 = 1.731). Normalised: sky 0.289, ground/breakable 0.578, rare 1.156.
*Alternative (documented, not implemented by default):* per-pixel weights by the **true class** $y_{ij}$, which target rare-tile *locations* rather than channels.

### 10.3 Auxiliary cross-entropy on $\hat x_0$ (experiment X1)
$$
\mathcal L = \mathcal L_{\text{simple}} + \lambda\;\frac1B\sum_b \bar\alpha_{t_b}\sum_{i,j}\mathrm{CE}\Big(\mathrm{softmax}_c\big(\tau\,\hat x_{0,b,c,i,j}\big),\ y_{b,i,j}\Big),\qquad \lambda = 10^{-3},\ \tau = 1,
$$
with $\hat x_0$ = `predict_x0_from_eps(x_t, t, eps_hat)` **unclipped** (clipping would kill the gradient), computed in fp32.

- **Paper parity.** MarioDiffusion's `reconstruction_loss` uses exactly these logits ($\tau = 1$, raw $\hat x_0$), **sums over pixels**, averages over the batch, and uses $\lambda = 0.001$. It has **no** $\bar\alpha_t$ weight. We keep their reduction and λ and **add** the $\bar\alpha_t$ weight (ledger M9). The exact-paper variant is available as `loss.aux_ce_weighting: none` for a sanity check only.
- **Why $\bar\alpha_t$ weighting (gradient-bound derivation).** Per pixel, $\partial\,\mathrm{CE}/\partial\hat x_0 = \tau(p - e_y)$ with entries in $[-1,1]$, and $\partial\hat x_0/\partial\hat\epsilon = -\sqrt{1-\bar\alpha_t}/\sqrt{\bar\alpha_t}$. With the weight $\bar\alpha_t$:
$$
\Big\lvert\frac{\partial}{\partial\hat\epsilon}\Big[\bar\alpha_t\,\mathrm{CE}\Big]\Big\rvert \le \tau\,\sqrt{\bar\alpha_t(1-\bar\alpha_t)}\,\lvert p-e_y\rvert \le \tfrac{\tau}{2}.
$$
The CE gradient into the network is **bounded for every t** (maximum at $\bar\alpha_t = 1/2$). Without the weight it grows as $1/\sqrt{\bar\alpha_t}$: ×157 (linear) and ×20,291 (cosine) at t = 999, which is exactly where $\hat x_0$ carries no information.
- **Calibration of λ.** Per element, the MSE gradient is $2(\hat\epsilon-\epsilon)/(B D)$, i.e. ≈ $6\times10^{-4}(\hat\epsilon - \epsilon)/B$ for $D = 3328$. The CE term (summed over pixels, × λ) contributes at most $\lambda\cdot\tfrac12 = 5\times10^{-4}/B$ per element. The two are the same order of magnitude at mid-t, so λ = 0.001 with sum-over-pixels is a meaningful but not dominant regulariser.
- **CE floor.** For a perfect $\hat x_0 = x_0$ (logit +1 for the true class, −1 otherwise), $p_y = e/(e+12e^{-1}) = 0.381$, so CE = 0.965 per pixel and never reaches 0. The CE term therefore always pushes $\hat x_0$ slightly outside ±1 for the true class. λ is small so that this pressure stays negligible against the MSE term. Do **not** raise τ or λ without an experiment.

---

## 11. Why the continuous relaxation works, and discrete alternatives

- The model learns the score of the Gaussian-smoothed distribution over the 13-vertex set. At small t the smoothing is negligible and $\hat x_0$ concentrates on a vertex, and argmax recovers the tile (§1). Two practical facts make this robust here: the vertices are maximally separated (all pairwise distances equal), and the final DDIM step outputs $\hat x_0$, an estimate of class probabilities, not a noisy $x_0$.
- **Failure mode specific to relaxation:** "between-class" pixels, where $\hat x_0$ has two channels near 0 at the end of sampling, signal unresolved ambiguity. Monitor the **final-step margin** (top-1 minus top-2 of $\hat x_{0}$ at the last step; healthy median > 1.0) in validation sampling (doc 16 §11).
- **Discrete alternatives (discussion only):** D3PM with uniform or absorbing ("mask") transition matrices (Austin et al. 2021), multinomial diffusion (Hoogeboom et al. 2021), and masked generative transformers (MaskGIT-style iterative unmasking). They model categories natively and avoid clipping and argmax, but they are less standard for CFG and DDIM-style fast sampling, and they are not what the PDF or the paper evaluated. Out of scope. Mention them at the viva as the natural next step (doc 32).

---

## 12. Network building blocks (definitions the architecture doc 14 instantiates)

### 12.1 Sinusoidal time embedding (M4) — exactly as `reference/ref_unet.py`
For integer code index t and d = 128 (half = 64):
$$
\omega_k = \exp\!\Big(-\ln(10000)\,\frac{k}{64}\Big),\ k = 0..63,\qquad \mathrm{emb}(t) = \big[\sin(t\,\omega_0),\dots,\sin(t\,\omega_{63}),\ \cos(t\,\omega_0),\dots,\cos(t\,\omega_{63})\big].
$$
Then $e(t) = W_2\,\mathrm{SiLU}(W_1\mathrm{emb}(t) + b_1) + b_2\in\mathbb R^{256}$. Feed the integer **code index** (0..999) and never a rescaled t, because DDIM strided steps pass the actual $\tau_i$.

### 12.2 FiLM / AdaGN injection
In each ResBlock: $h \leftarrow \mathrm{GN}(h)\odot(1+\gamma) + \beta_{\text{sh}}$, with $[\gamma,\beta_{\text{sh}}] = W_t\,\mathrm{SiLU}(e(t)) + b_t\in\mathbb R^{2C}$. The $(1+\gamma)$ form makes $\gamma = 0$ the identity.

### 12.3 GroupNorm (8 groups, ε = 1e-5)
For channels split into G = 8 groups, $\mu_g, \sigma_g^2$ are computed over (channels-in-group × H × W) **per sample**: $\mathrm{GN}(h) = \gamma_c\frac{h-\mu_g}{\sqrt{\sigma_g^2+\varepsilon}} + \beta_c$. GroupNorm is **batch-independent**, so micro-batching with gradient accumulation (the OOM fallback, H4) is mathematically equivalent to the full batch (doc 16 §17). This is one reason it is used instead of BatchNorm.

### 12.4 SiLU
$\mathrm{SiLU}(x) = x\,\sigma(x)$.

### 12.5 Attention (M5, M6)
For queries $Q\in\mathbb R^{N\times C}$ (spatial tokens, N = H·W of that level), keys and values $K,V\in\mathbb R^{M\times C}$, with h = 4 heads of size $d_h = C/4$:
$$
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_h}} + \text{mask}\Big)V,\qquad \text{mask}_{ij} = \begin{cases}0 & \text{token } j \text{ valid}\\ -\infty & \text{token } j \text{ padding}\end{cases}
$$
- **Self-attention:** $Q,K,V$ are projections of $\mathrm{GN}(h)$ (M = N).
- **Cross-attention:** $Q = W_q\mathrm{GN}(h)$, $K = W_k\,\text{ctx}$, $V = W_v\,\text{ctx}$, with ctx $\in\mathbb R^{L\times128}$ the adapter output (M7) and the key-padding mask from the tokenizer. A row can never be fully masked, because [CLS] and [SEP] are always valid (also for `""`).
- Output: $h \leftarrow h + W_o\,\mathrm{Attn}(\cdot)$ (residual).
- Cost: $O(N^2C)$ for self-attention and $O(NLC)$ for cross-attention, both negligible here (N ≤ 64 at 8×8; the attention matmuls total 0.025 GFLOP per 16×16 sample, about 2 % of the forward pass; doc 14 §8).

### 12.6 Zero-initialisation consequence
With the output conv zero-initialised (doc 14 §6), $\epsilon_\theta\equiv0$ at step 0, so the initial loss is $\mathbb E[\epsilon^2] = 1$. **Verified: 0.997 at B = 64.** Test: `initial_loss ∈ [0.95, 1.05]`.

---

## 13. Optimisation mathematics

### 13.1 AdamW (M10) — PyTorch's decoupled form
With gradient $g_n$, lr $\eta_n$, decay $\lambda_{wd}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$:
$$
\theta \leftarrow \theta(1-\eta_n\lambda_{wd});\quad m\leftarrow\beta_1m+(1-\beta_1)g;\quad v\leftarrow\beta_2v+(1-\beta_2)g^2;\quad \hat m = \tfrac{m}{1-\beta_1^n},\ \hat v = \tfrac{v}{1-\beta_2^n};\quad \theta\leftarrow\theta-\eta_n\frac{\hat m}{\sqrt{\hat v}+\varepsilon}.
$$
Weight decay applies only to the "decay" parameter group (doc 16 §5). Total multiplicative shrink from decay alone, $\exp(-\lambda_{wd}\sum_n\eta_n)$ (verified): **0.984** over a 15k-step run, **0.939** over a 60k-step run. Weight decay is a mild regulariser here, not a dominant force.

### 13.2 Learning-rate schedule (M10): linear warm-up + cosine decay
$$
\eta_n = \begin{cases}\eta_{\max}\,\dfrac{n}{W}, & n < W\\[2mm] \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})\Big(1+\cos\big(\pi\,\tfrac{n-W}{N-W}\big)\Big), & W\le n\le N\end{cases}\qquad \eta_{\max} = 2\times10^{-4},\ \eta_{\min} = 10^{-5},\ W = 1000,
$$
with N = `train.max_steps` (15,000 in Phase 3, 60,000 in Phase 4, 40,000 for ablations). Verified values:

| n | 0 | 500 | 1k | 5k | 10k | 15k | 20k | 30k | 40k | 60k |
|---|---|---|---|---|---|---|---|---|---|---|
| N = 15k | 0 | 1.0e-4 | 2.0e-4 | 1.64e-4 | 6.38e-5 | 1.0e-5 | — | — | — | — |
| N = 60k | 0 | 1.0e-4 | 2.0e-4 | 1.98e-4 | 1.89e-4 | 1.75e-4 | 1.55e-4 | 1.08e-4 | 5.9e-5 | 1.0e-5 |

Early stopping **does not** rescale the schedule (reproducibility). A run stopped at 40k of a 60k horizon has lr ≈ 5.9e-5 there.

### 13.3 Gradient clipping
If $\lVert g\rVert_2 > 1.0$, then $g\leftarrow g/\lVert g\rVert_2$ (global norm over all trainable parameters). Log the pre-clip norm and the clipped fraction.

### 13.4 EMA (M11)
$$
\theta^{\text{EMA}}_n = d_n\,\theta^{\text{EMA}}_{n-1} + (1-d_n)\,\theta_n,\qquad d_n = \min\!\Big(0.999,\ \frac{1+n}{10+n}\Big),
$$
with n = the number of optimizer steps completed. Verified ramp: $d_0 = 0.1$, $d_{10} = 0.55$, $d_{100} = 0.918$, $d_{1000} = 0.991$, $d_{5000} = 0.998$; **$d$ reaches 0.999 at n = 8,990**. Effective averaging window $1/(1-d)$: 1,000 steps for 0.999 and 200 for 0.995. Half-life $\ln 0.5/\ln d$: **692.8 steps** (0.999) vs **138.3 steps** (0.995, the PDF value, ablated in X2). EMA averages away SGD noise (a variance reduction) at the cost of lagging the current weights (a small bias). The lag is irrelevant late in training, when the LR is small.

---

## 14. Numerical precision policy (H5)

| Quantity | Precision | Why |
|---|---|---|
| β, α, ᾱ, posterior coefficients | **float64** buffers; cast to float32 at gather | $1-\bar\alpha_t$ and $\bar\alpha_t/\bar\alpha_s$ lose digits in fp32 near t = 0 and t = T−1 |
| $x_0$, ε, $x_t$ construction | float32 | inputs to the net are cast by autocast |
| Network forward | **bf16 autocast** | Ampere/Ada tensor cores; bf16 has fp32's dynamic range (8-bit exponent) and 8 bits of mantissa, unit roundoff $2^{-8}\approx 0.0039$ |
| Network output | **cast to float32 immediately** (`.float()`) | the loss and the sampler arithmetic must not run in bf16 |
| Loss, aux-CE, $\hat x_0$ | float32 | the $1/\sqrt{\bar\alpha}$ amplification would turn bf16 roundoff of ε̂ into errors of ±0.6 (linear) to ±80 (cosine) in $\hat x_0$ at t = 999 |
| Softmax inside SDPA | fp32 accumulation (kernel-internal) | stability |
| EMA shadow weights, optimizer moments | float32 | small $(1-d)\theta$ increments vanish in bf16 |
| DDIM/RePaint updates | float32 (coefficients from fp64 buffers) | deterministic, reproducible sampling |
| Frozen MiniLM | bf16 weights, `no_grad` | inference only (doc 15) |

TF32 (`torch.backends.cuda.matmul.allow_tf32 = True`, `cudnn.allow_tf32 = True`) is enabled on both machines for the remaining fp32 matmuls/convs (both GPUs support it; identical setting on both machines, so not a per-machine knob).

---

## 15. Worked example (one pixel, sanity intuition)

A sky pixel has $x_0 = v_0 = (+1,-1,\dots,-1)$. At t = 500:
- **Linear** ($\sqrt{\bar\alpha} = 0.279$, $\sqrt{1-\bar\alpha} = 0.960$): $x_{500} = 0.279\,v_0 + 0.960\,\epsilon$. Per channel, signal ±0.28 against noise std 0.96. SNR = 0.084 < 1/8, so the pixel cannot be identified from itself; only context (floor rows, neighbours, caption) can tell the model it is sky.
- **Cosine** ($\sqrt{\bar\alpha} = 0.702$, $\sqrt{1-\bar\alpha} = 0.713$): signal ±0.70 against noise 0.71. SNR = 0.97, so the pixel is still largely identifiable locally.

The same t means very different information content under the two schedules. This is why per-bucket losses (doc 16) are always reported **per schedule** and never compared across schedules bucket by bucket.

---

## 16. Implementation map and tests

| Function (`scheduler.py`) | Equation | Test (`tests/test_scheduler.py`) |
|---|---|---|
| `make_betas(kind, T, ...)` | §3.1–3.2 | values at t ∈ {0,100,250,500,750,999} match §3.3 to 4 significant digits; monotone ᾱ; exactly 1 clipped beta for cosine |
| `q_sample(x0, t, eps)` | §2 | empirical mean/std vs theory at 5 t's (hypothesis, n = 200k) |
| `predict_x0_from_eps`, `predict_eps_from_x0` | §6 | round trip identity (fp64, < 1e-12) |
| `q_posterior_mean_variance` | §4 | equals Monte-Carlo posterior moments; $\tilde\beta_0 = 0$ handled |
| `ddpm_step` | §7.1 | oracle ε → posterior mean exact |
| `ddim_timesteps(S)` | §7.3 | equals the verified 50- and 25-lists |
| `ddim_step(..., eta, clip)` | §7.2 | **oracle-ε exact reconstruction** (G-P2); η = 1, s = t−1 equals DDPM in distribution |
| `renoise(x_s, s, t)` | §9 | mean/var match $q(x_t\mid x_0)$ (n = 200k) |
| `guided_eps(...)` | §8 | w = 1 → cond; w = 0 → uncond; batched = separate |
| `loss_simple`, `loss_channel_weighted`, `loss_aux_ce` | §5, §10 | normalised weights mean 1; aux-CE gradient bound ≤ λ·τ/2 per element (autograd check) |
| `snr(t)`, `context_free_floor` table | §5 | used by validation reports (doc 16) |
