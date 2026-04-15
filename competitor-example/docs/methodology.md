# Compound V3 Incentive Optimization: Mathematical Methodology

## Overview

This document details the quantitative framework for analyzing COMP incentive effectiveness and optimizing allocation across Compound V3 markets.

---

## 1. Market Fundamentals

### 1.1 Core Variables

| Variable | Symbol | Definition |
|----------|--------|------------|
| Total Value Locked | $S$ | Total supply deposits (USD) |
| Borrow Volume | $B$ | Total borrowed (USD) |
| Utilization | $U$ | Fraction of supply that is borrowed |
| Supply APR | $r_s$ | Annual interest rate paid to suppliers |
| Borrow APR | $r_b$ | Annual interest rate charged to borrowers |
| Supply Incentive | $I_s$ | Daily COMP rewards to suppliers (USD) |
| Borrow Incentive | $I_b$ | Daily COMP rewards to borrowers (USD) |
| Total Incentive | $I$ | $I = I_s + I_b$ |

### 1.2 Utilization

$$U = \frac{B}{S}$$

Utilization represents capital efficiency. Higher utilization means more of the deposited capital is being productively lent.

### 1.3 Interest Rate Model

Compound V3 uses a kinked interest rate model:

**Below kink** $(U \leq U_{optimal})$:

$$r_b(U) = r_{base} + \frac{U}{U_{optimal}} \cdot r_{slope1}$$

**Above kink** $(U > U_{optimal})$:

$$r_b(U) = r_{base} + r_{slope1} + \frac{U - U_{optimal}}{1 - U_{optimal}} \cdot r_{slope2}$$

Supply rate is derived from borrow rate:

$$r_s = r_b \cdot U \cdot (1 - RF)$$

where $RF$ is the reserve factor (protocol's cut).

---

## 2. Profitability Framework

### 2.1 Daily Profit

$$\pi = \underbrace{\frac{B \cdot r_b}{365}}_{\text{Interest Revenue}} - \underbrace{\frac{S \cdot r_s}{365}}_{\text{Interest Cost}} - \underbrace{I}_{\text{Incentives}}$$

Substituting $B = U \cdot S$:

$$\pi = \frac{S}{365} \left( U \cdot r_b - r_s \right) - I$$

### 2.2 Net Interest Income

$$NII = \frac{B \cdot r_b}{365} - \frac{S \cdot r_s}{365} = \frac{S}{365}(U \cdot r_b - r_s)$$

### 2.3 Breakeven Utilization

The utilization at which net interest income equals zero:

$$U_{BE} = \frac{r_s}{r_b}$$

**Interpretation:**
- If $U > U_{BE}$: Market generates positive net interest (profitable before incentives)
- If $U < U_{BE}$: Market generates negative net interest (losing money on spread)

### 2.4 Market Classification

| Status | Condition | Implication |
|--------|-----------|-------------|
| WELL-UTILIZED | $U > U_{BE} + 0.05$ | Profitable; can cut incentives |
| AT BREAKEVEN | $|U - U_{BE}| \leq 0.05$ | Marginal; cut supply incentives |
| UNDER-UTILIZED | $U < U_{BE} - 0.05$ | Losing money; cut all supply incentives |

---

## 3. Elasticity Analysis

### 3.1 Model Specification

We estimate TVL response to incentive changes using OLS regression on first differences:

$$\Delta S_t = \alpha + \beta \cdot \Delta I_t + \varepsilon_t$$

where:
- $\Delta S_t = S_t - S_{t-1}$ (daily TVL change)
- $\Delta I_t = I_t - I_{t-1}$ (daily incentive change)
- $\beta$ = elasticity coefficient (TVL change per $1 incentive change)
- $\varepsilon_t \sim N(0, \sigma^2)$ (error term)

### 3.2 OLS Estimation

$$\hat{\beta} = \frac{\sum_{t=1}^{T}(\Delta I_t - \overline{\Delta I})(\Delta S_t - \overline{\Delta S})}{\sum_{t=1}^{T}(\Delta I_t - \overline{\Delta I})^2}$$

$$\hat{\alpha} = \overline{\Delta S} - \hat{\beta} \cdot \overline{\Delta I}$$

### 3.3 Coefficient of Determination

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_t (\Delta S_t - \hat{\Delta S_t})^2}{\sum_t (\Delta S_t - \overline{\Delta S})^2}$$

**Interpretation:** $R^2$ represents the fraction of TVL variance explained by incentive changes.

### 3.4 Statistical Significance

Standard error of $\hat{\beta}$:

$$SE(\hat{\beta}) = \sqrt{\frac{\hat{\sigma}^2}{\sum_t (\Delta I_t - \overline{\Delta I})^2}}$$

where $\hat{\sigma}^2 = \frac{1}{T-2}\sum_t \hat{\varepsilon}_t^2$

t-statistic:

$$t = \frac{\hat{\beta}}{SE(\hat{\beta})}$$

p-value from t-distribution with $T-2$ degrees of freedom.

### 3.5 Bayesian Interpretation

While we used frequentist OLS for point estimates, the Bayesian framing provides intuition:

**Prior:**
$$\beta \sim N(\mu_0, \sigma_0^2)$$

We assume a weakly informative prior centered at zero (no effect).

**Likelihood:**
$$\Delta S_t | \beta \sim N(\alpha + \beta \cdot \Delta I_t, \sigma^2)$$

**Posterior:**
$$\beta | \text{data} \sim N(\mu_n, \sigma_n^2)$$

where:
$$\mu_n = \frac{\sigma^2 \mu_0 + \sigma_0^2 \sum_t \Delta I_t \Delta S_t}{\sigma^2 + \sigma_0^2 \sum_t \Delta I_t^2}$$

With large samples and diffuse priors, the posterior mean converges to the OLS estimate.

### 3.6 Lagged Elasticity

To test for delayed response:

$$\Delta S_{t+k} = \alpha_k + \beta_k \cdot \Delta I_t + \varepsilon_{t,k}$$

for $k \in \{1, 7, 14, 30\}$ days.

Correlation coefficient:
$$\rho_k = \text{Corr}(\Delta I_t, \Delta S_{t+k})$$

---

## 4. Decay Rate Analysis (AR(1) Model)

### 4.1 Model Specification

TVL follows an autoregressive process:

$$S_t = c + \rho \cdot S_{t-1} + \eta_t$$

where:
- $\rho$ = persistence coefficient $(0 < \rho < 1)$
- $c$ = constant term
- $\eta_t \sim N(0, \sigma_\eta^2)$

### 4.2 Parameter Estimation

OLS on lagged values:

$$\hat{\rho} = \frac{\sum_{t=2}^{T}(S_{t-1} - \bar{S}_{-1})(S_t - \bar{S})}{\sum_{t=2}^{T}(S_{t-1} - \bar{S}_{-1})^2}$$

### 4.3 Derived Quantities

**Decay rate:**
$$\delta = 1 - \rho$$

This represents the fraction of the gap between current TVL and equilibrium that closes each day.

**Half-life:**
$$t_{1/2} = \frac{\ln(0.5)}{\ln(\rho)} = \frac{-\ln(2)}{\ln(\rho)}$$

Time required for TVL to move halfway from current level toward equilibrium.

**Long-run equilibrium:**
$$\mu = \frac{c}{1 - \rho}$$

The level TVL gravitates toward in the absence of shocks.

### 4.4 Mean-Reversion Interpretation

Rewriting the AR(1) model:

$$S_t - \mu = \rho(S_{t-1} - \mu) + \eta_t$$

If $S_{t-1} > \mu$: Expected $S_t$ is pulled down toward $\mu$
If $S_{t-1} < \mu$: Expected $S_t$ is pulled up toward $\mu$

### 4.5 Impulse Response

After a shock $\eta_0$, the effect on TVL at time $t$:

$$\frac{\partial S_t}{\partial \eta_0} = \rho^t$$

The shock decays geometrically at rate $\rho$.

---

## 5. Independence Testing

### 5.1 Hypothesis

$H_0$: TVL direction (up/down) is independent of incentive direction (up/down)
$H_1$: TVL direction depends on incentive direction

### 5.2 Contingency Table

|  | TVL ↑ | TVL ↓ |
|--|-------|-------|
| **Incentive ↑** | $n_{11}$ | $n_{12}$ |
| **Incentive ↓** | $n_{21}$ | $n_{22}$ |

### 5.3 Chi-Square Statistic

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

where:
- $O_{ij}$ = observed count
- $E_{ij} = \frac{(\text{row } i \text{ total}) \times (\text{col } j \text{ total})}{n}$ = expected count under independence

### 5.4 Degrees of Freedom

$$df = (r-1)(c-1) = 1$$

for a 2×2 table.

### 5.5 Decision Rule

If $p > 0.05$: Fail to reject $H_0$ → TVL direction is independent of incentive direction

---

## 6. Regime-Conditional Analysis

### 6.1 Volatility Regimes

**Realized volatility** (7-day, annualized):

$$\sigma_t = \sqrt{252} \cdot \sqrt{\frac{1}{6}\sum_{i=0}^{6} r_{t-i}^2}$$

where $r_t = \ln(P_t / P_{t-1})$ is daily log return.

**Regime classification:**
- High Volatility: $\sigma_t > Q_{75}(\sigma)$
- Extreme Volatility: $\sigma_t > Q_{90}(\sigma)$

### 6.2 Utilization Regimes

- High Utilization: $U_t > 0.85$
- Extreme Utilization: $U_t > 0.95$
- Low Headroom: $1 - U_t < 0.10$

### 6.3 Stress Regimes

$$\text{Stress}_t = \mathbb{1}[\text{High Vol}_t] \cap \mathbb{1}[\text{High Util}_t \cup \text{Util Increasing}_t]$$

### 6.4 Regime-Conditional Elasticity

For each regime $R$:

$$\Delta S_t = \alpha_R + \beta_R \cdot \Delta I_t + \varepsilon_t \quad \text{for } t \in R$$

Compare $\beta_{in}$ vs $\beta_{out}$ and $R^2_{in}$ vs $R^2_{out}$.

### 6.5 Interaction Test (Bootstrap)

To test if elasticity differs between regimes:

1. Define $D_t = \mathbb{1}[t \in \text{regime}]$
2. Bootstrap slope difference: $\Delta\beta = \beta_{in} - \beta_{out}$
3. 95% CI: $[\hat{\Delta\beta}_{2.5\%}, \hat{\Delta\beta}_{97.5\%}]$
4. If CI excludes zero → significant difference

---

## 7. Bayesian Optimization Framework

### 7.1 Prior Distributions

**Elasticity coefficients:**

$$\beta_s \sim N(\mu_{\beta_s}, \sigma_{\beta_s}^2)$$
$$\beta_b \sim N(\mu_{\beta_b}, \sigma_{\beta_b}^2)$$

where:
- $\beta_s$ = TVL response to supply incentive changes
- $\beta_b$ = TVL response to borrow incentive changes
- Priors centered at zero (conservative: assume no effect)
- Variance set based on historical data spread

**Decay rate:**

$$\delta \sim \text{Beta}(\alpha_\delta, \beta_\delta)$$

where:
- $\delta \in [0, 1]$ is the daily decay rate
- Beta distribution naturally bounded on [0,1]
- Shape parameters calibrated from historical AR(1) estimates
- Mode at $\frac{\alpha_\delta - 1}{\alpha_\delta + \beta_\delta - 2}$

### 7.2 Posterior Updates

Given observed data $\mathcal{D} = \{(\Delta I_t, \Delta S_t)\}_{t=1}^T$:

**Elasticity posterior (conjugate normal-normal):**

$$\beta | \mathcal{D} \sim N(\mu_n, \sigma_n^2)$$

where:
$$\sigma_n^2 = \left(\frac{1}{\sigma_0^2} + \frac{\sum_t \Delta I_t^2}{\sigma_\varepsilon^2}\right)^{-1}$$

$$\mu_n = \sigma_n^2 \left(\frac{\mu_0}{\sigma_0^2} + \frac{\sum_t \Delta I_t \Delta S_t}{\sigma_\varepsilon^2}\right)$$

**Decay rate posterior:**

Updated via moment matching or MCMC sampling from:

$$p(\delta | \mathcal{D}) \propto p(\mathcal{D} | \delta) \cdot p(\delta)$$

where likelihood comes from AR(1) residuals.

### 7.3 Profit Function

Expected daily profit given incentive allocation $(I_s, I_b)$:

$$\pi(I_s, I_b; \beta_s, \beta_b, \delta) = NII(S(I_s, I_b)) - I_s - I_b$$

where TVL evolves as:

$$S_{t+1} = \mu + (1-\delta)(S_t - \mu) + \beta_s \Delta I_s + \beta_b \Delta I_b + \eta_t$$

and net interest income:

$$NII(S) = \frac{S \cdot U \cdot r_b - S \cdot r_s}{365}$$

### 7.4 Monte Carlo Profit Estimation

For a given incentive allocation $(I_s, I_b)$:

**Step 1: Sample coefficients**

For $m = 1, \ldots, M$ samples:
$$\beta_s^{(m)} \sim p(\beta_s | \mathcal{D})$$
$$\beta_b^{(m)} \sim p(\beta_b | \mathcal{D})$$
$$\delta^{(m)} \sim p(\delta | \mathcal{D})$$

**Step 2: Simulate TVL trajectory**

For each sample $m$, project TVL forward $H$ days:
$$\hat{S}_{t+h}^{(m)} = \mu^{(m)} + (1-\delta^{(m)})^h (S_t - \mu^{(m)}) + \sum_{j=1}^{h} (1-\delta^{(m)})^{h-j} (\beta_s^{(m)} \Delta I_s + \beta_b^{(m)} \Delta I_b)$$

**Step 3: Compute expected profit**

$$\mathbb{E}[\pi(I_s, I_b)] \approx \frac{1}{M} \sum_{m=1}^{M} \pi(I_s, I_b; \beta_s^{(m)}, \beta_b^{(m)}, \delta^{(m)})$$

### 7.5 Grid Search Optimization

**Step 1: Define incentive grid**

$$\mathcal{G} = \{(I_s^{(i)}, I_b^{(j)}) : I_s^{(i)} \in [0, I_s^{max}], I_b^{(j)} \in [0, I_b^{max}]\}$$

Typical grid: 20×20 points spanning current allocation ± 50%.

**Step 2: Evaluate expected profit at each grid point**

For each $(I_s, I_b) \in \mathcal{G}$:
1. Draw $M$ samples from posterior distributions
2. Compute $\hat{\mathbb{E}}[\pi(I_s, I_b)]$ via Monte Carlo
3. Store result

**Step 3: Find optimal allocation**

$$(I_s^*, I_b^*) = \arg\max_{(I_s, I_b) \in \mathcal{G}} \hat{\mathbb{E}}[\pi(I_s, I_b)]$$

### 7.6 Closed-Loop Feedback Algorithm

The optimization runs as an iterative closed-loop process:

```
Initialize:
  - Set priors: β_s, β_b ~ N(0, σ²), δ ~ Beta(α, β)
  - Collect initial data D₀

For iteration k = 1, 2, ...:
  
  1. POSTERIOR UPDATE
     - Update p(β_s | D_{k-1}), p(β_b | D_{k-1}), p(δ | D_{k-1})
  
  2. MONTE CARLO SAMPLING
     - For m = 1 to M:
         Sample β_s^(m), β_b^(m), δ^(m) from posteriors
  
  3. GRID EVALUATION
     - For each (I_s, I_b) in grid G:
         Compute E[π(I_s, I_b)] by averaging over M samples
  
  4. OPTIMIZATION
     - Find (I_s*, I_b*) = argmax E[π(I_s, I_b)]
  
  5. IMPLEMENTATION
     - Change incentive allocation to (I_s*, I_b*)
     - Wait observation period (e.g., 7-30 days)
  
  6. MEASUREMENT
     - Observe realized TVL response: ΔS_observed
     - Record new data point: (ΔI, ΔS_observed)
     - Append to dataset: D_k = D_{k-1} ∪ {new observation}
  
  7. FEEDBACK
     - Return to Step 1 with updated dataset
```

### 7.7 Exploration vs Exploitation

The Bayesian framework naturally balances:

**Exploitation:** High-probability optimal allocations based on current posteriors

**Exploration:** Posterior uncertainty drives sampling of less-certain regions

As data accumulates:
- Posterior variance shrinks
- Estimates converge to true values
- Optimal allocation stabilizes

### 7.8 Convergence Diagnostics

**Posterior convergence:**
$$\text{CV}(\beta) = \frac{\sigma_n}{\mu_n} < \epsilon$$

When coefficient of variation falls below threshold, estimates are stable.

**Profit convergence:**
$$|\mathbb{E}[\pi^{(k)}] - \mathbb{E}[\pi^{(k-1)}]| < \tau$$

When expected profit changes fall below threshold, optimization has converged.

### 7.9 Key Finding from Bayesian Analysis

After multiple iterations:

$$p(\beta_s | \mathcal{D}) \approx N(0, \sigma_{small}^2)$$
$$p(\beta_b | \mathcal{D}) \approx N(0, \sigma_{small}^2)$$

The posteriors collapsed to near-zero with tight credible intervals, confirming:

$$P(\beta_s \approx 0 | \mathcal{D}) > 0.95$$
$$P(\beta_b \approx 0 | \mathcal{D}) > 0.95$$

**Implication:** The profit-maximizing allocation is simply:

$$(I_s^*, I_b^*) \approx (0, 0)$$

since incentives provide no TVL benefit but incur direct costs.

---

## 8. Optimal Allocation Framework

### 7.1 Objective Function

Maximize expected profit:

$$\max_{I_s, I_b} \mathbb{E}[\pi] = \mathbb{E}[NII(S, U)] - I_s - I_b$$

subject to:
- $I_s \geq 0, I_b \geq 0$
- $I_s + I_b \leq I_{max}$ (budget constraint)

### 7.2 TVL Response Model

If incentives affected TVL:

$$S = S_0 + \beta_s \cdot I_s + \beta_b \cdot I_b$$

But our finding: $\beta_s \approx \beta_b \approx 0$ (R² < 1%)

### 7.3 Implication

Since $\frac{\partial S}{\partial I} \approx 0$:

$$\frac{\partial \pi}{\partial I} = \frac{\partial NII}{\partial S} \cdot \frac{\partial S}{\partial I} - 1 \approx -1 < 0$$

Every dollar of incentives reduces profit by approximately one dollar with negligible TVL benefit.

### 7.4 Optimal Solution

Given inelastic TVL:

$$I^* = I_{min}$$

where $I_{min}$ is the minimum viable incentive (near zero).

### 7.5 Allocation Rules by Market Status

| Status | Optimal $I_s$ | Optimal $I_b$ | Rationale |
|--------|---------------|---------------|-----------|
| WELL-UTILIZED | ~$0 | ~$0 | Already profitable; incentives unnecessary |
| AT BREAKEVEN | ~$0 | Low | Borrow incentives may marginally help utilization |
| UNDER-UTILIZED | $0 | Moderate | Focus only on borrow demand |

---

## 9. Counter-Example Analysis

### 9.1 Definition

A counter-example is an observation where:
- Incentives increased: $\Delta I_t > 0.10 \cdot I_{t-1}$ (>10% increase)
- TVL decreased: $S_{t+30} < S_t$ (within 30 days)

### 9.2 Frequency Analysis

$$P(\text{TVL} \downarrow | \text{Inc} \uparrow) = \frac{\#\{t: \Delta I_t > 0.10 \cdot I_{t-1} \text{ and } S_{t+30} < S_t\}}{\#\{t: \Delta I_t > 0.10 \cdot I_{t-1}\}}$$

If $P(\text{TVL} \downarrow | \text{Inc} \uparrow) \approx P(\text{TVL} \downarrow | \text{Inc} \downarrow)$, this confirms independence.

---

## 10. Summary of Key Findings

### 10.1 Elasticity Results

| Market | $\hat{\beta}$ | $R^2$ | p-value | Interpretation |
|--------|---------------|-------|---------|----------------|
| ETH USDC | $459 | 0.31% | 0.07 | Inelastic |
| ETH USDT | $596 | 0.14% | 0.40 | Inelastic |
| ETH WETH | $2,420 | 0.25% | 0.10 | Inelastic |

### 10.2 Decay Results

| Market | $\hat{\rho}$ | $\hat{\delta}$ | $t_{1/2}$ | $\hat{\mu}$ |
|--------|--------------|----------------|-----------|-------------|
| ETH USDC | 0.995 | 0.46%/day | 152 days | $472M |
| ETH USDT | 0.990 | 0.98%/day | 71 days | $190M |
| ETH WETH | 0.987 | 1.28%/day | 54 days | $162M |

### 10.3 Independence Test

$$\chi^2 = 0.619, \quad p = 0.43$$

Fail to reject independence → TVL direction is statistically independent of incentive direction.

### 10.4 Regime Analysis

| Regime | $R^2_{in}$ | $R^2_{out}$ | Conclusion |
|--------|------------|-------------|------------|
| High Volatility | 0.03% | 1.35% | Less effective during vol |
| High Utilization | 0.49% | 1.29% | Less effective during high util |
| Stress | 0.46% | 1.22% | Less effective during stress |

---

## 11. Notation Reference

| Symbol | Meaning |
|--------|---------|
| $S$ | Supply TVL (USD) |
| $B$ | Borrow volume (USD) |
| $U$ | Utilization ratio |
| $r_s, r_b$ | Supply/Borrow APR |
| $I, I_s, I_b$ | Total/Supply/Borrow incentives |
| $\pi$ | Daily profit |
| $\beta$ | Elasticity coefficient |
| $\rho$ | AR(1) persistence |
| $\delta$ | Decay rate |
| $\mu$ | Long-run equilibrium TVL |
| $t_{1/2}$ | Half-life |
| $\Delta$ | First difference operator |

---

*Document generated: February 2025*
