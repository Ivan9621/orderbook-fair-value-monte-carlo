# Mathematical Derivations and Theory

## Detailed Mathematical Framework for Order Book Fair Value Estimation

This document provides detailed mathematical derivations for the Monte Carlo approach to order book fair value estimation.

---

## 1. Stochastic Order Book Model

### 1.1 State Space Definition

The order book state at time $t$ can be represented as:

$$\mathcal{L}_t = \left\{ (p_i^b, q_i^b)_{i=1}^{N_b}, (p_j^a, q_j^a)_{j=1}^{N_a} \right\}$$

where:
- $(p_i^b, q_i^b)$ represents price and quantity at bid level $i$
- $(p_j^a, q_j^a)$ represents price and quantity at ask level $j$
- $N_b$ and $N_a$ are the number of bid and ask levels respectively

The bid side satisfies: $p_1^b > p_2^b > ... > p_{N_b}^b$

The ask side satisfies: $p_1^a < p_2^a < ... < p_{N_a}^a$

### 1.2 Spread Dynamics

The bid-ask spread is defined as:

$$S_t = p_1^a(t) - p_1^b(t)$$

The mid-price is:

$$m_t = \frac{p_1^b(t) + p_1^a(t)}{2}$$

---

## 2. Order Arrival Process

### 2.1 Poisson Process Model

Orders arrive according to a non-homogeneous Poisson process with intensity $\lambda_t$. The probability of $k$ orders arriving in time interval $[t, t+\Delta t]$ is:

$$P(N(t+\Delta t) - N(t) = k) = \frac{(\lambda \Delta t)^k e^{-\lambda \Delta t}}{k!}$$

For small $\Delta t$:

$$P(\text{1 order in } \Delta t) \approx \lambda \Delta t$$
$$P(\text{0 orders in } \Delta t) \approx 1 - \lambda \Delta t$$

### 2.2 Order Characteristics

Each order is characterized by a tuple $(s, \tau, p, q)$ where:
- $s \in \{\text{buy}, \text{sell}\}$ is the side
- $\tau \in \{\text{market}, \text{limit}\}$ is the type
- $p$ is the limit price (for limit orders)
- $q$ is the quantity

#### Side Distribution

$$P(s = \text{buy}) = P(s = \text{sell}) = \frac{1}{2}$$

This assumes no directional bias in order flow (which can be relaxed for more sophisticated models).

#### Type Distribution

$$P(\tau = \text{market}) = P(\tau = \text{limit}) = \frac{1}{2}$$

In practice, the proportion of market vs. limit orders varies by asset and market conditions.

#### Quantity Distribution

Order quantities follow a truncated normal distribution:

$$q \sim \mathcal{N}^+(\mu_q, \sigma_q^2)$$

where $\mathcal{N}^+$ denotes the positive part of the normal distribution:

$$f_Q(q) = \frac{1}{Z} \cdot \frac{1}{\sigma_q \sqrt{2\pi}} e^{-\frac{(q-\mu_q)^2}{2\sigma_q^2}} \cdot \mathbb{1}_{q > 0}$$

with normalization constant:

$$Z = \int_0^{\infty} \frac{1}{\sigma_q \sqrt{2\pi}} e^{-\frac{(q-\mu_q)^2}{2\sigma_q^2}} dq = \frac{1}{2}\left(1 + \text{erf}\left(\frac{\mu_q}{\sigma_q \sqrt{2}}\right)\right)$$

#### Price Distribution for Limit Orders

**Buy Limit Orders:**

For a buy limit order when $p_1^b$ and $p_1^a$ exist, the price is drawn from:

$$p \sim \mathcal{U}[p_1^b, p_1^a]$$

This ensures the limit order is placed within or at the current spread.

**Sell Limit Orders:**

Similarly for sell orders:

$$p \sim \mathcal{U}[p_1^b, p_1^a]$$

**Market Orders (Crossing Orders):**

Market orders cross the spread. For buy market orders:

$$p \sim \mathcal{U}[p_1^a, p_1^a(1 + \sigma_m)]$$

For sell market orders:

$$p \sim \mathcal{U}[p_1^b(1 - \sigma_m), p_1^b]$$

where $\sigma_m$ is the market volatility parameter.

---

## 3. Price Formation and Execution

### 3.1 Execution Mechanism

When a market order (or aggressive limit order) crosses the spread, execution occurs at the best available prices.

For a buy order of quantity $Q$ at price $p$:
- If $p \geq p_1^a$: Order executes against ask side
- Execution proceeds through price levels until $Q$ is filled or book is exhausted
- Remaining unfilled quantity becomes a limit order

The execution price is the volume-weighted average:

$$p_{\text{exec}} = \frac{\sum_{i} p_i \cdot q_i}{\sum_{i} q_i}$$

### 3.2 Last Trade Price

The last trade price $P_t$ is updated whenever execution occurs:

$$P_t = p_{\text{exec}}(\text{last execution})$$

This becomes the reference price for computing returns.

---

## 4. Monte Carlo Simulation Framework

### 4.1 Simulation Algorithm

For each Monte Carlo iteration $i \in \{1, 2, ..., N_{\text{sim}}\}$:

1. **Initialize**: Start with current order book state $\mathcal{L}_0^{(i)} = \mathcal{L}_{\text{current}}$

2. **For each time step** $t \in \{1, 2, ..., T\}$:
   
   a. **Generate orders**: Sample $M$ orders from the order arrival process
   
   b. **Process orders**: For each order:
      - Match against opposite side if crossing spread
      - Add to book if not fully executed
      - Update $P_t^{(i)}$ if trade occurs
   
   c. **Record price**: Store $P_t^{(i)}$

3. **Return path**: $\{P_1^{(i)}, P_2^{(i)}, ..., P_T^{(i)}\}$

### 4.2 Convergence Properties

By the Strong Law of Large Numbers, the sample mean converges almost surely:

$$\frac{1}{N_{\text{sim}}} \sum_{i=1}^{N_{\text{sim}}} P_T^{(i)} \xrightarrow{a.s.} \mathbb{E}[P_T]$$

The rate of convergence is given by the Central Limit Theorem:

$$\sqrt{N_{\text{sim}}} \left( \frac{1}{N_{\text{sim}}} \sum_{i=1}^{N_{\text{sim}}} P_T^{(i)} - \mathbb{E}[P_T] \right) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

where $\sigma^2 = \text{Var}(P_T)$.

---

## 5. Statistical Estimation

### 5.1 Expected Value Estimation

The expected price at time $\tau$ is estimated by:

$$\hat{\mathbb{E}}[P_\tau] = \frac{1}{N_{\text{sim}}} \sum_{i=1}^{N_{\text{sim}}} P_\tau^{(i)}$$

The standard error of this estimate is:

$$\text{SE}(\hat{\mathbb{E}}[P_\tau]) = \frac{\hat{\sigma}_\tau}{\sqrt{N_{\text{sim}}}}$$

where:

$$\hat{\sigma}_\tau = \sqrt{\frac{1}{N_{\text{sim}}-1} \sum_{i=1}^{N_{\text{sim}}} \left(P_\tau^{(i)} - \hat{\mathbb{E}}[P_\tau]\right)^2}$$

### 5.2 Confidence Intervals

The $(1-\alpha)$ confidence interval for $\mathbb{E}[P_\tau]$ is:

$$\text{CI}_{1-\alpha} = \hat{\mathbb{E}}[P_\tau] \pm z_{\alpha/2} \cdot \text{SE}(\hat{\mathbb{E}}[P_\tau])$$

where $z_{\alpha/2}$ is the critical value from the standard normal distribution.

For $\alpha = 0.05$ (95% confidence): $z_{0.025} = 1.96$

### 5.3 Expected Profit/Loss

Given current price $P_0$, the expected profit/loss is:

$$\text{EV}(\tau) = \hat{\mathbb{E}}[P_\tau] - P_0$$

For a long position of size $Q$:

$$\text{P/L}_{\text{long}} = Q \cdot (\hat{\mathbb{E}}[P_\tau] - P_0)$$

For a short position:

$$\text{P/L}_{\text{short}} = Q \cdot (P_0 - \hat{\mathbb{E}}[P_\tau])$$

---

## 6. Hypothesis Testing

### 6.1 One-Sample t-Test

We test whether the expected return differs significantly from zero:

$$H_0: \mu_{\Delta P} = 0$$
$$H_1: \mu_{\Delta P} \neq 0$$

where $\Delta P^{(i)} = P_T^{(i)} - P_0^{(i)}$ are the final returns.

The test statistic is:

$$t = \frac{\bar{\Delta P}}{\hat{\sigma}_{\Delta P} / \sqrt{N_{\text{sim}}}}$$

Under $H_0$, $t \sim t_{N_{\text{sim}}-1}$ (Student's t-distribution).

The p-value is:

$$p = 2 \cdot P(T > |t|)$$

where $T \sim t_{N_{\text{sim}}-1}$.

We reject $H_0$ if $p < \alpha$ (typically $\alpha = 0.05$).

### 6.2 Z-Score

The z-score measures how many standard errors the mean is from the current price:

$$z = \frac{\hat{\mathbb{E}}[P_T] - P_0}{\text{SE}(\hat{\mathbb{E}}[P_T])} = \frac{\hat{\mathbb{E}}[P_T] - P_0}{\hat{\sigma}_T / \sqrt{N_{\text{sim}}}}$$

For large $N_{\text{sim}}$, by the CLT:

$$z \sim \mathcal{N}(0, 1) \text{ under } H_0: \mathbb{E}[P_T] = P_0$$

Interpretation:
- $|z| < 1.96$: Not significant at 5% level
- $1.96 \leq |z| < 2.58$: Significant at 5% level
- $|z| \geq 2.58$: Significant at 1% level

---

## 7. Risk-Adjusted Metrics

### 7.1 Sharpe-like Ratio

We define a Sharpe-like ratio for the expected move:

$$\mathcal{S}(\tau) = \frac{\text{EV}(\tau)}{\hat{\sigma}_\tau} = \frac{\hat{\mathbb{E}}[P_\tau] - P_0}{\hat{\sigma}_\tau}$$

This measures the expected return per unit of volatility.

Properties:
- $\mathcal{S} > 0$: Expected price increase relative to volatility
- $\mathcal{S} < 0$: Expected price decrease relative to volatility
- $|\mathcal{S}| > 1$: Strong signal relative to noise

### 7.2 Information Ratio

The information ratio can be computed as:

$$\text{IR}(\tau) = \frac{\hat{\mathbb{E}}[P_\tau] - P_0}{\text{TrackingError}}$$

where the tracking error is the volatility of the difference from the benchmark (current price).

---

## 8. Optimal Holding Period

### 8.1 Long Position

The optimal holding period for a long position maximizes the Sharpe ratio:

$$\tau^*_{\text{long}} = \arg\max_{\tau \in \{1, ..., T\}} \mathcal{S}(\tau)$$

This can be solved by computing $\mathcal{S}(\tau)$ for all $\tau$ and selecting the maximum.

### 8.2 Short Position

Similarly for short positions:

$$\tau^*_{\text{short}} = \arg\min_{\tau \in \{1, ..., T\}} \mathcal{S}(\tau)$$

### 8.3 Dynamic Optimization

For continuous time, this becomes a stochastic control problem:

$$\max_{\tau} \mathbb{E}\left[ \frac{P_\tau - P_0}{\sqrt{\text{Var}(P_\tau)}} \right]$$

subject to constraints on holding time and transaction costs.

---

## 9. Extensions and Improvements

### 9.1 Incorporating Transaction Costs

With proportional transaction costs $c$:

$$\text{P/L}_{\text{net}} = Q \cdot (\hat{\mathbb{E}}[P_\tau] - P_0) - c \cdot Q \cdot (P_0 + \hat{\mathbb{E}}[P_\tau])$$

The break-even price move is:

$$\Delta P_{\text{min}} = c \cdot (P_0 + \hat{\mathbb{E}}[P_\tau])$$

### 9.2 Market Impact

For large orders, price impact can be modeled as:

$$\mathbb{E}[\text{Impact}] = \gamma \cdot Q^{\beta}$$

where typical values are $\beta \in [0.5, 0.7]$ and $\gamma$ depends on market liquidity.

### 9.3 Confidence-Based Position Sizing

Position size can be scaled by statistical confidence:

$$Q^* = Q_{\max} \cdot \Phi\left(\frac{|z|}{2}\right)$$

where $\Phi$ is the standard normal CDF, ensuring larger positions only when signal is strong.

### 9.4 Multi-Asset Extension

For $n$ assets, the joint price process can be modeled with correlation matrix $\Sigma$:

$$\begin{pmatrix} P_1^{(i)}(\tau) \\ P_2^{(i)}(\tau) \\ \vdots \\ P_n^{(i)}(\tau) \end{pmatrix} \sim \mathcal{N}\left(\boldsymbol{\mu}(\tau), \Sigma(\tau)\right)$$

This allows for correlation-based trading strategies.

---

## 10. Model Validation

### 10.1 Out-of-Sample Testing

The model should be validated using:

1. **Walk-forward analysis**: Rolling window testing
2. **Cross-validation**: Multiple time periods
3. **Monte Carlo validation**: Simulating from known distributions

### 10.2 Reality Checks

Essential checks include:

- **Distribution of returns**: Should match empirical data
- **Spread dynamics**: Should match observed spreads
- **Trade frequency**: Should match actual execution rates
- **Price impact**: Should match measured impact

### 10.3 Calibration

Parameters should be calibrated to historical data:

$$(\mu_q^*, \sigma_q^*, \sigma_m^*, \lambda^*) = \arg\min_{\theta} \sum_{t=1}^{T_{\text{hist}}} L(P_t^{\text{obs}}, P_t^{\text{sim}}(\theta))$$

where $L$ is a loss function (e.g., MSE, MAE, or likelihood).

---

## References

### Core Papers

1. Cont, R., Stoikov, S., & Talreja, R. (2010). A stochastic model for order book dynamics. *Operations Research*, 58(3), 549-563.

2. Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217-224.

3. Glasserman, P. (2003). *Monte Carlo methods in financial engineering*. Springer.

4. Bouchaud, J. P., Mezard, M., & Potters, M. (2002). Statistical properties of stock order books: empirical results and models. *Quantitative Finance*, 2(4), 251-256.

### Additional Reading

5. Foucault, T., Kadan, O., & Kandel, E. (2005). Limit order book as a market for liquidity. *The Review of Financial Studies*, 18(4), 1171-1217.

6. Rosu, I. (2009). A dynamic model of the limit order book. *The Review of Financial Studies*, 22(11), 4601-4641.

7. Huang, W., Lehalle, C. A., & Rosenbaum, M. (2015). Simulating and analyzing order book data: The queue-reactive model. *Journal of the American Statistical Association*, 110(509), 107-122.

---

*Document Version: 1.0*  
*Last Updated: January 2026*
