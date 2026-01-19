# Order Book Fair Value Estimation using Monte Carlo Simulation

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A sophisticated framework for estimating the fair value of assets using Monte Carlo simulation of limit order book dynamics. This approach combines microstructure modeling with statistical inference to generate trading signals based on expected price movements.

## 📊 Overview

This repository implements a Monte Carlo-based approach to compute the fair value of an asset from its order book state. By simulating thousands of possible future order flow scenarios, we can estimate the expected price trajectory and assess the statistical significance of predicted movements.

### Key Features

- **Monte Carlo Order Book Simulation**: Realistic modeling of limit order book dynamics
- **Statistical Fair Value Estimation**: Compute expected values with confidence intervals
- **Signal Generation**: Automated trading signals based on statistical significance
- **Comprehensive Visualization**: Publication-quality plots for analysis
- **Parallel Processing**: Efficient computation using joblib

## 🔬 Mathematical Framework

### 1. Order Book Dynamics Model

The limit order book (LOB) is modeled as a discrete-time stochastic process. At each time step $t$, the order book state is characterized by:

$$\mathcal{L}_t = \{(\text{Bid}_t, Q^b_t), (\text{Ask}_t, Q^a_t)\}$$

where:
- $\text{Bid}_t$ and $\text{Ask}_t$ are the best bid and ask prices
- $Q^b_t$ and $Q^a_t$ are the quantities available at these prices

#### Order Arrival Process

New orders arrive according to a compound Poisson process with the following characteristics:

**Order Type Distribution:**
$$\mathbb{P}(\text{Market Order}) = \mathbb{P}(\text{Limit Order}) = 0.5$$

**Order Side Distribution:**
$$\mathbb{P}(\text{Buy}) = \mathbb{P}(\text{Sell}) = 0.5$$

**Quantity Distribution:**
$$Q \sim \mathcal{N}(\mu_q, \sigma_q^2), \quad Q \geq \epsilon$$

where $\mu_q$ is the mean order size, $\sigma_q$ is the standard deviation, and $\epsilon$ is a minimum quantity threshold.

**Price Placement (Limit Orders):**

For limit orders, prices are drawn from a distribution centered around the current spread:

- **Buy Limit Orders:** $P \sim \mathcal{U}[\text{Bid}_t, \text{Ask}_t]$
- **Sell Limit Orders:** $P \sim \mathcal{U}[\text{Bid}_t, \text{Ask}_t]$

For market orders (aggressive orders that cross the spread):

- **Buy Market Orders:** $P \sim \mathcal{U}[\text{Bid}_t, \text{Ask}_t \cdot (1 + \sigma_m)]$
- **Sell Market Orders:** $P \sim \mathcal{U}[\text{Bid}_t \cdot (1 - \sigma_m), \text{Ask}_t]$

where $\sigma_m$ is the maker volatility parameter.

### 2. Monte Carlo Fair Value Estimation

#### Price Path Simulation

For each Monte Carlo iteration $i \in \{1, ..., N\}$, we simulate a price path:

$$\mathcal{P}^{(i)} = \{P^{(i)}_1, P^{(i)}_2, ..., P^{(i)}_T\}$$

where $T$ is the number of time steps and each $P^{(i)}_t$ represents the last traded price at time $t$.

#### Expected Value Computation

The fair value at time horizon $\tau$ is estimated as:

$$\hat{V}(\tau) = \frac{1}{N} \sum_{i=1}^{N} P^{(i)}_\tau$$

with standard error:

$$\text{SE}(\hat{V}(\tau)) = \frac{\sigma_\tau}{\sqrt{N}}$$

where:

$$\sigma_\tau = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (P^{(i)}_\tau - \hat{V}(\tau))^2}$$

#### Expected Profit/Loss

Given current price $P_0$, the expected profit/loss is:

$$\text{EV}(\tau) = \hat{V}(\tau) - P_0$$

### 3. Statistical Significance Testing

To ensure we only trade when there is strong statistical evidence, we employ multiple significance tests:

#### T-Test for Mean Difference

We test the null hypothesis $H_0: \mathbb{E}[\Delta P] = 0$ using a one-sample t-test on final returns:

$$\Delta P^{(i)} = P^{(i)}_T - P^{(i)}_0$$

The t-statistic is:

$$t = \frac{\bar{\Delta P}}{\text{SE}(\Delta P)} = \frac{\bar{\Delta P}}{s / \sqrt{N}}$$

where $s$ is the sample standard deviation of returns.

Under $H_0$, $t \sim t_{N-1}$ (Student's t-distribution with $N-1$ degrees of freedom).

#### Z-Score of Mean Drift

We compute the z-score of the mean price drift:

$$z = \frac{\hat{V}(\tau) - P_0}{\sigma_\tau / \sqrt{N}}$$

This measures how many standard errors the expected value is from the current price.

#### Sharpe-like Ratio

We define a risk-adjusted metric similar to the Sharpe ratio:

$$\mathcal{S}(\tau) = \frac{\text{EV}(\tau)}{\sigma_\tau}$$

This represents the expected profit per unit of risk (standard deviation).

### 4. Optimal Holding Period

The optimal holding period for a long position is:

$$\tau^*_{\text{long}} = \arg\max_{\tau} \mathcal{S}(\tau)$$

And for a short position:

$$\tau^*_{\text{short}} = \arg\min_{\tau} \mathcal{S}(\tau)$$

### 5. Signal Generation

A trading signal is generated according to:

$$\text{Signal} = \begin{cases}
+1 & \text{if } p\text{-value} < \alpha \text{ and } \text{EV}(T) > 0 \text{ (Long)} \\
-1 & \text{if } p\text{-value} < \alpha \text{ and } \text{EV}(T) < 0 \text{ (Short)} \\
0 & \text{otherwise (Neutral)}
\end{cases}$$

where $\alpha$ is the significance level (typically 0.05).

## 📚 Theoretical Foundation

This approach is grounded in several areas of financial mathematics and market microstructure:

### Key References

1. **Market Microstructure Theory:**
   - Cont, R., Stoikov, S., & Talreja, R. (2010). "A Stochastic Model for Order Book Dynamics." *Operations Research*, 58(3), 549-563.
   - [DOI: 10.1287/opre.1090.0780](https://doi.org/10.1287/opre.1090.0780)
   
2. **High-Frequency Trading:**
   - Avellaneda, M., & Stoikov, S. (2008). "High-frequency trading in a limit order book." *Quantitative Finance*, 8(3), 217-224.
   - [DOI: 10.1080/14697680701381228](https://doi.org/10.1080/14697680701381228)

3. **Monte Carlo Methods in Finance:**
   - Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
   - [DOI: 10.1007/978-0-387-21617-1](https://doi.org/10.1007/978-0-387-21617-1)

4. **Order Book Shape and Price Impact:**
   - Bouchaud, J. P., Mezard, M., & Potters, M. (2002). "Statistical properties of stock order books: empirical results and models." *Quantitative Finance*, 2(4), 251-256.
   - [DOI: 10.1088/1469-7688/2/4/301](https://doi.org/10.1088/1469-7688/2/4/301)

5. **Price Discovery in Order Books:**
   - Hasbrouck, J. (1991). "Measuring the Information Content of Stock Trades." *The Journal of Finance*, 46(1), 179-207.
   - [DOI: 10.1111/j.1540-6261.1991.tb03749.x](https://doi.org/10.1111/j.1540-6261.1991.tb03749.x)

### Additional Resources

- **Order Flow and Price Formation:**
  - Gould, M. D., Porter, M. A., Williams, S., McDonald, M., Fenn, D. J., & Howison, S. D. (2013). "Limit order books." *Quantitative Finance*, 13(11), 1709-1742.
  
- **Statistical Arbitrage:**
  - Avellaneda, M., & Lee, J. H. (2010). "Statistical arbitrage in the US equities market." *Quantitative Finance*, 10(7), 761-782.

## 🚀 Installation

### Requirements

```bash
Python 3.8+
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
seaborn>=0.11.0
pandas>=1.3.0
joblib>=1.0.0
tqdm>=4.60.0
```

### Setup

```bash
# Clone the repository
git clone https://github.com/Ivan9621/orderbook-fair-value-monte-carlo.git
cd orderbook-fair-value-monte-carlo

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage

### Basic Example

```python
from lobby import OrderBook
from src.fair_value_model import (
    simulate_lob, 
    get_strategy_prediction,
    get_strategy_stats
)
from src.visualization import (
    plot_monte_carlo_paths,
    plot_return_distribution,
    setup_plot_style
)

# Initialize order book
lob = OrderBook()

# Generate historical context
context_price = simulate_lob(
    lob, 
    random_seed=True, 
    steps_per_sequence=100, 
    events_per_step=50
)

# Run Monte Carlo simulation and get statistics
stats, mean_path, std_path, _ = get_strategy_stats(
    lob,
    context_price,
    simulations=1000,
    steps_per_sequence=20
)

# Print statistics
print(f"Expected Value: {stats['neutral']['ev']:.6f}")
print(f"Sharpe Ratio: {stats['neutral']['sharpe']:.4f}")
print(f"p-value: {stats['neutral']['p_value']:.4f}")

# Visualize results
setup_plot_style(dark_mode=True)

# Monte Carlo paths
fig1 = plot_monte_carlo_paths(
    mc_array, 
    context_price, 
    mean_path, 
    std_path,
    save_path='figures/monte_carlo_paths.png'
)

# Return distribution
fig2 = plot_return_distribution(
    mc_array, 
    context_price,
    save_path='figures/return_distribution.png'
)
```

### Generate Trading Signals

```python
# Get trading signal
signal, mean_path, std_path, _ = get_strategy_prediction(
    lob,
    context_price,
    simulations=500,
    steps_per_sequence=10,
    significance_level=0.05
)

if signal == 1:
    print("LONG signal: Expected upward movement")
elif signal == -1:
    print("SHORT signal: Expected downward movement")
else:
    print("NEUTRAL: No statistically significant movement")
```

### Backtest Example

```python
from tqdm import tqdm

def backtest_strategy(initial_capital=10000, n_periods=100):
    lob = OrderBook()
    context_price = simulate_lob(
        lob, 
        random_seed=True, 
        steps_per_sequence=300
    )
    
    equity = [initial_capital]
    
    for i in tqdm(range(n_periods)):
        # Simulate next price
        new_price = simulate_lob(
            lob, 
            random_seed=True, 
            steps_per_sequence=10
        )[0]
        
        # Get signal
        signal, _, _, _ = get_strategy_prediction(
            lob, 
            context_price, 
            simulations=100,
            steps_per_sequence=10
        )
        
        # Update equity based on signal
        pnl = (new_price - context_price[-1]) * signal
        equity.append(pnl + equity[-1])
        context_price.append(new_price)
    
    return equity

# Run backtest
equity_curve = backtest_strategy()

# Visualize performance
from src.visualization import plot_strategy_performance
fig = plot_strategy_performance(equity_curve, save_path='figures/backtest.png')
```

## 📊 Example Results

### Monte Carlo Simulation Paths

![image alt](https://github.com/Ivan9621/orderbook-fair-value-monte-carlo/blob/66bd0134402bc107fdda6748db844bee40cce298/example_monte_carlo_paths.png)

The plot shows:
- Historical price data (cyan)
- Individual Monte Carlo paths (gray, transparent)
- Expected path (yellow dashed line)
- Confidence intervals (yellow shaded regions)

### Return Distribution Analysis

![image alt](https://github.com/Ivan9621/orderbook-fair-value-monte-carlo/blob/66bd0134402bc107fdda6748db844bee40cce298/example_return_distribution.png)

Statistical analysis of simulated returns:
- Histogram with kernel density estimate
- Q-Q plot for normality testing
- Mean return and confidence intervals

### Strategy Performance

![image alt](https://github.com/Ivan9621/orderbook-fair-value-monte-carlo/blob/66bd0134402bc107fdda6748db844bee40cce298/example_backtest_performance.png)

Backtest results showing:
- Equity curve over time
- Period-by-period returns
- Drawdown analysis
- Performance metrics (Sharpe ratio, max drawdown, total return)

## 🔧 Configuration Parameters

### Monte Carlo Simulation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `simulations` | 100 | Number of Monte Carlo paths |
| `steps_per_sequence` | 10 | Time steps per simulation |
| `events_per_step` | 50 | Order events per time step |
| `order_qty_mean` | 5.0 | Mean order quantity |
| `order_qty_std` | 1.0 | Std dev of order quantity |
| `maker_volatility` | 0.01 | Price volatility for limit orders |

### Statistical Testing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `significance_level` | 0.05 | p-value threshold for signals |

## 🎯 Performance Considerations

### Computational Complexity

- **Single simulation:** $O(T \cdot E)$ where $T$ is time steps and $E$ is events per step
- **Monte Carlo:** $O(N \cdot T \cdot E)$ where $N$ is number of simulations
- **Parallel speedup:** Near-linear with number of CPU cores

### Optimization Tips

1. **Reduce simulations** for faster results (100-500 is usually sufficient)
2. **Use fewer time steps** for shorter-term predictions
3. **Enable parallel processing** (automatically uses all CPU cores)
4. **Cache order book states** when running multiple analyses

## 📈 Interpreting Results

### Statistical Significance

- **p-value < 0.05**: Strong evidence of directional movement
- **|z-score| > 2**: Movement is more than 2 standard errors from zero
- **Sharpe ratio > 1**: Risk-adjusted return is favorable

### Signal Confidence

- **High confidence:** p < 0.01, |Sharpe| > 2, large number of simulations
- **Medium confidence:** p < 0.05, |Sharpe| > 1, moderate simulations
- **Low confidence:** p < 0.1, |Sharpe| < 1, or few simulations

### Practical Considerations

⚠️ **Important Notes:**

1. This model assumes **no transaction costs** - real trading has fees and slippage
2. **Market impact** is not modeled - large orders move prices
3. **Historical patterns** may not repeat - past ≠ future
4. **Statistical significance** ≠ profitability - manage risk appropriately
5. The model is **research-oriented** - not financial advice

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Order book dynamics inspired by the work of Cont, Stoikov, and Talreja
- Monte Carlo framework based on Glasserman's methodologies
- Statistical testing approaches from classical econometrics

## 📧 Contact

For questions, suggestions, or collaborations, please open an issue on GitHub.

## 🔗 Related Projects

- [LOBSTER](https://lobsterdata.com/) - Limit Order Book System
- [Zipline](https://github.com/quantopian/zipline) - Algorithmic Trading Library
- [QuantLib](https://www.quantlib.org/) - Quantitative Finance Library

---

**Disclaimer:** This software is for educational and research purposes only. It is not intended to be financial advice or a recommendation to buy or sell any security. Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
