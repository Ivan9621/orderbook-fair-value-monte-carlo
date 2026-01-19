# Quick Start Guide

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/orderbook-fair-value-monte-carlo.git
cd orderbook-fair-value-monte-carlo
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install the order book library (lobby):**
```bash
# You'll need to obtain and install the 'lobby' order book implementation
# See: https://github.com/your-lobby-repo
pip install lobby  # or clone and install from source
```

## Basic Usage

### 1. Simple Fair Value Estimation

```python
from lobby import OrderBook
from src.fair_value_model import simulate_lob, get_strategy_stats

# Initialize order book
lob = OrderBook()

# Generate historical context
context_price = simulate_lob(lob, steps_per_sequence=100)

# Run Monte Carlo simulation
stats, mean_path, std_path, _ = get_strategy_stats(
    lob, context_price, 
    simulations=500, 
    steps_per_sequence=20
)

print(f"Expected Value: {stats['neutral']['ev']:.6f}")
print(f"p-value: {stats['neutral']['p_value']:.4f}")
```

### 2. Generate Trading Signal

```python
from src.fair_value_model import get_strategy_prediction

signal, mean_path, std_path, _ = get_strategy_prediction(
    lob, context_price,
    simulations=500,
    steps_per_sequence=10
)

if signal == 1:
    print("LONG: Expected upward movement")
elif signal == -1:
    print("SHORT: Expected downward movement")
else:
    print("NEUTRAL: No significant signal")
```

### 3. Visualize Results

```python
from src.visualization import (
    setup_plot_style, 
    plot_monte_carlo_paths
)

setup_plot_style(dark_mode=True)

# You'll need to run the simulations first to get mc_array
# See examples/demo.py for complete code
fig = plot_monte_carlo_paths(
    mc_array, context_price, 
    mean_path, std_path
)
```

### 4. Run the Demo

```bash
cd examples
python demo.py
```

This will:
- Run a complete Monte Carlo simulation
- Generate trading signals
- Create visualizations
- Run a backtest

### 5. Generate Example Figures

```bash
cd examples
python generate_figures.py
```

This creates example visualizations without requiring the full lobby implementation.

## Configuration

Key parameters you can adjust:

### Monte Carlo Parameters

```python
simulations = 500           # Number of Monte Carlo paths
steps_per_sequence = 20     # Time steps to simulate
events_per_step = 50        # Order events per step
order_qty_mean = 5.0        # Mean order size
order_qty_std = 1.0         # Std dev of order size
maker_volatility = 0.01     # Price volatility for limit orders
```

### Statistical Testing

```python
significance_level = 0.05   # p-value threshold (5%)
```

## Understanding the Output

### Statistics Dictionary

```python
stats = {
    "neutral": {
        "ev": 0.0123,          # Expected value
        "sharpe": 1.45,        # Sharpe-like ratio
        "t_stat": 3.21,        # t-statistic
        "p_value": 0.0013,     # p-value
        "z_score": 2.89,       # z-score
        "at_index": -1         # Time index
    },
    "long": {...},             # Optimal long strategy stats
    "short": {...}             # Optimal short strategy stats
}
```

### Interpreting Results

**p-value < 0.05**: Statistically significant movement expected
- If `ev > 0`: Bullish signal
- If `ev < 0`: Bearish signal

**Sharpe Ratio**:
- `|sharpe| > 1`: Strong risk-adjusted signal
- `|sharpe| > 2`: Very strong signal

**z-score**:
- `|z| > 1.96`: Significant at 5% level
- `|z| > 2.58`: Significant at 1% level

## Common Issues

### 1. Import Error: No module named 'lobby'

You need to install the lobby order book library. See installation instructions above.

### 2. Slow Performance

- Reduce `simulations` (try 100-200 for faster results)
- Reduce `steps_per_sequence`
- Ensure you're using parallel processing (automatic with joblib)

### 3. Memory Issues

- Reduce `simulations`
- Process results in batches
- Use generators instead of storing all paths

## Next Steps

1. **Read the Documentation**: Check `docs/MATHEMATICAL_DERIVATIONS.md` for theory
2. **Experiment with Parameters**: Try different configurations
3. **Implement with Real Data**: Connect to your order book data source
4. **Add Features**: Extend the model with transaction costs, etc.

## Getting Help

- Open an issue on GitHub
- Check the examples directory for more code samples
- Read the academic papers referenced in the README

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

**Remember**: This is for research and education. Always conduct your own analysis before trading.
