"""
Complete demonstration of Order Book Fair Value Monte Carlo model.

This script demonstrates all key features:
1. Order book simulation
2. Monte Carlo fair value estimation
3. Statistical significance testing
4. Visualization
5. Backtesting
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fair_value_model import (
    simulate_lob,
    get_strategy_stats,
    get_strategy_prediction
)
from visualization import (
    setup_plot_style,
    plot_monte_carlo_paths,
    plot_return_distribution,
    plot_strategy_performance,
    plot_sharpe_over_time
)

# Note: You need to have the 'lobby' OrderBook implementation
# This is a placeholder - replace with your actual OrderBook
try:
    from lobby import OrderBook
except ImportError:
    print("Warning: 'lobby' module not found. Using mock OrderBook.")
    print("Please install the lobby order book library to run this example.")
    
    # Mock OrderBook for demonstration
    class OrderBook:
        def __init__(self):
            self.bids = {}
            self.asks = {}
            self._last_price = 10000
            
        def getBestBid(self):
            if not self.bids:
                return None
            return max(self.bids.keys())
        
        def getBestAsk(self):
            if not self.asks:
                return None
            return min(self.asks.keys())
        
        def processOrder(self, order, verbose, log):
            # Simple mock implementation
            price = int(order['price'] * 1000)
            
            # Simulate a trade
            trades = []
            if np.random.random() > 0.3:  # 70% chance of trade
                trades.append({'price': price})
            
            # Update book state
            if order['side'] == 'bid':
                if price not in self.bids:
                    self.bids[price] = []
                self.bids[price].append(order)
            else:
                if price not in self.asks:
                    self.asks[price] = []
                self.asks[price].append(order)
            
            # Clean up old orders randomly
            if len(self.bids) > 20:
                self.bids.pop(list(self.bids.keys())[0], None)
            if len(self.asks) > 20:
                self.asks.pop(list(self.asks.keys())[0], None)
            
            return trades, 0


def run_example_simulation():
    """Run a single Monte Carlo simulation example."""
    print("=" * 70)
    print("EXAMPLE 1: Monte Carlo Fair Value Estimation")
    print("=" * 70)
    
    # Setup
    setup_plot_style(dark_mode=True)
    
    # Initialize order book
    lob = OrderBook()
    
    # Generate historical price context
    print("\n[1/5] Generating historical price context...")
    context_price = simulate_lob(
        lob,
        random_seed=True,
        steps_per_sequence=100,
        events_per_step=50
    )
    print(f"Generated {len(context_price)} historical price points")
    print(f"Current price: {context_price[-1]:.4f}")
    
    # Run Monte Carlo simulation
    print("\n[2/5] Running Monte Carlo simulations...")
    stats, mean_path, std_path, _ = get_strategy_stats(
        lob,
        context_price,
        simulations=500,
        steps_per_sequence=20,
        events_per_step=50
    )
    
    # Display statistics
    print("\n[3/5] Statistical Results:")
    print("-" * 50)
    print(f"Expected Value: {stats['neutral']['ev']:.6f}")
    print(f"Sharpe Ratio: {stats['neutral']['sharpe']:.4f}")
    print(f"t-statistic: {stats['neutral']['t_stat']:.4f}")
    print(f"p-value: {stats['neutral']['p_value']:.6f}")
    print(f"z-score: {stats['neutral']['z_score']:.4f}")
    
    # Interpret results
    print("\n[4/5] Interpretation:")
    if stats['neutral']['p_value'] < 0.05:
        if stats['neutral']['ev'] > 0:
            print("✓ SIGNIFICANT BULLISH SIGNAL")
            print(f"  Expected upward movement of {stats['neutral']['ev']:.6f}")
        else:
            print("✓ SIGNIFICANT BEARISH SIGNAL")
            print(f"  Expected downward movement of {stats['neutral']['ev']:.6f}")
    else:
        print("○ NO SIGNIFICANT SIGNAL")
        print("  Price movement not statistically significant")
    
    print(f"\nOptimal Long Hold Time: {stats['long']['at_index']} steps")
    print(f"Optimal Short Hold Time: {stats['short']['at_index']} steps")
    
    # Generate visualizations
    print("\n[5/5] Generating visualizations...")
    
    # Note: We need the mc_array for visualization
    # Re-run to get the array (in production, you'd modify the function to return it)
    from fair_value_model import run_sim
    from joblib import Parallel, delayed
    
    mcs = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_sim)(lob, 50, 5, 1, 0.01, 20) for _ in range(500)
    )
    mc_array = np.array(mcs)
    
    # Create plots
    fig1 = plot_monte_carlo_paths(
        mc_array,
        context_price,
        mean_path,
        std_path,
        n_paths_to_show=100,
        save_path='../figures/example_monte_carlo_paths.png'
    )
    print("  ✓ Saved Monte Carlo paths plot")
    
    fig2 = plot_return_distribution(
        mc_array,
        context_price,
        save_path='../figures/example_return_distribution.png'
    )
    print("  ✓ Saved return distribution plot")
    
    fig3 = plot_sharpe_over_time(
        mean_path,
        std_path,
        context_price,
        save_path='../figures/example_sharpe_over_time.png'
    )
    print("  ✓ Saved Sharpe ratio plot")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("Example 1 Complete!")
    print("=" * 70)


def run_backtest_example():
    """Run a simple backtest example."""
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Backtesting Trading Strategy")
    print("=" * 70)
    
    from tqdm import tqdm
    
    # Initialize
    lob = OrderBook()
    initial_capital = 10000
    n_periods = 50
    
    print(f"\nInitial Capital: ${initial_capital:,.2f}")
    print(f"Periods: {n_periods}")
    
    # Generate initial context
    print("\n[1/3] Generating initial order book state...")
    context_price = simulate_lob(
        lob,
        random_seed=True,
        steps_per_sequence=200,
        events_per_step=50
    )
    
    # Run backtest
    print("\n[2/3] Running backtest with Monte Carlo signals...")
    equity = [initial_capital]
    signals = []
    
    for i in tqdm(range(n_periods), desc="Backtesting"):
        # Simulate next price
        new_price = simulate_lob(
            lob,
            random_seed=True,
            steps_per_sequence=5,
            events_per_step=50
        )[0]
        
        # Get signal
        signal, _, _, _ = get_strategy_prediction(
            lob,
            context_price,
            simulations=100,
            steps_per_sequence=10,
            significance_level=0.05
        )
        
        signals.append(signal)
        
        # Calculate P&L
        price_change = new_price - context_price[-1]
        pnl = price_change * signal
        equity.append(pnl + equity[-1])
        
        # Update context
        context_price.append(new_price)
    
    # Calculate performance metrics
    print("\n[3/3] Performance Metrics:")
    print("-" * 50)
    
    equity_arr = np.array(equity)
    returns = np.diff(equity_arr) / equity_arr[:-1] * 100
    
    total_return = (equity[-1] - equity[0]) / equity[0] * 100
    avg_return = np.mean(returns)
    vol = np.std(returns)
    sharpe = avg_return / vol * np.sqrt(252) if vol > 0 else 0
    
    max_equity = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - max_equity) / max_equity * 100
    max_dd = np.min(drawdown)
    
    win_rate = np.sum(np.array(returns) > 0) / len(returns) * 100
    
    print(f"Final Equity: ${equity[-1]:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Return per Trade: {avg_return:.3f}%")
    
    # Signal statistics
    long_signals = signals.count(1)
    short_signals = signals.count(-1)
    neutral_signals = signals.count(0)
    
    print(f"\nSignal Distribution:")
    print(f"  Long: {long_signals} ({long_signals/len(signals)*100:.1f}%)")
    print(f"  Short: {short_signals} ({short_signals/len(signals)*100:.1f}%)")
    print(f"  Neutral: {neutral_signals} ({neutral_signals/len(signals)*100:.1f}%)")
    
    # Visualize
    print("\n[4/4] Generating performance visualization...")
    fig = plot_strategy_performance(
        equity,
        save_path='../figures/example_backtest_performance.png'
    )
    print("  ✓ Saved backtest performance plot")
    
    plt.show()
    
    print("\n" + "=" * 70)
    print("Example 2 Complete!")
    print("=" * 70)


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "   Order Book Fair Value Estimation - Monte Carlo Approach   ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        # Run examples
        run_example_simulation()
        run_backtest_example()
        
        print("\n\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nGenerated figures saved to: ../figures/")
        print("\nNext steps:")
        print("  1. Examine the generated plots in the figures/ directory")
        print("  2. Experiment with different parameters")
        print("  3. Implement with real order book data")
        print("  4. Add transaction costs and slippage for realistic backtests")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
