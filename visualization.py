"""
Visualization utilities for Monte Carlo order book fair value analysis.

This module provides functions to create publication-quality plots for
visualizing Monte Carlo simulations, statistical distributions, and
trading strategy performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec


def setup_plot_style(dark_mode: bool = True):
    """
    Configure matplotlib style for consistent, professional-looking plots.
    
    Parameters
    ----------
    dark_mode : bool
        If True, use dark background style
    """
    if dark_mode:
        plt.style.use('dark_background')
    else:
        plt.style.use('seaborn-v0_8-darkgrid')
    
    # Set default parameters
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def plot_monte_carlo_paths(mc_array: np.ndarray, 
                          context_price: List[float],
                          mean_path: np.ndarray,
                          std_path: np.ndarray,
                          n_paths_to_show: int = 50,
                          figsize: Tuple[int, int] = (14, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize Monte Carlo simulation paths with confidence intervals.
    
    Parameters
    ----------
    mc_array : np.ndarray
        Array of simulated price paths (n_simulations x n_steps)
    context_price : list of float
        Historical price data
    mean_path : np.ndarray
        Mean of Monte Carlo paths
    std_path : np.ndarray
        Standard deviation of Monte Carlo paths
    n_paths_to_show : int
        Number of individual paths to display
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        If provided, save figure to this path
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot context price history
    context_steps = len(context_price)
    ax.plot(range(-context_steps, 0), context_price, 
            color='cyan', linewidth=2.5, label='Historical Price', alpha=0.9)
    
    # Plot individual Monte Carlo paths
    n_sims = min(n_paths_to_show, mc_array.shape[0])
    for i in range(n_sims):
        ax.plot(range(mc_array.shape[1]), mc_array[i], 
                color='gray', alpha=0.15, linewidth=0.8)
    
    # Plot mean path
    ax.plot(range(len(mean_path)), mean_path, 
            color='yellow', linewidth=3, label='Expected Path (Mean)', 
            linestyle='--', alpha=0.9)
    
    # Add confidence intervals
    steps = np.arange(len(mean_path))
    ax.fill_between(steps, mean_path - std_path, mean_path + std_path,
                    color='yellow', alpha=0.2, label='±1 Std Dev')
    ax.fill_between(steps, mean_path - 2*std_path, mean_path + 2*std_path,
                    color='yellow', alpha=0.1, label='±2 Std Dev')
    
    # Mark current price
    current_price = context_price[-1]
    ax.axhline(y=current_price, color='red', linestyle=':', 
              linewidth=2, alpha=0.7, label=f'Current Price: {current_price:.2f}')
    
    # Mark expected final price
    expected_final = mean_path[-1]
    ax.axhline(y=expected_final, color='lime', linestyle=':', 
              linewidth=2, alpha=0.7, label=f'Expected Final: {expected_final:.2f}')
    
    # Formatting
    ax.set_xlabel('Time Steps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Price', fontsize=13, fontweight='bold')
    ax.set_title('Monte Carlo Order Book Simulation\nPrice Path Predictions', 
                fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add vertical line separating historical and simulated
    ax.axvline(x=0, color='white', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(0.5, ax.get_ylim()[1] * 0.95, 'Simulation Start', 
           rotation=0, fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor=fig.get_facecolor())
    
    return fig


def plot_return_distribution(mc_array: np.ndarray,
                            context_price: List[float],
                            figsize: Tuple[int, int] = (14, 6),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of returns from Monte Carlo simulations.
    
    Parameters
    ----------
    mc_array : np.ndarray
        Array of simulated price paths
    context_price : list of float
        Historical price data
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    final_returns = mc_array[:, -1] - mc_array[:, 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram with KDE
    ax1.hist(final_returns, bins=50, density=True, alpha=0.7, 
            color='skyblue', edgecolor='black')
    
    # Add KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(final_returns)
    x_range = np.linspace(final_returns.min(), final_returns.max(), 200)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2.5, 
            label='KDE', alpha=0.8)
    
    # Mark mean and zero
    mean_return = np.mean(final_returns)
    ax1.axvline(mean_return, color='yellow', linestyle='--', 
               linewidth=2.5, label=f'Mean: {mean_return:.4f}')
    ax1.axvline(0, color='red', linestyle=':', linewidth=2, 
               label='Zero Return')
    
    ax1.set_xlabel('Final Return', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of Final Returns', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot to check normality
    from scipy import stats
    stats.probplot(final_returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Check)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=fig.get_facecolor())
    
    return fig


def plot_strategy_performance(equity_curve: List[float],
                             figsize: Tuple[int, int] = (14, 8),
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot strategy equity curve with performance metrics.
    
    Parameters
    ----------
    equity_curve : list of float
        Equity values over time
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
    
    # Main equity curve
    ax1 = plt.subplot(gs[0])
    equity = np.array(equity_curve)
    steps = np.arange(len(equity))
    
    ax1.plot(steps, equity, linewidth=2.5, color='lime', label='Equity Curve')
    ax1.fill_between(steps, equity_curve[0], equity, 
                    alpha=0.3, color='lime')
    
    # Mark drawdown regions
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    
    ax1.axhline(y=equity_curve[0], color='white', linestyle='--', 
               linewidth=1.5, alpha=0.5, label='Starting Capital')
    
    ax1.set_ylabel('Equity ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Strategy Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Returns
    ax2 = plt.subplot(gs[1])
    returns = np.diff(equity) / equity[:-1] * 100
    colors = ['green' if r > 0 else 'red' for r in returns]
    ax2.bar(steps[1:], returns, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='white', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Returns (%)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Drawdown
    ax3 = plt.subplot(gs[2])
    ax3.fill_between(steps, 0, drawdown, color='red', alpha=0.5)
    ax3.plot(steps, drawdown, color='darkred', linewidth=1.5)
    ax3.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add performance metrics
    total_return = (equity[-1] - equity[0]) / equity[0] * 100
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)  # Annualized
    max_dd = np.min(drawdown)
    
    metrics_text = f'Total Return: {total_return:.2f}%\n'
    metrics_text += f'Sharpe Ratio: {sharpe:.2f}\n'
    metrics_text += f'Max Drawdown: {max_dd:.2f}%'
    
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=fig.get_facecolor())
    
    return fig


def plot_sharpe_over_time(mean_path: np.ndarray,
                         std_path: np.ndarray,
                         context_price: List[float],
                         figsize: Tuple[int, int] = (12, 6),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Sharpe ratio evolution over forecast horizon.
    
    Parameters
    ----------
    mean_path : np.ndarray
        Expected price path
    std_path : np.ndarray
        Standard deviation of paths
    context_price : list of float
        Historical prices
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    ev_over_time = mean_path - context_price[-1]
    sharpe_over_time = ev_over_time / (std_path + 1e-10)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    steps = np.arange(len(mean_path))
    
    # Expected value over time
    ax1.plot(steps, ev_over_time, linewidth=2.5, color='cyan', label='Expected Value')
    ax1.fill_between(steps, 0, ev_over_time, alpha=0.3, color='cyan')
    ax1.axhline(y=0, color='white', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_ylabel('Expected Value', fontsize=11, fontweight='bold')
    ax1.set_title('Expected Value and Sharpe Ratio Over Time', 
                 fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Sharpe ratio over time
    ax2.plot(steps, sharpe_over_time, linewidth=2.5, color='yellow', label='Sharpe Ratio')
    ax2.fill_between(steps, 0, sharpe_over_time, 
                    where=(sharpe_over_time > 0), color='green', alpha=0.3,
                    label='Positive Sharpe')
    ax2.fill_between(steps, 0, sharpe_over_time, 
                    where=(sharpe_over_time < 0), color='red', alpha=0.3,
                    label='Negative Sharpe')
    ax2.axhline(y=0, color='white', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Mark optimal long and short points
    long_idx = np.argmax(sharpe_over_time)
    short_idx = np.argmin(sharpe_over_time)
    
    ax2.scatter([long_idx], [sharpe_over_time[long_idx]], 
               color='lime', s=200, marker='o', zorder=5,
               label=f'Best Long (t={long_idx})')
    ax2.scatter([short_idx], [sharpe_over_time[short_idx]], 
               color='red', s=200, marker='o', zorder=5,
               label=f'Best Short (t={short_idx})')
    
    ax2.set_xlabel('Time Steps', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=fig.get_facecolor())
    
    return fig


def plot_order_book_snapshot(lob,
                            n_levels: int = 10,
                            figsize: Tuple[int, int] = (10, 6),
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize current order book state.
    
    Parameters
    ----------
    lob : OrderBook
        Current order book
    n_levels : int
        Number of price levels to display
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract order book data
    bids = []
    asks = []
    
    # Get bid side
    if hasattr(lob, 'bids') and lob.bids:
        for price in sorted(lob.bids.keys(), reverse=True)[:n_levels]:
            qty = sum([order['qty'] for order in lob.bids[price]])
            bids.append((price * 0.001, qty))
    
    # Get ask side
    if hasattr(lob, 'asks') and lob.asks:
        for price in sorted(lob.asks.keys())[:n_levels]:
            qty = sum([order['qty'] for order in lob.asks[price]])
            asks.append((price * 0.001, qty))
    
    if bids:
        bid_prices, bid_qtys = zip(*bids)
        ax.barh(bid_prices, bid_qtys, color='green', alpha=0.7, label='Bids')
    
    if asks:
        ask_prices, ask_qtys = zip(*asks)
        ax.barh(ask_prices, [-q for q in ask_qtys], color='red', alpha=0.7, label='Asks')
    
    # Mark mid price and spread
    if bids and asks:
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        ax.axhline(y=mid_price, color='yellow', linestyle='--', 
                  linewidth=2, label=f'Mid: {mid_price:.2f}')
        ax.axhline(y=best_bid, color='lime', linestyle=':', 
                  linewidth=1.5, alpha=0.7)
        ax.axhline(y=best_ask, color='orange', linestyle=':', 
                  linewidth=1.5, alpha=0.7)
        
        ax.text(0.02, 0.98, f'Spread: {spread:.4f}', 
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.axvline(x=0, color='white', linestyle='-', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Quantity (Negative = Ask, Positive = Bid)', 
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Price', fontsize=11, fontweight='bold')
    ax.set_title('Order Book Depth', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=fig.get_facecolor())
    
    return fig
