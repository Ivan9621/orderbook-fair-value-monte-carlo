"""
Order Book Fair Value Model using Monte Carlo Simulation

This module implements a Monte Carlo-based approach to estimate the fair value
of an asset based on order book dynamics. The model simulates multiple possible
future paths of the order book to compute expected values and statistical measures.
"""

import numpy as np
import random
import copy
from scipy.stats import ttest_1samp
from joblib import Parallel, delayed
from typing import Dict, Tuple, List, Optional


def generate_random_trade(lob, 
                          order_qty_mean: float = 5, 
                          order_qty_std: float = 1, 
                          maker_volatility: float = 0.01) -> Dict:
    """
    Generate a random trade order based on current order book state.
    
    This function simulates realistic order flow by considering the current
    bid-ask spread and generating both limit and market orders with appropriate
    price distributions.
    
    Parameters
    ----------
    lob : OrderBook
        Current limit order book state
    order_qty_mean : float
        Mean order quantity (Gaussian distribution)
    order_qty_std : float
        Standard deviation of order quantity
    maker_volatility : float
        Price volatility parameter for limit orders
        
    Returns
    -------
    dict
        Order dictionary with keys: type, side, qty, price, tid
        
    Notes
    -----
    The function handles three cases:
    1. Empty book: Creates initial liquidity at fallback price
    2. One-sided book: Creates opposing side to establish spread
    3. Two-sided book: Generates market/limit orders based on current spread
    """
    fall_back_on_empty_book = 10000 * 1000
    side = random.choice(["bid", "ask"])
    order_type = random.choice(["limit", "market"])
    
    best_bid = lob.getBestBid()
    best_ask = lob.getBestAsk()
    
    # Empty book case
    if best_ask is None and best_bid is None:
        return {
            'type': 'limit', 
            'side': side, 
            'qty': max(random.gauss(order_qty_mean, order_qty_std), 0.00001), 
            'price': random.uniform(
                fall_back_on_empty_book * (1.00 + maker_volatility), 
                fall_back_on_empty_book * (1.00 - maker_volatility)
            ) * 0.001, 
            'tid': "noise"
        }
    
    # Only asks, no bids
    if best_bid is None:
        return {
            'type': 'limit', 
            'side': "bid", 
            'qty': max(random.gauss(order_qty_mean, order_qty_std), 0.00001), 
            'price': random.uniform(best_ask, best_ask * (1.00 - maker_volatility)) * 0.001, 
            'tid': "noise"
        }
    
    # Only bids, no asks
    if best_ask is None:
        return {
            'type': 'limit', 
            'side': "ask", 
            'qty': max(random.gauss(order_qty_mean, order_qty_std), 0.00001), 
            'price': random.uniform(best_bid, best_bid * (1.00 + maker_volatility)) * 0.001, 
            'tid': "noise"
        }
    
    # Two-sided book
    if side == "bid":
        if order_type == "market":
            # Aggressive buy (market order)
            return {
                'type': 'limit', 
                'side': "bid", 
                'qty': max(random.gauss(order_qty_mean, order_qty_std), 0.00001), 
                'price': random.uniform(best_bid, best_ask * (1.00 + maker_volatility)) * 0.001, 
                'tid': "noise"
            }
        else:  # limit order
            return {
                'type': 'limit', 
                'side': "bid", 
                'qty': max(random.gauss(order_qty_mean, order_qty_std), 0.00001), 
                'price': random.uniform(best_bid, best_ask) * 0.001, 
                'tid': "noise"
            }
    else:  # ask side
        if order_type == "market":
            # Aggressive sell (market order)
            return {
                'type': 'limit', 
                'side': "ask", 
                'qty': max(random.gauss(order_qty_mean, order_qty_std), 0.00001), 
                'price': random.uniform(best_bid, best_bid * (1.00 - maker_volatility)) * 0.001, 
                'tid': "noise"
            }
        else:  # limit order
            return {
                'type': 'limit', 
                'side': "ask", 
                'qty': max(random.gauss(order_qty_mean, order_qty_std), 0.00001), 
                'price': random.uniform(best_bid, best_ask) * 0.001, 
                'tid': "noise"
            }


def simulate_lob(lob, 
                events_per_step: int = 50, 
                steps_per_sequence: int = 50, 
                order_qty_mean: float = 5, 
                order_qty_std: float = 1, 
                maker_volatility: float = 0.01, 
                random_seed: bool = False, 
                seed: int = 8) -> List[float]:
    """
    Simulate order book evolution over multiple time steps.
    
    This function generates a synthetic price path by simulating order arrivals
    and executions in the limit order book.
    
    Parameters
    ----------
    lob : OrderBook
        Initial order book state
    events_per_step : int
        Number of order events per time step
    steps_per_sequence : int
        Number of time steps to simulate
    order_qty_mean : float
        Mean order quantity
    order_qty_std : float
        Standard deviation of order quantity
    maker_volatility : float
        Price volatility for limit orders
    random_seed : bool
        If True, use random seed; if False, use fixed seed
    seed : int
        Fixed seed value when random_seed=False
        
    Returns
    -------
    list of float
        Price history over the simulation period
        
    Notes
    -----
    Each time step consists of multiple order events. The last trade price
    in each step is recorded as the price for that step.
    """
    if not random_seed:
        random.seed(seed)
    
    price_history = []
    last_price = None
    
    for step in range(steps_per_sequence):
        for event in range(events_per_step):
            new_order = generate_random_trade(
                lob, order_qty_mean, order_qty_std, maker_volatility
            )
            trades, idNum = lob.processOrder(new_order, False, False)
            
            if len(trades) != 0:
                last_price = trades[-1]["price"] * 0.001
        
        if last_price is not None:
            price_history.append(last_price)
    
    return price_history


def compute_statistics(mc_array: np.ndarray, 
                      context_price: List[float], 
                      print_output: bool = True, 
                      at_index: int = -1) -> Optional[Dict]:
    """
    Compute statistical measures from Monte Carlo simulation results.
    
    This function calculates key metrics including expected value, Sharpe ratio,
    t-statistics, and z-scores to assess the statistical significance of
    predicted price movements.
    
    Parameters
    ----------
    mc_array : np.ndarray
        Array of shape (n_simulations, n_steps) containing simulated price paths
    context_price : list of float
        Historical price context
    print_output : bool
        If True, print statistics; if False, return as dictionary
    at_index : int
        Time index at which to compute statistics (-1 for final step)
        
    Returns
    -------
    dict or None
        Dictionary containing statistical measures:
        - sharpe: Sharpe ratio of expected move
        - ev: Expected value relative to current price
        - t_stat: t-statistic for mean difference from zero
        - p_value: p-value for two-tailed t-test
        - z_score: z-score of the mean
        - at_index: time index used
        
    Notes
    -----
    The Sharpe ratio here represents the expected value divided by the
    standard deviation, providing a risk-adjusted measure of the expected move.
    
    Statistical significance is assessed via:
    1. t-test: Tests if final returns differ significantly from zero
    2. z-score: Standardized measure of the mean drift
    """
    final_returns = mc_array[:, at_index] - mc_array[:, 0]
    mean_path = mc_array.mean(axis=0)
    std_path = mc_array.std(axis=0)
    
    # t-test: Are final returns significantly different from zero?
    t, p = ttest_1samp(final_returns, popmean=0)
    
    drift = mean_path - mean_path[0]
    
    # Expected value relative to current price
    ev = mean_path[at_index] - context_price[-1]
    
    mean_final = drift[at_index]
    std_final = std_path[at_index]
    
    # z-score of the mean
    z = mean_final / (std_final / np.sqrt(mc_array.shape[0]))
    
    # Sharpe-like ratio: expected move / volatility
    sharpe = ev / std_final if std_final != 0 else 0
    
    if print_output:
        print(f"t-statistic: {t:.4f}, p-value: {p:.4f}")
        print(f"z-score: {z:.4f}")
        print(f"Sharpe ratio: {sharpe:.4f}")
        print(f"Expected value: {ev:.6f}")
        return None
    
    # Check for invalid values
    if np.isnan(sharpe) or np.isnan(ev) or np.isnan(t) or np.isnan(p) or np.isnan(z):
        print("Warning: NaN values detected in statistics")
        print(mc_array)
    
    return {
        "sharpe": sharpe, 
        "ev": ev, 
        "t_stat": t, 
        "p_value": p, 
        "z_score": z, 
        "at_index": at_index
    }


def run_sim(context_lob, 
           events_per_step: int, 
           order_qty_mean: float = 5, 
           order_qty_std: float = 1, 
           maker_volatility: float = 0.01, 
           steps_per_sequence: int = 50) -> List[float]:
    """
    Run a single Monte Carlo simulation path.
    
    This is a wrapper function for parallel execution of order book simulations.
    
    Parameters
    ----------
    context_lob : OrderBook
        Current order book state (will be deep copied)
    events_per_step : int
        Number of order events per time step
    order_qty_mean : float
        Mean order quantity
    order_qty_std : float
        Standard deviation of order quantity
    maker_volatility : float
        Price volatility for limit orders
    steps_per_sequence : int
        Number of time steps to simulate
        
    Returns
    -------
    list of float
        Simulated price path
    """
    return simulate_lob(
        lob=copy.deepcopy(context_lob),
        events_per_step=events_per_step,
        steps_per_sequence=steps_per_sequence,
        order_qty_mean=order_qty_mean,
        order_qty_std=order_qty_std,
        maker_volatility=maker_volatility,
        random_seed=True
    )


def get_strategy_stats(context_lob, 
                      context_price: List[float], 
                      events_per_step: int = 50, 
                      order_qty_mean: float = 5, 
                      order_qty_std: float = 1, 
                      maker_volatility: float = 0.01, 
                      simulations: int = 100, 
                      steps_per_sequence: int = 10) -> Tuple[Dict, np.ndarray, np.ndarray, object]:
    """
    Compute strategy statistics using Monte Carlo simulation.
    
    This function runs multiple simulations of the order book and computes
    statistics for optimal long and short holding periods.
    
    Parameters
    ----------
    context_lob : OrderBook
        Current order book state
    context_price : list of float
        Historical price context
    events_per_step : int
        Number of order events per time step
    order_qty_mean : float
        Mean order quantity
    order_qty_std : float
        Standard deviation of order quantity
    maker_volatility : float
        Price volatility for limit orders
    simulations : int
        Number of Monte Carlo paths to simulate
    steps_per_sequence : int
        Number of time steps in each simulation
        
    Returns
    -------
    stats : dict
        Statistics for long, short, and neutral strategies
    mean_path : np.ndarray
        Mean price path across all simulations
    std_path : np.ndarray
        Standard deviation of price paths
    context_lob : OrderBook
        The input order book (returned for convenience)
        
    Notes
    -----
    The function identifies optimal holding periods by maximizing/minimizing
    the Sharpe ratio over time, providing long and short strategy statistics.
    """
    # Run parallel Monte Carlo simulations
    mcs = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_sim)(
            context_lob, events_per_step, order_qty_mean, 
            order_qty_std, maker_volatility, steps_per_sequence
        ) for _ in range(simulations)
    )
    
    mcs_array = np.array(mcs)
    
    mean_path = mcs_array.mean(axis=0)
    std_path = mcs_array.std(axis=0)
    
    # Expected value and Sharpe ratio over time
    ev_over_time = mean_path - context_price[-1]
    sharpe_over_time = ev_over_time / (std_path + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Find optimal holding periods
    long_hold_time = np.argmax(sharpe_over_time)
    short_hold_time = np.argmin(sharpe_over_time)
    
    stats = {"long": {}, "short": {}, "neutral": {}}
    
    # Compute statistics for each strategy
    stats["neutral"] = compute_statistics(mcs_array, context_price, print_output=False)
    stats["long"] = compute_statistics(
        mcs_array, context_price, at_index=long_hold_time, print_output=False
    )
    stats["short"] = compute_statistics(
        mcs_array, context_price, at_index=short_hold_time, print_output=False
    )
    
    return stats, mean_path, std_path, context_lob


def get_strategy_prediction(lob, 
                           context_price: List[float], 
                           events_per_step: int = 50, 
                           order_qty_mean: float = 5, 
                           order_qty_std: float = 1, 
                           maker_volatility: float = 0.01, 
                           simulations: int = 100, 
                           steps_per_sequence: int = 10,
                           significance_level: float = 0.05) -> Tuple[int, np.ndarray, np.ndarray, object]:
    """
    Generate trading signal based on Monte Carlo fair value estimation.
    
    This function uses Monte Carlo simulation to estimate the fair value
    and generates a trading signal if the expected move is statistically
    significant.
    
    Parameters
    ----------
    lob : OrderBook
        Current order book state
    context_price : list of float
        Historical price context
    events_per_step : int
        Number of order events per time step
    order_qty_mean : float
        Mean order quantity
    order_qty_std : float
        Standard deviation of order quantity
    maker_volatility : float
        Price volatility for limit orders
    simulations : int
        Number of Monte Carlo paths
    steps_per_sequence : int
        Number of time steps to simulate
    significance_level : float
        p-value threshold for statistical significance
        
    Returns
    -------
    exposure : int
        Trading signal: 1 (long), -1 (short), or 0 (neutral)
    mean_path : np.ndarray
        Expected price path
    std_path : np.ndarray
        Standard deviation of price path
    context_lob : OrderBook
        The input order book
        
    Notes
    -----
    The signal is generated only if:
    1. p-value < significance_level (default 0.05)
    2. Expected value is significantly positive (long) or negative (short)
    
    This ensures we only trade when there is statistical evidence of
    directional movement.
    """
    stats, mean_path, std_path, context_lob = get_strategy_stats(
        lob, context_price, events_per_step, order_qty_mean, 
        order_qty_std, maker_volatility, simulations, steps_per_sequence
    )
    
    exposure = 0
    
    # Generate signal only if statistically significant
    if stats["neutral"]["p_value"] <= significance_level:
        if stats["neutral"]["ev"] > 0:
            exposure = 1  # Long signal
        elif stats["neutral"]["ev"] < 0:
            exposure = -1  # Short signal
    
    return exposure, mean_path, std_path, context_lob
