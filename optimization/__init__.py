"""
Optimization Package
Parameter optimization and backtesting modules
"""

from .optimizer import ParameterOptimizer
from .backtest import BacktestTrader, OptimizedBacktestTrader
from .threaded_optimizer import ThreadedParameterOptimizer, run_parameter_optimization_threaded

__version__ = "1.0.0"
__all__ = [
    'ParameterOptimizer', 
    'ThreadedParameterOptimizer', 
    'run_parameter_optimization_threaded',
    'BacktestTrader', 
    'OptimizedBacktestTrader'
]
