"""
Utilities Package
Flask web interface and logging setup
"""

from .flask_app import (
    start_flask_server, 
    update_live_status, 
    update_test_results, 
    update_optimization_status,
    live_status,
    test_results,
    optimization_status
)
from .logging_setup import setup_logging, get_logger

__version__ = "1.0.0"
__all__ = [
    'start_flask_server',
    'update_live_status',
    'update_test_results', 
    'update_optimization_status',
    'live_status',
    'test_results',
    'optimization_status',
    'setup_logging',
    'get_logger'
]
