"""
Machine Learning Package
AI-powered trading system with predictive analytics
"""

from .ml_system import MLDaytradingSystem

# Create models directory if it doesn't exist
import os
models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

# Create empty __init__.py in models directory
models_init = os.path.join(models_dir, '__init__.py')
if not os.path.exists(models_init):
    with open(models_init, 'w') as f:
        f.write('# Models directory for ML system\n')

__version__ = "1.0.0"
__all__ = ['MLDaytradingSystem']
