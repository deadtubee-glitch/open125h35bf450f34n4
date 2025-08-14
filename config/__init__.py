"""
Configuration Package
Handles loading and management of YAML configuration files
"""

from .config_loader import load_config, create_advanced_config

__version__ = "1.0.0"
__all__ = ['load_config', 'validate_config', 'create_advanced_config']
