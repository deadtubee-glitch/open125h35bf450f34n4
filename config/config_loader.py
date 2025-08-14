import yaml
import logging
from pathlib import Path

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "config.yaml"
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logging.info("✅ Configuration loaded successfully")
        return config
        
    except FileNotFoundError:
        logging.error(f"❌ Config file not found: {config_path}")
        create_advanced_config()      # erzeugt Default config.yaml
        return None

    except Exception as e:
        logging.error(f"❌ Error loading config: {e}")
        return None

def create_advanced_config():
    """Creates advanced configuration with optimization parameters"""
    example_config = {
        'api': {
            'key': 'VpPEJQYLzMShqXN8BghHViuXQSfgkhD2X0gfXlyOtVGbJZQIgvL8wwuQlGz82j3D',
            'secret': 'fjuyYBfRl1mhnMVh13n4BhVXOKak7gKmk2DFWq0JQsgleMPodL9CeKpl9qubFKFv',
            'testnet': True
        },
        'trading': {
            'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT'],
            'interval': '1m',
            'max_risk_per_trade': 0.15,
            'min_usdt_balance': 20.0,
            'max_concurrent_positions': 3,
            'capital_protection_threshold': 0.8,
            'emergency_stop_loss': 0.2,
            'daily_loss_limit': 0.05,
            'cooldown_period': 300,
        },
        'optimization': {
            'enabled': True,
            'test_duration_hours': 24,
            'min_trades_for_validity': 10,
            'optimization_runs': 50,
            'historical_data_days': 7,
        },
        'parameter_ranges': {
            'rsi_period': [10, 14, 21, 28],
            'rsi_oversold': [25, 30, 35, 40],
            'rsi_overbought': [60, 65, 70, 75],
            'ema_fast': [5, 8, 12, 16],
            'ema_slow': [16, 21, 26, 34],
            'take_profit_pct': [0.02, 0.03, 0.05, 0.08],
            'stop_loss_pct': [0.015, 0.02, 0.03, 0.04],
            'trailing_sl_pct': [0.01, 0.015, 0.02, 0.025],
            'volume_threshold': [1.2, 1.5, 2.0, 2.5],
            'bb_period': [16, 20, 24, 28],
            'confidence_threshold': [0.5, 0.6, 0.7, 0.8]
        },
        'risk_management': {
            'max_drawdown': 0.15,
            'position_correlation_limit': 0.7,
            'adaptive_risk': True,
            'capital_preservation_mode': True
        },
        'advanced': {
            'use_machine_learning': True,
            'news_sentiment': True,
        }
    }
    
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "w", encoding='utf-8') as f:
        yaml.dump(example_config, f, default_flow_style=False)
    
    print("📝 Erweiterte config.yaml mit Optimierungs-Parametern erstellt!")

