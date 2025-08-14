"""
Logging Setup Module
Centralized logging configuration for the trading bot
"""

import logging
import os
from pathlib import Path
from datetime import datetime

def setup_logging():
    """Setup logging configuration for the entire application"""
    
    # Create logs directory
    log_dir = Path(__file__).parent.parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"trading_bot_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Set specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    logging.info("✅ Logging system initialized")
    logging.info(f"📁 Log file: {log_file}")

def get_logger(name):
    """Get a logger with the specified name"""
    return logging.getLogger(name)