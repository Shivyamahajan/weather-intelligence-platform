"""
Logging Utility
Project: Weather Intelligence Platform
Author: Shivya
Date: July 2026

What is logging and why do you need it?
    Right now when something goes wrong in your code, you only
    see the error when you are watching. In a production system
    running 24/7, you need a permanent record of:
    - Every prediction made
    - Any errors that occurred
    - Performance metrics over time
    
    This is called logging. All serious production systems have it.
"""

import logging
import os
from datetime import datetime

def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create a logger that writes to both console and a file.
    
    Level hierarchy:
    DEBUG < INFO < WARNING < ERROR < CRITICAL
    
    In production you typically log INFO and above.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Format: timestamp | level | message
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File handler — new file each day
    log_filename = os.path.join(
        log_dir,
        f"weather_platform_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized: {name}")
    return logger


# Global application logger
app_logger = setup_logger("WeatherPlatform")