"""
Analysis Package
Performance tracking and sentiment analysis modules
"""

from .performance import PerformanceAnalyzer
from .sentiment import NewsSentimentAnalyzer

__version__ = "1.0.0"
__all__ = ['PerformanceAnalyzer', 'NewsSentimentAnalyzer']
