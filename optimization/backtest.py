import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend     import EMAIndicator
from ta.volatility import BollingerBands



class BacktestTrader:
    """Vereinfachte Trading-Klasse für Backtesting"""
    
    def __init__(self, params):
        self.params = params
        self.positions = {}
        self.trades = []
        self.balance = 10000  # Start mit 10k USDT für Backtest
        self.initial_balance = self.balance
        self.max_balance = self.balance
        
    def calculate_indicators(self, df, lookback=50):
        """Berechnet Indikatoren für Backtest"""
        indicators = {}
        
        # RSI
        rsi = RSIIndicator(df['close'], window=self.params['rsi_period'])
        indicators['rsi'] = rsi.rsi()
        
        # EMAs
        ema_fast = EMAIndicator(df['close'], window=self.params['ema_fast'])
        ema_slow = EMAIndicator(df['close'], window=self.params['ema_slow'])
        indicators['ema_fast'] = ema_fast.ema_indicator()
        indicators['ema_slow'] = ema_slow.ema_indicator()
        
        # Bollinger Bands
        bb = BollingerBands(df['close'], window=self.params['bb_period'])
        indicators['bb_upper'] = bb.bollinger_hband()
        indicators['bb_lower'] = bb.bollinger_lband()
        
        # Volume
        volume_sma = df['volume'].rolling(20).mean()
        indicators['volume_ratio'] = df['volume'] / volume_sma
        
        return indicators
    
    def should_buy(self, df, indicators, i):
        """Kaufsignal-Logik für Backtest - KORRIGIERT: df Parameter hinzugefügt"""
        if i < max(self.params['rsi_period'], self.params['ema_slow']):
            return False
            
        signals = 0
        
        # RSI oversold
        if indicators['rsi'].iloc[i] < self.params['rsi_oversold']:
            signals += 1
            
        # EMA crossover
        if (indicators['ema_fast'].iloc[i] > indicators['ema_slow'].iloc[i] and
            indicators['ema_fast'].iloc[i-1] <= indicators['ema_slow'].iloc[i-1]):
            signals += 1
            
        # Bollinger Band touch
        if df['close'].iloc[i] < indicators['bb_lower'].iloc[i]:
            signals += 1
            
        # Volume confirmation
        if indicators['volume_ratio'].iloc[i] > self.params['volume_threshold']:
            signals += 1
            
        return signals >= 2
    
    def should_sell(self, indicators, i, entry_price, current_price):
        """Verkaufssignal-Logik für Backtest"""
        profit_pct = (current_price - entry_price) / entry_price
        
        # Take Profit
        if profit_pct >= self.params['take_profit_pct']:
            return True, "Take Profit"
            
        # Stop Loss
        if profit_pct <= -self.params['stop_loss_pct']:
            return True, "Stop Loss"
            
        # RSI overbought
        if indicators['rsi'].iloc[i] > self.params['rsi_overbought']:
            return True, "RSI Overbought"
            
        return False, ""
    
    def run_backtest(self, symbol, df):
        """Führt kompletten Backtest durch"""
        indicators = self.calculate_indicators(df)
        
        in_position = False
        entry_price = 0
        entry_time = None
        position_size = 0
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            current_time = df['timestamp'].iloc[i]
            
            if not in_position:
                if self.should_buy(df, indicators, i):  # KORRIGIERT: df Parameter hinzugefügt
                    # Kaufen
                    position_size = self.balance * 0.1  # 10% pro Trade
                    quantity = position_size / current_price
                    
                    entry_price = current_price
                    entry_time = current_time
                    in_position = True
                    
            else:
                should_exit, exit_reason = self.should_sell(indicators, i, entry_price, current_price)
                
                if should_exit:
                    # Verkaufen
                    profit_pct = (current_price - entry_price) / entry_price
                    trade_profit = position_size * profit_pct
                    
                    self.balance += trade_profit
                    self.max_balance = max(self.max_balance, self.balance)
                    
                    # Trade aufzeichnen
                    trade = {
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'profit_pct': profit_pct,
                        'profit_usdt': trade_profit,
                        'exit_reason': exit_reason
                    }
                    self.trades.append(trade)
                    
                    in_position = False
        
        # Ergebnisse berechnen
        return self.calculate_backtest_results()
    
    def calculate_backtest_results(self):
        """Berechnet Backtest-Ergebnisse"""
        if not self.trades:
            return {
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        winning_trades = len([t for t in self.trades if t['profit_pct'] > 0])
        win_rate = winning_trades / len(self.trades)
        
        # Drawdown berechnen
        running_balance = self.initial_balance
        max_balance = self.initial_balance
        max_drawdown = 0
        
        for trade in self.trades:
            running_balance += trade['profit_usdt']
            max_balance = max(max_balance, running_balance)
            drawdown = (max_balance - running_balance) / max_balance
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe Ratio (vereinfacht)
        returns = [t['profit_pct'] for t in self.trades]
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_return': total_return,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': self.balance
        }
    
class OptimizedBacktestTrader(BacktestTrader):
    """Optimierte Version des Backtest-Traders für mehr Geschwindigkeit"""
    
    def run_backtest_fast(self, symbol, df):
        """Schnellere Backtest-Version mit weniger Features"""
        try:
            # Nur die wichtigsten Indikatoren berechnen
            indicators = self.calculate_minimal_indicators(df)
            
            in_position = False
            entry_price = 0
            entry_time = None
            position_size = 0
            
            # Vectorized operations wo möglich
            df['returns'] = df['close'].pct_change()
            df['volume_ma'] = df['volume'].rolling(20).mean()
            
            for i in range(max(50, self.params['ema_slow']), len(df)):  # Start nach Warmup
                current_price = df['close'].iloc[i]
                current_time = df['timestamp'].iloc[i] if 'timestamp' in df.columns else i
                
                if not in_position:
                    if self.should_buy_fast(indicators, i, df):
                        position_size = self.balance * 0.1
                        entry_price = current_price
                        entry_time = current_time
                        in_position = True
                        
                else:
                    should_exit, exit_reason = self.should_sell_fast(indicators, i, entry_price, current_price)
                    
                    if should_exit:
                        profit_pct = (current_price - entry_price) / entry_price
                        trade_profit = position_size * profit_pct
                        
                        self.balance += trade_profit
                        self.max_balance = max(self.max_balance, self.balance)
                        
                        # Minimale Trade-Info speichern
                        trade = {
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'profit_pct': profit_pct,
                            'profit_usdt': trade_profit,
                            'exit_reason': exit_reason
                        }
                        self.trades.append(trade)
                        
                        in_position = False
            
            return self.calculate_backtest_results()
            
        except Exception as e:
            logging.error(f"Fehler im schnellen Backtest: {e}")
            return None
    
    def calculate_minimal_indicators(self, df):
        """Berechnet nur die wichtigsten Indikatoren für Geschwindigkeit"""
        indicators = {}
        
        try:
            # RSI (vectorized)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=self.params['rsi_period']).mean()
            avg_loss = loss.rolling(window=self.params['rsi_period']).mean()
            
            rs = avg_gain / avg_loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # EMAs (vectorized)
            indicators['ema_fast'] = df['close'].ewm(span=self.params['ema_fast']).mean()
            indicators['ema_slow'] = df['close'].ewm(span=self.params['ema_slow']).mean()
            
            # Volume Ratio
            volume_ma = df['volume'].rolling(20).mean()
            indicators['volume_ratio'] = df['volume'] / volume_ma
            
            return indicators
            
        except Exception as e:
            logging.error(f"Fehler bei minimaler Indikator-Berechnung: {e}")
            return {}
    
    def should_buy_fast(self, indicators, i, df):
        """Schnelle Kaufsignal-Prüfung"""
        try:
            if i < 1:
                return False
                
            signals = 0
            
            # RSI oversold
            if not pd.isna(indicators['rsi'].iloc[i]) and indicators['rsi'].iloc[i] < self.params['rsi_oversold']:
                signals += 1
                
            # EMA crossover
            if (not pd.isna(indicators['ema_fast'].iloc[i]) and not pd.isna(indicators['ema_slow'].iloc[i]) and
                indicators['ema_fast'].iloc[i] > indicators['ema_slow'].iloc[i] and
                indicators['ema_fast'].iloc[i-1] <= indicators['ema_slow'].iloc[i-1]):
                signals += 1
                
            # Volume confirmation
            if (not pd.isna(indicators['volume_ratio'].iloc[i]) and 
                indicators['volume_ratio'].iloc[i] > self.params['volume_threshold']):
                signals += 1
                
            return signals >= 2
            
        except Exception as e:
            return False
    
    def should_sell_fast(self, indicators, i, entry_price, current_price):
        """Schnelle Verkaufssignal-Prüfung"""
        try:
            profit_pct = (current_price - entry_price) / entry_price
            
            # Take Profit
            if profit_pct >= self.params['take_profit_pct']:
                return True, "Take Profit"
                
            # Stop Loss
            if profit_pct <= -self.params['stop_loss_pct']:
                return True, "Stop Loss"
                
            # RSI overbought
            if (not pd.isna(indicators['rsi'].iloc[i]) and 
                indicators['rsi'].iloc[i] > self.params['rsi_overbought']):
                return True, "RSI Overbought"
                
            return False, ""
            
        except Exception as e:
            return False, ""

        

