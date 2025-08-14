"""
Parameter Optimizer
==================
Base parameter optimization system
"""

import itertools
import logging
import time
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class ParameterOptimizer:
    """Klasse für die Optimierung von Trading-Parametern durch Backtesting"""
    
    def __init__(self, config, client):
        self.config = config
        self.client = client
        self.parameter_ranges = config['parameter_ranges']
        self.optimization_config = config['optimization']
        self.best_params = None
        self.test_results = []
        
    def generate_parameter_combinations(self):
        """Generiert alle Parameterkombinationen für Tests"""
        param_keys = list(self.parameter_ranges.keys())
        param_values = [self.parameter_ranges[key] for key in param_keys]
        
        combinations = list(itertools.product(*param_values))
        
        # Begrenzte Anzahl für praktikable Tests
        max_combinations = self.optimization_config['optimization_runs']
        if len(combinations) > max_combinations:
            # Zufällige Auswahl
            import random
            combinations = random.sample(combinations, max_combinations)
        
        parameter_sets = []
        for combo in combinations:
            param_set = dict(zip(param_keys, combo))
            # Validierung: ema_fast < ema_slow
            if param_set['ema_fast'] < param_set['ema_slow']:
                parameter_sets.append(param_set)
        
        return parameter_sets

    def get_historical_data(self, symbol, days=7):
        """Holt historische Daten für Backtesting"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval='1m',
                start_str=start_time.strftime('%Y-%m-%d'),
                end_str=end_time.strftime('%Y-%m-%d')
            )
            
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = df[col].astype(float)
            
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            
            return df
            
        except Exception as e:
            logging.error(f"Fehler beim Abrufen historischer Daten für {symbol}: {e}")
            return None

    def backtest_strategy(self, symbol, params, historical_data):
        """Führt Backtest für spezifische Parameter durch"""
        try:
            backtest_trader = BacktestTrader(params)
            results = backtest_trader.run_backtest(symbol, historical_data)
            return results
            
        except Exception as e:
            logging.error(f"Backtest Fehler für {symbol}: {e}")
            return None

    def evaluate_parameter_set(self, params):
        """Evaluiert ein Parameter-Set über alle Symbole"""
        total_profit = 0
        total_trades = 0
        win_rate = 0
        max_drawdown = 0
        sharpe_ratio = 0
        
        symbol_results = {}
        
        for symbol in self.config['trading']['symbols']:
            historical_data = self.get_historical_data(symbol, 
                                                    self.optimization_config['historical_data_days'])
            if historical_data is None or len(historical_data) < 1000:
                continue
                
            result = self.backtest_strategy(symbol, params, historical_data)
            if result is None:
                continue
                
            symbol_results[symbol] = result
            total_profit += result['total_return']
            total_trades += result['total_trades']
            win_rate += result['win_rate']
            max_drawdown = max(max_drawdown, result['max_drawdown'])
            sharpe_ratio += result['sharpe_ratio']
        
        if len(symbol_results) == 0:
            return None
            
        # Durchschnitte berechnen
        avg_profit = total_profit / len(symbol_results)
        avg_win_rate = win_rate / len(symbol_results)
        avg_sharpe = sharpe_ratio / len(symbol_results)
        
        # Score-Berechnung (gewichtete Kombination verschiedener Metriken)
        score = (avg_profit * 0.4 + 
                avg_win_rate * 0.3 + 
                avg_sharpe * 0.2 - 
                max_drawdown * 0.1)
        
        return {
            'params': params,
            'score': score,
            'avg_profit': avg_profit,
            'total_trades': total_trades,
            'avg_win_rate': avg_win_rate,
            'max_drawdown': max_drawdown,
            'avg_sharpe': avg_sharpe,
            'symbol_results': symbol_results
        }

    def optimize_parameters(self):
        """Hauptfunktion für Parameter-Optimierung"""
        logging.info("Starte Parameter-Optimierung...")
        optimization_status['running'] = True
        optimization_status['progress'] = 0
        
        parameter_sets = self.generate_parameter_combinations()
        results = []
        
        logging.info(f"Teste {len(parameter_sets)} Parameter-Kombinationen...")
        
        for i, params in enumerate(parameter_sets):
            if not optimization_status['running']:
                break
                
            logging.info(f"Teste Parameter-Set {i+1}/{len(parameter_sets)}: {params}")
            
            result = self.evaluate_parameter_set(params)
            if result and result['total_trades'] >= self.optimization_config['min_trades_for_validity']:
                results.append(result)
                logging.info(f"Score: {result['score']:.4f}, Profit: {result['avg_profit']:.2%}, Win Rate: {result['avg_win_rate']:.2%}")
            
            optimization_status['progress'] = int((i + 1) / len(parameter_sets) * 100)
        
        # Beste Parameter finden
        if results:
            best_result = max(results, key=lambda x: x['score'])
            self.best_params = best_result['params']
            optimization_status['best_params'] = best_result
            
            logging.info("Optimierung abgeschlossen!")
            logging.info(f"Beste Parameter: {self.best_params}")
            logging.info(f"Bester Score: {best_result['score']:.4f}")
            
            # Ergebnisse speichern
            test_results.update({
                'optimization_completed': True,
                'best_params': self.best_params,
                'best_score': best_result['score'],
                'all_results': results[:10]  # Top 10 Ergebnisse
            })
        else:
            logging.warning("Keine gültigen Parameter-Sets gefunden!")
            
        optimization_status['running'] = False
        return self.best_params
