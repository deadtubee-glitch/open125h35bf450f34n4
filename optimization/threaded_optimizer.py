import threading
import time
import logging
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Local imports
from .optimizer import ParameterOptimizer
from .backtest import BacktestTrader, OptimizedBacktestTrader

# Globals (werden in anderen Modulen definiert)
optimization_status = {'running': False, 'progress': 0}
test_results = {}

class ThreadedParameterOptimizer(ParameterOptimizer):
    """Threading-optimierte Version der Parameter-Optimierung"""
    
    def __init__(self, config, client):
        super().__init__(config, client)
        self.max_workers = min(8, len(config['trading']['symbols']))  # Max 8 Threads
        self.thread_pool = None
        
    def optimize_parameters_threaded(self):
        """Multi-threaded Parameter-Optimierung für deutlich höhere Geschwindigkeit"""
        logging.info("Starte MULTI-THREADED Parameter-Optimierung...")
        optimization_status['running'] = True
        optimization_status['progress'] = 0
        
        parameter_sets = self.generate_parameter_combinations()
        results = []
        completed_tests = 0
        
        logging.info(f"Teste {len(parameter_sets)} Parameter-Kombinationen mit {self.max_workers} Threads...")
        
        # Historical Data vorher für alle Symbole laden (einmalig)
        logging.info("Lade historische Daten für alle Symbole...")
        historical_data_cache = {}
        for symbol in self.config['trading']['symbols']:
            historical_data_cache[symbol] = self.get_historical_data(
                symbol, self.optimization_config['historical_data_days']
            )
            if historical_data_cache[symbol] is None:
                logging.warning(f"Keine Daten für {symbol} - wird übersprungen")
        
        # Threading für Parameter-Tests
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Alle Jobs parallel submiten
            future_to_params = {
                executor.submit(self.evaluate_parameter_set_threaded, params, historical_data_cache): params 
                for params in parameter_sets
            }
            
            # Ergebnisse sammeln wenn sie fertig sind
            for future in as_completed(future_to_params):
                if not optimization_status['running']:
                    break
                    
                params = future_to_params[future]
                completed_tests += 1
                
                try:
                    result = future.result(timeout=60)  # 60s Timeout pro Test
                    if result and result['total_trades'] >= self.optimization_config['min_trades_for_validity']:
                        results.append(result)
                        logging.info(f"✅ Test {completed_tests}/{len(parameter_sets)}: Score {result['score']:.4f}, "
                                   f"Profit: {result['avg_profit']:.2%}, Win Rate: {result['avg_win_rate']:.2%}")
                    else:
                        logging.info(f"❌ Test {completed_tests}/{len(parameter_sets)}: Ungültig (zu wenig Trades)")
                        
                except Exception as e:
                    logging.error(f"Fehler bei Parameter-Test: {e}")
                
                optimization_status['progress'] = int(completed_tests / len(parameter_sets) * 100)
        
        # Beste Parameter finden
        if results:
            best_result = max(results, key=lambda x: x['score'])
            self.best_params = best_result['params']
            optimization_status['best_params'] = best_result
            
            logging.info("🎉 MULTI-THREADED Optimierung abgeschlossen!")
            logging.info(f"Beste Parameter: {self.best_params}")
            logging.info(f"Bester Score: {best_result['score']:.4f}")
            logging.info(f"Getestete Kombinationen: {completed_tests}")
            logging.info(f"Gültige Ergebnisse: {len(results)}")
            
            # Top 5 Ergebnisse zeigen
            top_results = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
            logging.info("🏆 TOP 5 Parameter-Sets:")
            for i, result in enumerate(top_results, 1):
                logging.info(f"  #{i}: Score {result['score']:.4f}, "
                           f"Profit: {result['avg_profit']:.2%}, "
                           f"Win Rate: {result['avg_win_rate']:.2%}, "
                           f"Trades: {result['total_trades']}")
            
            # Ergebnisse speichern
            test_results.update({
                'optimization_completed': True,
                'optimization_method': 'MULTI-THREADED',
                'threads_used': self.max_workers,
                'total_tests': completed_tests,
                'valid_results': len(results),
                'best_params': self.best_params,
                'best_score': best_result['score'],
                'top_results': top_results[:10]  # Top 10 speichern
            })
        else:
            logging.warning("Keine gültigen Parameter-Sets gefunden!")
            
        optimization_status['running'] = False
        return self.best_params
    
    def evaluate_parameter_set_threaded(self, params, historical_data_cache):
        """Thread-sichere Evaluierung eines Parameter-Sets"""
        try:
            total_profit = 0
            total_trades = 0
            win_rate_sum = 0
            max_drawdown = 0
            sharpe_ratio_sum = 0
            valid_symbols = 0
            
            symbol_results = {}
            
            for symbol in self.config['trading']['symbols']:
                historical_data = historical_data_cache.get(symbol)
                if historical_data is None or len(historical_data) < 1000:
                    continue
                    
                # Backtest für dieses Symbol
                result = self.backtest_strategy_optimized(symbol, params, historical_data)
                if result is None:
                    continue
                    
                symbol_results[symbol] = result
                total_profit += result['total_return']
                total_trades += result['total_trades']
                win_rate_sum += result['win_rate']
                max_drawdown = max(max_drawdown, result['max_drawdown'])
                sharpe_ratio_sum += result['sharpe_ratio']
                valid_symbols += 1
            
            if valid_symbols == 0:
                return None
                
            # Durchschnitte berechnen
            avg_profit = total_profit / valid_symbols
            avg_win_rate = win_rate_sum / valid_symbols
            avg_sharpe = sharpe_ratio_sum / valid_symbols
            
            # Verbesserter Score (mehr Gewichtung auf Trades und Konsistenz)
            consistency_bonus = 1.0 if valid_symbols == len(self.config['trading']['symbols']) else 0.8
            trades_factor = min(total_trades / 50.0, 2.0)  # Bonus für mehr Trades (bis 2x)
            
            score = (avg_profit * 0.35 + 
                    avg_win_rate * 0.25 + 
                    avg_sharpe * 0.15 - 
                    max_drawdown * 0.1 +
                    consistency_bonus * 0.1) * trades_factor
            
            return {
                'params': params,
                'score': score,
                'avg_profit': avg_profit,
                'total_trades': total_trades,
                'avg_win_rate': avg_win_rate,
                'max_drawdown': max_drawdown,
                'avg_sharpe': avg_sharpe,
                'valid_symbols': valid_symbols,
                'consistency_bonus': consistency_bonus,
                'symbol_results': symbol_results
            }
            
        except Exception as e:
            logging.error(f"Fehler bei threaded Parameter-Evaluierung: {e}")
            return None
    
    def backtest_strategy_optimized(self, symbol, params, historical_data):
        """Optimierte Backtest-Strategie (weniger Berechnungen)"""
        try:
            # Leichtgewichtige Backtest-Version
            backtest_trader = OptimizedBacktestTrader(params)
            results = backtest_trader.run_backtest_fast(symbol, historical_data)
            return results
            
        except Exception as e:
            logging.error(f"Optimierter Backtest Fehler für {symbol}: {e}")
            return None

def run_parameter_optimization_threaded(config, client):
    """
    Hauptfunktion für Multi-threaded Parameter-Optimierung
    """
    logger = logging.getLogger(__name__)
    logger.info("🧵 Starte MULTI-THREADED Parameter-Optimierung...")
    
    try:
        # Import hier um circular imports zu vermeiden
        from utils.flask_app import update_optimization_status, update_test_results
        
        # Optimizer erstellen
        optimizer = ThreadedParameterOptimizer(config, client)
        
        # Update status
        update_optimization_status({
            'running': True, 
            'progress': 0, 
            'method': 'MULTI-THREADED',
            'threads': optimizer.max_workers
        })
        
        # Optimierung starten
        best_params = optimizer.optimize_parameters_threaded()
        
        if best_params:
            logger.info(f"✅ Optimierung abgeschlossen! Beste Parameter gefunden.")
            logger.info(f"🏆 Score: {best_params.get('score', 0):.4f}")
            
            # Update final status
            update_optimization_status({
                'running': False,
                'completed': True,
                'best_params': best_params
            })
            
            return best_params
        else:
            logger.warning("⚠️ Keine optimalen Parameter gefunden - verwende Standard-Parameter")
            return _get_default_parameters(config)
            
    except Exception as e:
        logger.error(f"❌ Fehler bei Parameter-Optimierung: {e}")
        logger.info("🔄 Verwende Standard-Parameter...")
        
        # Update error status
        try:
            update_optimization_status({'running': False, 'error': str(e)})
        except:
            pass
            
        return _get_default_parameters(config)

def _get_default_parameters(config):
    """
    Standard-Parameter wenn Optimierung fehlschlägt
    """
    parameter_ranges = config.get('parameter_ranges', {})
    
    # Mittlere Werte aus Ranges nehmen
    default_params = {}
    for key, values in parameter_ranges.items():
        if isinstance(values, list) and values:
            # Mittleren Wert nehmen
            middle_idx = len(values) // 2
            default_params[key] = values[middle_idx]
        else:
            # Fallback defaults
            fallback_defaults = {
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'ema_fast': 12,
                'ema_slow': 26,
                'take_profit_pct': 0.015,
                'stop_loss_pct': 0.02,
                'trailing_sl_pct': 0.006,
                'volume_threshold': 2.0,
                'bb_period': 20,
                'confidence_threshold': 0.7
            }
            default_params[key] = fallback_defaults.get(key, 0.5)
    
    return {
        'parameters': default_params,
        'score': 0.5,
        'method': 'default',
        'total_tests': 0,
        'best_win_rate': 0.0,
        'best_profit': 0.0,
        'optimization_time': 0.0
    }
