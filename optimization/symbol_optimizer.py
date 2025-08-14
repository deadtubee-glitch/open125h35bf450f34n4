# optimization/symbol_optimizer.py
import json
import os
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import numpy as np
from optimization.threaded_optimizer import OptimizedBacktestTrader

logger = logging.getLogger(__name__)

class SymbolSpecificOptimizer:
    """Pro-Symbol Optimierung mit Profit-Focus und Feintuning"""
    
    def __init__(self, config, client):
        self.config = config
        self.client = client
        self.min_profit_threshold = 0.15  # 15% minimum profit
        self.results_file = 'optimization_results_per_symbol.json'
        self.max_threads = min(8, len(config['trading']['symbols']))
        
        # Erweiterte Parameter-Ranges für Feintuning
        self.base_ranges = config['parameter_ranges']
        self.fine_ranges = self._generate_fine_ranges()
        
        # Gespeicherte Ergebnisse laden
        self.saved_results = self._load_saved_results()
        
    def _generate_fine_ranges(self):
        """Generiert feinere Parameter-Ranges für Feintuning"""
        return {
            'rsi_period': list(range(8, 25, 1)),  # Feinere Schritte
            'rsi_oversold': list(range(20, 45, 2)),
            'rsi_overbought': list(range(60, 85, 2)), 
            'ema_fast': list(range(3, 20, 1)),
            'ema_slow': list(range(15, 40, 1)),
            'take_profit_pct': [x/1000 for x in range(15, 100, 5)],  # 1.5% bis 10%
            'stop_loss_pct': [x/1000 for x in range(8, 50, 2)],     # 0.8% bis 5%
            'trailing_sl_pct': [x/1000 for x in range(5, 30, 2)],   # 0.5% bis 3%
            'volume_threshold': [x/10 for x in range(10, 35, 2)],   # 1.0 bis 3.5
            'bb_period': list(range(12, 35, 1)),
            'confidence_threshold': [x/100 for x in range(30, 85, 5)]  # 0.3 bis 0.8
        }
    
    def _load_saved_results(self):
        """Lädt gespeicherte Optimierungs-Ergebnisse"""
        try:
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load saved results: {e}")
        return {}
    
    def _save_results(self):
        """Speichert Optimierungs-Ergebnisse"""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.saved_results, f, indent=2, default=str)
            logger.info(f"✅ Results saved to {self.results_file}")
        except Exception as e:
            logger.error(f"❌ Could not save results: {e}")
    
    def should_optimize_symbol(self, symbol):
        """Prüft ob Symbol optimiert werden sollte"""
        if symbol not in self.saved_results:
            return True, "Noch nie optimiert"
            
        last_opt = self.saved_results[symbol].get('last_optimization')
        if not last_opt:
            return True, "Keine gültige letzte Optimierung"
            
        try:
            last_time = datetime.fromisoformat(last_opt)
            age_hours = (datetime.now() - last_time).total_seconds() / 3600
            
            if age_hours >= 48:
                return True, f"Letzte Optimierung vor {age_hours:.1f}h"
            else:
                return False, f"Letzte Optimierung vor {age_hours:.1f}h (zu neu)"
                
        except Exception as e:
            return True, f"Fehler beim Prüfen der Zeit: {e}"
    
    def optimize_symbol(self, symbol, use_fine_tuning=False):
        """Optimiert einzelnes Symbol"""
        logger.info(f"🎯 Starte Optimierung für {symbol}")
        
        # Parameter-Ranges wählen
        ranges = self.fine_ranges if use_fine_tuning else self.base_ranges
        
        # Historische Daten laden
        historical_data = self._get_historical_data(symbol, days=14)  # 2 Wochen
        if historical_data is None or len(historical_data) < 1000:
            logger.warning(f"❌ Nicht genug Daten für {symbol}")
            return None
        
        logger.info(f"📊 {len(historical_data)} Kerzen für {symbol} geladen")
        
        # Parameter-Kombinationen generieren
        combinations = self._generate_combinations(ranges, symbol, use_fine_tuning)
        logger.info(f"🔧 Teste {len(combinations)} Kombinationen für {symbol}")
        
        best_result = None
        results = []
        
        # Threading für schnellere Tests
        with ThreadPoolExecutor(max_workers=min(4, len(combinations))) as executor:
            futures = {
                executor.submit(self._test_parameters, symbol, params, historical_data): params 
                for params in combinations
            }
            
            completed = 0
            for future in as_completed(futures, timeout=300):  # 5min timeout
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"⏳ {symbol}: {completed}/{len(combinations)} getestet")
                
                try:
                    result = future.result()
                    if result and result['total_profit'] >= self.min_profit_threshold:
                        results.append(result)
                        
                        if not best_result or result['score'] > best_result['score']:
                            best_result = result
                            logger.info(f"🏆 {symbol}: Neuer Bestwert - Profit: {result['total_profit']*100:.1f}%, Score: {result['score']:.4f}")
                            
                except Exception as e:
                    logger.warning(f"Test fehlgeschlagen für {symbol}: {e}")
        
        # Ergebnisse speichern
        if best_result:
            self.saved_results[symbol] = {
                'best_params': best_result['params'],
                'best_score': best_result['score'],
                'total_profit': best_result['total_profit'],
                'win_rate': best_result['win_rate'],
                'total_trades': best_result['total_trades'],
                'max_drawdown': best_result['max_drawdown'],
                'last_optimization': datetime.now().isoformat(),
                'optimization_type': 'fine_tuning' if use_fine_tuning else 'standard',
                'profit_meets_threshold': best_result['total_profit'] >= self.min_profit_threshold
            }
            
            logger.info(f"✅ {symbol} optimiert: {best_result['total_profit']*100:.1f}% Profit, {best_result['win_rate']*100:.1f}% Win Rate")
            return best_result
        else:
            logger.warning(f"❌ {symbol}: Keine Parameter mit ≥15% Profit gefunden")
            return None
    
    def _get_historical_data(self, symbol, days=14):
        """Holt historische Daten"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval='1m',
                start_str=start_time.strftime('%Y-%m-%d'),
                end_str=end_time.strftime('%Y-%m-%d')
            )
            
            if not klines:
                return None
                
            import pandas as pd
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen historischer Daten für {symbol}: {e}")
            return None
    
    def _generate_combinations(self, ranges, symbol, fine_tuning=False):
        """Generiert Parameter-Kombinationen"""
        # Für Feintuning: Fokus um bekannte gute Parameter
        if fine_tuning and symbol in self.saved_results:
            return self._generate_fine_combinations(ranges, symbol)
        
        # Standard: Sampling für Performance
        param_keys = list(ranges.keys())
        param_values = [ranges[key] for key in param_keys]
        
        # Alle Kombinationen wären zu viele, also samplen
        import random
        all_combinations = list(itertools.product(*param_values))
        
        # Intelligentes Sampling: Mehr fokussiert auf wahrscheinlich gute Bereiche
        sampled = []
        
        # 1. Zufällige Auswahl
        random.shuffle(all_combinations)
        sampled.extend(all_combinations[:200])
        
        # 2. Profit-optimierte Kombinationen (niedrige Stop-Loss, moderate Take-Profit)
        profit_focused = [
            combo for combo in all_combinations[:1000] 
            if self._is_profit_focused_combo(dict(zip(param_keys, combo)))
        ]
        sampled.extend(profit_focused[:100])
        
        # Duplikate entfernen
        unique_combinations = []
        seen = set()
        for combo in sampled:
            combo_tuple = tuple(combo)
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                # Validierung: EMA Fast < EMA Slow
                param_dict = dict(zip(param_keys, combo))
                if param_dict['ema_fast'] < param_dict['ema_slow']:
                    unique_combinations.append(param_dict)
        
        return unique_combinations[:250]  # Max 250 Tests pro Symbol
    
    def _is_profit_focused_combo(self, params):
        """Prüft ob Kombination profit-optimiert ist"""
        return (
            params['take_profit_pct'] >= 0.025 and  # Mind. 2.5% Take Profit
            params['stop_loss_pct'] <= 0.025 and   # Max 2.5% Stop Loss
            params['rsi_oversold'] <= 35 and        # Nicht zu konservativ
            params['confidence_threshold'] <= 0.6   # Nicht zu restriktiv
        )
    
    def _generate_fine_combinations(self, ranges, symbol):
        """Generiert Feintuning-Kombinationen um bekannte gute Parameter"""
        best_params = self.saved_results[symbol]['best_params']
        combinations = []
        
        # Variationen um beste Parameter
        for key, best_value in best_params.items():
            if key in ranges:
                values = ranges[key]
                
                # Finde Index des besten Wertes
                if best_value in values:
                    idx = values.index(best_value)
                    # Nehme Werte um den besten herum
                    start = max(0, idx - 3)
                    end = min(len(values), idx + 4)
                    nearby_values = values[start:end]
                else:
                    # Falls exakter Wert nicht in Range, nehme nächste
                    nearby_values = [v for v in values if abs(v - best_value) <= abs(best_value * 0.2)][:7]
                
                # Kombinationen mit Variationen dieses Parameters
                for value in nearby_values:
                    new_params = best_params.copy()
                    new_params[key] = value
                    
                    # Validierung
                    if new_params['ema_fast'] < new_params['ema_slow']:
                        combinations.append(new_params)
        
        # Zusätzlich: Komplett neue vielversprechende Kombinationen
        for _ in range(50):
            random_params = {}
            for key, values in ranges.items():
                if key in ['take_profit_pct', 'stop_loss_pct']:
                    # Profit-fokussierte Werte
                    if key == 'take_profit_pct':
                        random_params[key] = np.random.choice([v for v in values if v >= 0.02])
                    else:
                        random_params[key] = np.random.choice([v for v in values if v <= 0.03])
                else:
                    random_params[key] = np.random.choice(values)
            
            if random_params['ema_fast'] < random_params['ema_slow']:
                combinations.append(random_params)
        
        return combinations[:200]  # Max 200 für Feintuning
    
    def _test_parameters(self, symbol, params, historical_data):
        """Testet Parameter-Set für Symbol"""
        try:
            trader = OptimizedBacktestTrader(params)
            result = trader.run_backtest_fast(symbol, historical_data)
            
            if not result or result['total_trades'] < 5:
                return None
            
            # Score-Berechnung mit Profit-Focus
            profit_weight = 0.5  # 50% Gewichtung auf Profit
            win_rate_weight = 0.2
            trades_weight = 0.15
            drawdown_weight = 0.15
            
            # Bonus für hohe Profits
            profit_bonus = max(0, result['total_return'] - 0.15) * 2  # Bonus ab 15%
            
            score = (
                result['total_return'] * profit_weight +
                result['win_rate'] * win_rate_weight +
                min(result['total_trades'] / 50.0, 1.0) * trades_weight -
                result['max_drawdown'] * drawdown_weight +
                profit_bonus
            )
            
            return {
                'params': params,
                'score': score,
                'total_profit': result['total_return'],
                'win_rate': result['win_rate'],
                'total_trades': result['total_trades'],
                'max_drawdown': result['max_drawdown'],
                'sharpe_ratio': result.get('sharpe_ratio', 0)
            }
            
        except Exception as e:
            logger.warning(f"Parameter test failed: {e}")
            return None
    
    def optimize_all_symbols(self, force_optimization=False):
        """Optimiert alle Symbole"""
        results = {}
        symbols_to_optimize = []
        
        # Prüfe welche Symbole optimiert werden sollen
        for symbol in self.config['trading']['symbols']:
            should_opt, reason = self.should_optimize_symbol(symbol)
            
            if should_opt or force_optimization:
                symbols_to_optimize.append(symbol)
                logger.info(f"📋 {symbol} wird optimiert: {reason}")
            else:
                logger.info(f"⏭️ {symbol} übersprungen: {reason}")
                # Lade gespeicherte Parameter
                if symbol in self.saved_results:
                    results[symbol] = self.saved_results[symbol]
        
        if not symbols_to_optimize:
            logger.info("✅ Alle Symbole sind aktuell optimiert")
            return self.saved_results
        
        logger.info(f"🚀 Starte Optimierung für {len(symbols_to_optimize)} Symbole")
        
        # Optimiere jedes Symbol einzeln
        for i, symbol in enumerate(symbols_to_optimize, 1):
            logger.info(f"📊 Optimiere Symbol {i}/{len(symbols_to_optimize)}: {symbol}")
            
            # Normale Optimierung
            result = self.optimize_symbol(symbol, use_fine_tuning=False)
            
            if result and result['total_profit'] >= self.min_profit_threshold:
                results[symbol] = self.saved_results[symbol]
                
                # Feintuning wenn sehr gute Ergebnisse
                if result['total_profit'] >= 0.3:  # 30%+ Profit
                    logger.info(f"🎯 {symbol} zeigt {result['total_profit']*100:.1f}% Profit - Starte Feintuning")
                    fine_result = self.optimize_symbol(symbol, use_fine_tuning=True)
                    
                    if fine_result and fine_result['score'] > result['score']:
                        logger.info(f"🏆 {symbol}: Feintuning verbesserte Score von {result['score']:.4f} auf {fine_result['score']:.4f}")
                        results[symbol] = self.saved_results[symbol]
            
            # Zwischenspeichern
            self._save_results()
        
        logger.info(f"✅ Symbol-Optimierung abgeschlossen für {len(symbols_to_optimize)} Symbole")
        return results
    
    def get_best_params_for_symbol(self, symbol):
        """Gibt beste Parameter für Symbol zurück"""
        if symbol in self.saved_results:
            return self.saved_results[symbol]['best_params']
        return None
    
    def print_optimization_summary(self):
        """Druckt Zusammenfassung aller Optimierungen"""
        if not self.saved_results:
            logger.info("Keine Optimierungs-Ergebnisse vorhanden")
            return
        
        print("\n" + "="*80)
        print("📊 SYMBOL-SPEZIFISCHE OPTIMIERUNGS-ZUSAMMENFASSUNG")
        print("="*80)
        
        total_profit = 0
        profitable_symbols = 0
        
        for symbol, data in self.saved_results.items():
            profit = data['total_profit'] * 100
            win_rate = data['win_rate'] * 100
            trades = data['total_trades']
            drawdown = data['max_drawdown'] * 100
            last_opt = data.get('last_optimization', 'Unknown')
            
            status = "✅" if profit >= 15 else "❌"
            total_profit += data['total_profit']
            
            if profit >= 15:
                profitable_symbols += 1
            
            print(f"{status} {symbol:>10}: {profit:>6.1f}% Profit | {win_rate:>5.1f}% Win | {trades:>3} Trades | {drawdown:>5.1f}% DD")
            
            if isinstance(last_opt, str) and 'T' in last_opt:
                try:
                    opt_time = datetime.fromisoformat(last_opt.replace('Z', ''))
                    age = datetime.now() - opt_time
                    print(f"              Letzte Optimierung: vor {age.total_seconds()/3600:.1f}h")
                except:
                    pass
        
        avg_profit = total_profit / len(self.saved_results) * 100
        success_rate = profitable_symbols / len(self.saved_results) * 100
        
        print("-"*80)
        print(f"📈 Durchschnittlicher Profit: {avg_profit:.1f}%")
        print(f"🎯 Erfolgreiche Symbole (≥15%): {profitable_symbols}/{len(self.saved_results)} ({success_rate:.1f}%)")
        print(f"🏆 Beste Performance: {max(self.saved_results.values(), key=lambda x: x['total_profit'])['total_profit']*100:.1f}%")
        print("="*80 + "\n")
