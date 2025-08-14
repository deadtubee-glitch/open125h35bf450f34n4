#!/usr/bin/env python3
"""
Advanced Binance Trading Bot - Main Entry Point with Symbol-Specific Optimization
"""

import sys
import time
import logging
from pathlib import Path
import json
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.config_loader import load_config
from core.enhanced_trader import EnhancedAdvancedBinanceTrader
from optimization.threaded_optimizer import run_parameter_optimization_threaded
from utils.logging_setup import setup_logging
from utils.flask_app import optimization_status, update_optimization_status, update_balance
from binance.client import Client

def main():
    """Enhanced Main-Funktion mit Symbol-spezifischer Optimierung"""
    print("🚀 Starting Enhanced Trading Bot with Symbol-Specific Optimization...")
    
    # Logging initialisieren
    setup_logging()
    
    # Config laden
    config = load_config()
    if config is None:
        print("❌ Bitte config.yaml konfigurieren und erneut starten!")
        return
    
    # Sicherheitsabfrage für Live Trading
    if not config['api'].get('testnet', True):
        print("⚠️  WARNUNG: LIVE TRADING AKTIVIERT!")
        confirm = input("Fortfahren mit echtem Geld? (ja/nein): ")
        if confirm.lower() not in ['ja', 'yes', 'y']:
            print("❌ Abgebrochen - Sicherheit geht vor!")
            return
    
    # Client erstellen
    try:
        client = Client(config['api']['key'], config['api']['secret'], 
                       testnet=config['api'].get('testnet', True))
        
        # Verbindungstest
        client.get_account()
        print("✅ Binance API Verbindung erfolgreich")
        
    except Exception as e:
        print(f"❌ Binance API Fehler: {e}")
        return

    def test_symbols():
        """Teste verfügbare Symbole"""
        test_symbols = config['trading']['symbols']
        available_symbols = []
        
        for symbol in test_symbols:
            try:
                ticker = client.get_symbol_ticker(symbol=symbol)
                print(f"✅ {symbol}: {ticker['price']} USDT - VERFÜGBAR")
                available_symbols.append(symbol)
            except Exception as e:
                print(f"❌ {symbol}: NICHT VERFÜGBAR - {e}")
        
        return available_symbols

    # Symbole testen
    print("\n🔍 Teste verfügbare Symbole...")
    available_symbols = test_symbols()
    
    if not available_symbols:
        print("❌ Keine verfügbaren Symbole gefunden!")
        return

    # Symbol-spezifische Optimierung
    optimized_params = {}
    use_symbol_specific = True
    
    if config['optimization']['enabled']:
        print("\n🎯 SYMBOL-SPEZIFISCHE OPTIMIERUNG verfügbar!")
        print("📊 Jedes Symbol wird einzeln optimiert für maximalen Profit (≥15%)")
        
        # Importiere Symbol-Optimizer nur wenn benötigt
        try:
            from optimization.symbol_optimizer import SymbolSpecificOptimizer
            symbol_optimizer = SymbolSpecificOptimizer(config, client)
            
            # Prüfe Optimierungs-Status für jedes Symbol
            needs_optimization = False
            optimization_status_info = []
            
            for symbol in config['trading']['symbols']:
                should_opt, reason = symbol_optimizer.should_optimize_symbol(symbol)
                optimization_status_info.append((symbol, should_opt, reason))
                if should_opt:
                    needs_optimization = True
            
            # Zeige Status aller Symbole
            print("\n📋 Optimierungs-Status der Symbole:")
            for symbol, should_opt, reason in optimization_status_info:
                status = "🔄 BENÖTIGT" if should_opt else "✅ AKTUELL"
                print(f"   {status} {symbol}: {reason}")
            
            if needs_optimization:
                print(f"\n🤔 {sum(1 for _, should_opt, _ in optimization_status_info if should_opt)} Symbole benötigen Optimierung")
                
                optimization_choice = input("\nOptimierung starten?\n  [1] Symbol-spezifische Optimierung (Empfohlen)\n  [2] Standard Multi-Threading Optimierung\n  [3] Überspringen\nWahl (1/2/3): ").strip()
                
                if optimization_choice == "1":
                    print("\n🎯 Starte SYMBOL-SPEZIFISCHE Optimierung...")
                    print("⚡ Mit Multi-Threading und Profit-Focus (≥15%)")
                    print("🔬 Automatisches Feintuning bei >30% Profit")
                    
                    start_time = time.time()
                    
                    try:
                        results = symbol_optimizer.optimize_all_symbols()
                        optimization_time = time.time() - start_time
                        
                        print(f"✅ Symbol-Optimierung abgeschlossen in {optimization_time/60:.1f} Minuten!")
                        symbol_optimizer.print_optimization_summary()
                        
                        # Extrahiere optimierte Parameter für jedes Symbol
                        successful_optimizations = 0
                        for symbol in config['trading']['symbols']:
                            best_params = symbol_optimizer.get_best_params_for_symbol(symbol)
                            if best_params:
                                optimized_params[symbol] = best_params
                                successful_optimizations += 1
                                
                                # Zeige wichtige Parameter
                                tp = best_params.get('take_profit_pct', 0) * 100
                                sl = best_params.get('stop_loss_pct', 0) * 100
                                rsi_os = best_params.get('rsi_oversold', 0)
                                print(f"🎯 {symbol}: TP={tp:.1f}%, SL={sl:.1f}%, RSI<{rsi_os}")
                        
                        print(f"✅ {successful_optimizations}/{len(config['trading']['symbols'])} Symbole erfolgreich optimiert")
                        
                        # Dashboard Update für symbol-spezifische Optimierung
                        if results:
                            avg_profit = sum(r['total_profit'] for r in results.values() if 'total_profit' in r) / len(results) if results else 0
                            successful_symbols = sum(1 for r in results.values() if r.get('profit_meets_threshold', False))
                            
                            update_optimization_status({
                                'completed': True,
                                'optimization_type': 'symbol_specific',
                                'symbols_optimized': len(optimized_params),
                                'total_symbols': len(config['trading']['symbols']),
                                'avg_profit': avg_profit * 100,
                                'successful_symbols': successful_symbols,
                                'profit_threshold': 25.0,
                                'optimization_time_minutes': optimization_time / 60,
                                'ml_enabled': config.get('advanced', {}).get('use_machine_learning', False)
                            })
                        
                    except Exception as e:
                        logging.error(f"Symbol-spezifische Optimierung fehlgeschlagen: {e}")
                        print(f"❌ Symbol-Optimierung Fehler: {e}")
                        print("🔄 Falle zurück auf Standard-Optimierung...")
                        use_symbol_specific = False
                
                elif optimization_choice == "2":
                    print("\n🔧 Starte Standard MULTI-THREADED Parameter-Optimierung...")
                    use_symbol_specific = False
                    
                elif optimization_choice == "3":
                    # ► ÜBERSPRINGEN – gespeicherte Optimierungen verwenden mit Prioritätensystem
                    print("⏭️ Optimierung übersprungen – lade gespeicherte Parameter...")
                    print("📋 Prioritätensystem: 1) Symbol-spezifisch → 2) Standard Multi-Threading → 3) Default")
                    
                    # Dateipfade definieren
                    data_dir = "data"
                    symbol_optimization_file = os.path.join(data_dir, "optimization_results_per_symbol.json")
                    standard_optimization_file = os.path.join(data_dir, "standard_multi_threading_optimization.json")
                    
                    try:
                        # ► PRIORITÄT 1: Symbol-spezifische Optimierungen laden
                        print("\n🔍 Suche symbol-spezifische Optimierungen...")
                        symbol_params_loaded = 0
                        for symbol in config['trading']['symbols']:
                            try:
                                params = symbol_optimizer.get_best_params_for_symbol(symbol)
                                if params:
                                    optimized_params[symbol] = params
                                    symbol_params_loaded += 1
                                    print(f"   ✅ {symbol}: Symbol-spezifische Parameter geladen")
                                else:
                                    print(f"   ⚠️  {symbol}: Keine symbol-spezifischen Parameter gefunden")
                            except Exception as symbol_error:
                                print(f"   ❌ {symbol}: Fehler beim Laden - {symbol_error}")
                        
                        print(f"📊 Symbol-spezifische Parameter: {symbol_params_loaded}/{len(config['trading']['symbols'])} geladen")
                        
                        # ► PRIORITÄT 2: Standard Multi-Threading Optimierung als Fallback
                        if symbol_params_loaded < len(config['trading']['symbols']):
                            print(f"\n🔍 Lade Standard Multi-Threading Optimierung als Fallback...")
                            
                            if os.path.exists(standard_optimization_file):
                                try:
                                    with open(standard_optimization_file, 'r', encoding='utf-8') as f:
                                        standard_data = json.load(f)
                                    
                                    # Prüfe Datenstruktur und extrahiere Parameter
                                    if 'parameters' in standard_data:
                                        standard_params = standard_data['parameters']
                                        optimization_info = standard_data.get('optimization_info', {})
                                        performance = standard_data.get('performance', {})
                                        
                                        print(f"   ✅ Standard-Optimierung gefunden!")
                                        print(f"   📅 Erstellt: {optimization_info.get('created_date', 'Unbekannt')}")
                                        print(f"   🏆 Score: {optimization_info.get('score', 0):.4f}")
                                        print(f"   💰 Avg Profit: {performance.get('avg_profit', 0):.2%}")
                                        print(f"   🎯 Win Rate: {performance.get('win_rate', 0):.2%}")
                                        
                                        # Verwende Standard-Parameter für Symbole ohne symbol-spezifische Parameter
                                        symbols_using_standard = []
                                        for symbol in config['trading']['symbols']:
                                            if symbol not in optimized_params:
                                                optimized_params[symbol] = standard_params
                                                symbols_using_standard.append(symbol)
                                        
                                        if symbols_using_standard:
                                            print(f"   📋 Standard-Parameter angewandt auf: {', '.join(symbols_using_standard)}")
                                        
                                    else:
                                        print(f"   ❌ Ungültige Struktur in Standard-Optimierung")
                                        
                                except Exception as standard_error:
                                    print(f"   ❌ Fehler beim Laden der Standard-Optimierung: {standard_error}")
                            else:
                                print(f"   ⚠️  Standard-Optimierung nicht gefunden: {standard_optimization_file}")
                                print(f"   💡 Tipp: Führe zuerst Option [2] aus, um Standard-Optimierung zu erstellen")
                        
                        # ► PRIORITÄT 3: Default-Parameter für verbleibende Symbole
                        symbols_without_params = [symbol for symbol in config['trading']['symbols'] if symbol not in optimized_params]
                        if symbols_without_params:
                            print(f"\n⚠️  Keine Parameter für {len(symbols_without_params)} Symbole gefunden")
                            print(f"📝 Verwende Default-Parameter für: {', '.join(symbols_without_params)}")
                            
                            # Default-Parameter setzen (falls verfügbar)
                            default_params = _get_default_parameters(config)
                            if default_params and 'parameters' in default_params:
                                for symbol in symbols_without_params:
                                    optimized_params[symbol] = default_params['parameters']
                        
                        # ► ERGEBNIS ANZEIGEN
                        if optimized_params:
                            total_symbols = len(config['trading']['symbols'])
                            print(f"\n✅ PARAMETER-ÜBERSICHT:")
                            print(f"   📊 Symbole gesamt: {total_symbols}")
                            print(f"   🎯 Mit symbol-spezifischen Parametern: {symbol_params_loaded}")
                            print(f"   ⚙️  Mit Standard-Parametern: {total_symbols - symbol_params_loaded}")
                            print(f"   🚀 Bereit für optimiertes Trading!")
                            
                            # Detaillierte Parameter-Übersicht (optional)
                            print(f"\n📋 PARAMETER-DETAILS PRO SYMBOL:")
                            for symbol in config['trading']['symbols']:
                                if symbol in optimized_params:
                                    params = optimized_params[symbol]
                                    param_type = "Symbol-spezifisch" if symbol_params_loaded > 0 and symbol in [s for s in config['trading']['symbols'][:symbol_params_loaded]] else "Standard"
                                    
                                    # Zeige wichtige Parameter
                                    tp = params.get('take_profit_pct', 0) * 100 if isinstance(params.get('take_profit_pct'), (int, float)) else 'N/A'
                                    sl = params.get('stop_loss_pct', 0) * 100 if isinstance(params.get('stop_loss_pct'), (int, float)) else 'N/A'
                                    rsi_os = params.get('rsi_oversold', 'N/A')
                                    
                                    print(f"   • {symbol} ({param_type}): TP={tp}%, SL={sl}%, RSI<{rsi_os}")
                                else:
                                    print(f"   • {symbol}: ❌ Keine Parameter geladen")
                        else:
                            print(f"\n❌ KEINE PARAMETER GEFUNDEN!")
                            print(f"💡 Empfehlung: Führe zuerst Option [1] oder [1] aus")
                            print(f"🔄 Starte mit Default-Konfiguration...")
                            
                    except Exception as e:
                        logging.warning(f"Überspringen-Modus Fehler: {e}")
                        print(f"❌ Fehler beim Laden gespeicherter Parameter: {e}")
                        print(f"🔄 Verwende Default-Konfiguration...")
                        optimized_params = {}


    if optimized_params:
        print(f"✅ Gesamtladen fertiger Parameter: {len(optimized_params)} Sets")
    else:
        print("⚠️ Keine Parameter verfügbar – starte mit Default-Konfiguration")

            else:
                print("\n✅ Alle Symbole sind aktuell optimiert (≤24h)")
                # Lade alle gespeicherten Parameter
                for symbol in config['trading']['symbols']:
                    best_params = symbol_optimizer.get_best_params_for_symbol(symbol)
                    if best_params:
                        optimized_params[symbol] = best_params
                
                if optimized_params:
                    print(f"✅ {len(optimized_params)} symbol-spezifische Parameter geladen")
                    symbol_optimizer.print_optimization_summary()
        
        except ImportError as e:
            logging.warning(f"Symbol-Optimizer nicht verfügbar: {e}")
            print("⚠️ Symbol-spezifische Optimierung nicht verfügbar, verwende Standard")
            use_symbol_specific = False
        except Exception as e:
            logging.error(f"Symbol-Optimizer Fehler: {e}")
            use_symbol_specific = False
        
        # Fallback auf Standard-Optimierung
        if not use_symbol_specific and config['optimization']['enabled']:
            print("\n🔧 Starte Standard MULTI-THREADED Parameter-Optimierung...")
            print("⚡ Mit Threading ist das deutlich schneller!")
            print("⏳ Geschätzte Zeit: 2-5 Minuten")
            
            start_time = time.time()
            
            try:
                standard_params = run_parameter_optimization_threaded(config, client)
                optimization_time = time.time() - start_time
                
                if standard_params:
                    print(f"✅ Standard-Optimierung abgeschlossen in {optimization_time:.1f} Sekunden!")
                    print(f"🎯 Beste Parameter gefunden:")
                    params_to_show = standard_params.get('parameters', standard_params)
                    for key, value in params_to_show.items():
                        if isinstance(value, (int, float, str)):
                            print(f"   {key}: {value}")
                    
                    # Verwende Standard-Parameter für alle Symbole
                    optimized_params = standard_params
                    
                    # Dashboard Update
                    update_optimization_status({
                        'completed': True,
                        'optimization_type': 'standard',
                        'best_params': params_to_show,
                        'best_score': standard_params.get('score', 0),
                        'expected_profit': standard_params.get('avg_profit', 0),
                        'expected_winrate': standard_params.get('win_rate', 0),
                        'total_tests': standard_params.get('total_tests', 0),
                        'optimization_time_seconds': optimization_time,
                        'ml_enabled': config.get('advanced', {}).get('use_machine_learning', False)
                    })
                else:
                    print("⚠️ Standard-Optimierung fehlgeschlagen, verwende Standard-Parameter")
                    
            except Exception as e:
                logging.error(f"Standard-Optimierung fehlgeschlagen: {e}")
                print(f"❌ Optimierung Fehler: {e}")
                print("🔄 Verwende Standard-Parameter")

    # Enhanced Bot erstellen
    print(f"\n🤖 Initialisiere Enhanced Trading Bot...")
    trader = None
    
    try:
        # Entscheide welche Parameter verwendet werden
        if isinstance(optimized_params, dict) and any(isinstance(v, dict) for v in optimized_params.values()):
            # Symbol-spezifische Parameter
            trader = EnhancedAdvancedBinanceTrader(config, optimized_params, symbol_specific=True)
            optimization_type = "Symbol-spezifisch"
        elif optimized_params:
            # Standard optimierte Parameter
            trader = EnhancedAdvancedBinanceTrader(config, optimized_params, symbol_specific=False)
            optimization_type = "Standard optimiert"
        else:
            # Standard Parameter
            trader = EnhancedAdvancedBinanceTrader(config, None, symbol_specific=False)
            optimization_type = "Standard"
        
        # Startbalance für Dashboard
        try:
            balances, total_value = trader.get_account_info()
            update_balance(total_value, total_value)
        except Exception as balance_error:
            logging.warning(f"Could not get initial balance: {balance_error}")
        
        # Status anzeigen
        print(f"\n{'='*70}")
        print(f"🎯 ENHANCED BINANCE TRADING BOT - SYMBOL-SPECIFIC")
        print(f"{'='*70}")
        print(f"📊 Symbole: {', '.join(trader.symbols)}")
        print(f"⏱️  Intervall: {trader.interval}")
        print(f"🛡️  Kapitalschutz: {trader.capital_protection_threshold*100}%")
        print(f"🚨 Emergency Stop: {trader.emergency_stop_loss*100}%")
        print(f"📈 Max Gleichzeitige Positionen: {trader.max_concurrent_positions}")
        print(f"🧵 Threading: {trader.max_symbol_threads} Symbol-Threads aktiv")
        print(f"⚙️  Parameter-Typ: {optimization_type}")
        
        # Parameter-Details anzeigen
        if hasattr(trader, 'symbol_specific_params') and trader.symbol_specific_params:
            print(f"🎯 Symbol-spezifische Optimierung: ✅ Aktiv")
            print(f"⚡ Optimierte Symbole: {len(trader.symbol_specific_params)}/{len(trader.symbols)}")
            
            # Zeige Parameter für jedes Symbol
            for symbol in trader.symbols:
                if symbol in trader.symbol_specific_params:
                    params = trader.symbol_specific_params[symbol]
                    tp = params.get('take_profit_pct', 0) * 100
                    sl = params.get('stop_loss_pct', 0) * 100
                    rsi_os = params.get('rsi_oversold', 'N/A')
                    conf = params.get('confidence_threshold', 0)
                    print(f"   📊 {symbol}: TP={tp:.1f}%, SL={sl:.1f}%, RSI<{rsi_os}, Conf={conf:.2f}")
                else:
                    print(f"   📝 {symbol}: Standard-Parameter")
        elif optimized_params and isinstance(optimized_params, dict):
            print(f"⚡ Standard-Optimierung: ✅ Aktiv")
            params = optimized_params.get('parameters', optimized_params)
            if isinstance(params, dict):
                rsi_os = params.get('rsi_oversold', 'N/A')
                tp = params.get('take_profit_pct', 0)
                sl = params.get('stop_loss_pct', 0)
                if isinstance(tp, (int, float)) and tp > 0:
                    print(f"   Take Profit: {tp*100:.1f}%")
                if isinstance(sl, (int, float)) and sl > 0:
                    print(f"   Stop Loss: {sl*100:.1f}%")
                print(f"   RSI Oversold: {rsi_os}")
        else:
            print(f"📝 Standard-Parameter: Aktiv")
        
        # System-Status
        ml_status = "❌ Nicht verfügbar"
        if hasattr(trader, 'ml_system') and trader.ml_system:
            ml_status = "✅ Aktiviert" if getattr(trader.ml_system, 'ml_enabled', False) else "⚪ Deaktiviert"
        
        sentiment_status = "❌ Nicht verfügbar"  
        if hasattr(trader, 'sentiment_analyzer') and trader.sentiment_analyzer:
            sentiment_status = "✅ Aktiviert" if getattr(trader.sentiment_analyzer, 'sentiment_enabled', False) else "⚪ Deaktiviert"
            
        print(f"🧠 Machine Learning: {ml_status}")
        print(f"📰 News Sentiment: {sentiment_status}")
        
        print(f"{'='*70}")
        print(f"🌐 Dashboard verfügbar auf:")
        print(f"   📊 Live Dashboard: http://localhost:5000")
        print(f"   📈 API Status: http://localhost:5000/status")
        print(f"   🏆 Beste Parameter: http://localhost:5000/best-params")
        print(f"   📋 Test Ergebnisse: http://localhost:5000/test-results")
        print(f"   ⚙️  Optimierung: http://localhost:5000/optimization")
        print(f"{'='*70}")
        print(f"📈 PERFORMANCE REPORTS alle 5 Minuten im Terminal")
        print(f"   • Risk/Reward Verhältnis")
        print(f"   • Beste Marktbedingungen pro Symbol")  
        print(f"   • Symbol-spezifische Performance")
        print(f"   • Zeitbasierte Performance-Analyse")
        print(f"{'='*70}\n")
        
        # Trading starten
        print("🚀 Trading Bot wird gestartet...")
        trading_thread = trader.start()
        
        print("✅ Bot läuft mit optimierten Parametern!")
        if hasattr(trader, 'symbol_specific_params') and trader.symbol_specific_params:
            print(f"🎯 Jedes Symbol nutzt seine optimalen Parameter für maximalen Profit!")
        print("📊 Detaillierte Performance-Reports alle 5 Minuten im Terminal.")
        print("🌐 Dashboard: http://localhost:5000")
        print("🛑 Drücke ENTER um den Bot zu stoppen...\n")
        
        # Auf Eingabe warten für Stop
        input()
        
    except KeyboardInterrupt:
        print("\n⚠️ Bot wird gestoppt...")
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        logging.error(f"Main function error: {e}", exc_info=True)
    finally:
        if trader:
            print("🛑 Bot wird beendet...")
            trader.stop()
            optimization_status['running'] = False
            print("✅ Bot sicher beendet")

    def _get_default_parameters(config):
    """
    Standard-Parameter wenn keine Optimierungen verfügbar sind
    (Fallback-Funktion für main.py)
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


if __name__ == "__main__":
    main()





