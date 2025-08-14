#!/usr/bin/env python3
"""
Advanced Binance Trading Bot - Main Entry Point with Symbol-Specific Optimization
"""

import sys
import time
import logging
from pathlib import Path

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
                    # ÜBERSPRINGEN – gespeicherte Einstellungen laden
                    print("⏭️ Optimierung übersprungen – lade gespeicherte Parameter…")
                    try:
                        for symbol in config['trading']['symbols']:
                            best = symbol_optimizer.get_best_params_for_symbol(symbol)
                            if best:
                                optimized_params[symbol] = best
                        if optimized_params:
                            print(f"✅ {len(optimized_params)} gespeicherte Parameter geladen")
                    except Exception as e:
                        logging.warning(f"Fehler beim Laden gespeicherter Parameter: {e}")
                        use_symbol_specific = False
                else:
                    # Fallback: Normale Analyse ohne Optimierung
                    print("❌ Ungültige Auswahl – keine Optimierung wird angewendet")
                    use_symbol_specific = False
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

if __name__ == "__main__":
    main()


