#!/usr/bin/env python3
"""
Advanced Binance Trading Bot - Main Entry Point with Symbol-Specific Optimization
"""

import sys
import time
import logging
import json
import os
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
        client = Client(
            config['api']['key'],
            config['api']['secret'],
            testnet=config['api'].get('testnet', True)
        )
        client.get_account()
        print("✅ Binance API Verbindung erfolgreich")
    except Exception as e:
        print(f"❌ Binance API Fehler: {e}")
        return

    def test_symbols():
        """Teste verfügbare Symbole"""
        available = []
        for symbol in config['trading']['symbols']:
            try:
                ticker = client.get_symbol_ticker(symbol=symbol)
                print(f"✅ {symbol}: {ticker['price']} USDT - VERFÜGBAR")
                available.append(symbol)
            except Exception as e:
                print(f"❌ {symbol}: NICHT VERFÜGBAR - {e}")
        return available

    print("\n🔍 Teste verfügbare Symbole...")
    available_symbols = test_symbols()
    if not available_symbols:
        print("❌ Keine verfügbaren Symbole gefunden!")
        return

    optimized_params = {}
    use_symbol_specific = True

    if config['optimization']['enabled']:
        print("\n🎯 SYMBOL-SPEZIFISCHE OPTIMIERUNG verfügbar!")
        try:
            from optimization.symbol_optimizer import SymbolSpecificOptimizer
            symbol_optimizer = SymbolSpecificOptimizer(config, client)
        else:
            # Prüfe Optimierungs-Status
            needs_opt = False
            status_info = []
            for s in config['trading']['symbols']:
                should, reason = symbol_optimizer.should_optimize_symbol(s)
                status_info.append((s, should, reason))
                if should:
                    needs_opt = True
            print("\n📋 Optimierungs-Status der Symbole:")
            for s, sh, rs in status_info:
                label = "🔄 BENÖTIGT" if sh else "✅ AKTUELL"
                print(f"   {label} {s}: {rs}")

            if needs_opt:
                count = sum(1 for _, sh, _ in status_info if sh)
                print(f"\n🤔 {count} Symbole benötigen Optimierung")
                choice = input(
                    "\nOptimierung starten?\n"
                    "  [1] Symbol-spezifische Optimierung (Empfohlen)\n"
                    "  [2] Standard Multi-Threading Optimierung\n"
                    "  [3] Überspringen\n"
                    "Wahl (1/2/3): "
                ).strip()
                if choice == "1":
                    print("\n🎯 Starte SYMBOL-SPEZIFISCHE Optimierung...")
                    start_t = time.time()
                    try:
                        results = symbol_optimizer.optimize_all_symbols()
                        dur = time.time() - start_t
                        print(f"✅ Symbol-Optimierung abgeschlossen in {dur/60:.1f} Minuten!")
                        symbol_optimizer.print_optimization_summary()
                        for s in config['trading']['symbols']:
                            bp = symbol_optimizer.get_best_params_for_symbol(s)
                            if bp:
                                optimized_params[s] = bp
                                tp = bp.get('take_profit_pct',0)*100
                                sl = bp.get('stop_loss_pct',0)*100
                                rsi = bp.get('rsi_oversold',0)
                                print(f"🎯 {s}: TP={tp:.1f}%, SL={sl:.1f}%, RSI<{rsi}")
                        update_optimization_status({
                            'completed': True,
                            'optimization_type': 'symbol_specific',
                            'symbols_optimized': len(optimized_params),
                            'total_symbols': len(config['trading']['symbols']),
                            'avg_profit': sum(r['total_profit'] for r in results.values())/len(results)*100 if results else 0,
                            'successful_symbols': sum(1 for r in results.values() if r.get('profit_meets_threshold')),
                            'profit_threshold': 25.0,
                            'optimization_time_minutes': dur/60,
                            'ml_enabled': config.get('advanced',{}).get('use_machine_learning', False)
                        })
                    except Exception as e:
                        logging.error(f"Symbol-Optimierung Fehler: {e}")
                        print("🔄 Fallback auf Standard-Optimierung...")
                        use_symbol_specific = False
                elif choice == "2":
                    print("\n🔧 Starte Standard MULTI-THREADED Parameter-Optimierung...")
                    use_symbol_specific = False
                elif choice == "3":
                    # ► ÜBERSPRINGEN – gespeicherte Optimierungen verwenden
                    print("⏭️ Optimierung übersprungen – lade gespeicherte Parameter...")
                    print("📋 Priorität: 1) symbol-specific → 2) standard → 3) default")
                    data_dir = "data"
                    so_file = os.path.join(data_dir, "optimization_results_per_symbol.json")
                    st_file = os.path.join(data_dir, "standard_multi_threading_optimization.json")
                    try:
                        # Priorität 1
                        loaded = 0
                        for s in config['trading']['symbols']:
                            bp = symbol_optimizer.get_best_params_for_symbol(s)
                            if bp:
                                optimized_params[s] = bp
                                loaded += 1
                                print(f"   ✅ {s}: Symbol-Parameter geladen")
                        print(f"📊 Symbol-Parameter: {loaded}/{len(config['trading']['symbols'])}")
                        # Priorität 2
                        if loaded < len(config['trading']['symbols']):
                            if os.path.exists(st_file):
                                with open(st_file,'r') as f:
                                    std = json.load(f).get('parameters',{})
                                print("   ✅ Standard-Optimierung gefunden")
                                for s in config['trading']['symbols']:
                                    if s not in optimized_params:
                                        optimized_params[s] = std
                                print(f"   📋 Standard-Parameter für: {', '.join([s for s in config['trading']['symbols'] if s not in optimized_params])}")
                            else:
                                print("   ⚠️ Standard-Optimierung nicht gefunden")
                        # Priorität 3
                        missing = [s for s in config['trading']['symbols'] if s not in optimized_params]
                        if missing:
                            print(f"📝 Default-Parameter für: {', '.join(missing)}")
                            default = _get_default_parameters(config)['parameters']
                            for s in missing:
                                optimized_params[s] = default
                    except Exception as e:
                        logging.warning(f"Überspringen-Modus Fehler: {e}")
                        print("🔄 Verwende Default-Konfiguration...")
            else:
                print("\n✅ Alle Symbole aktuell optimiert")
                for s in config['trading']['symbols']:
                    bp = symbol_optimizer.get_best_params_for_symbol(s)
                    if bp:
                        optimized_params[s] = bp
                if optimized_params:
                    symbol_optimizer.print_optimization_summary()
        except Exception as e:
            logging.error(f"Symbol-Optimizer Fehler: {e}")
            use_symbol_specific = False

    # Fallback Standard-Optimierung
    if not use_symbol_specific and config['optimization']['enabled']:
        print("\n🔧 Starte Standard MULTI-THREADED Parameter-Optimierung...")
        start_t = time.time()
        try:
            std = run_parameter_optimization_threaded(config, client)
            dur = time.time() - start_t
            print(f"✅ Standard-Optimierung abgeschlossen in {dur:.1f}s")
            if isinstance(std, dict) and 'parameters' in std:
                optimized_params = std['parameters']
            update_optimization_status({
                'completed': True,
                'optimization_type': 'standard',
                'best_params': optimized_params
            })
        except Exception as e:
            logging.error(f"Standard-Optimierung Fehler: {e}")
            print("🔄 Verwende Default-Parameter...")

    # Enhanced Bot erstellen und starten
    print("\n🤖 Initialisiere Enhanced Trading Bot...")
    trader = None
    try:
        if any(isinstance(v, dict) for v in optimized_params.values()):
            trader = EnhancedAdvancedBinanceTrader(config, optimized_params, symbol_specific=True)
            opt_type = "Symbol-spezifisch"
        elif optimized_params:
            trader = EnhancedAdvancedBinanceTrader(config, optimized_params, symbol_specific=False)
            opt_type = "Standard optimiert"
        else:
            trader = EnhancedAdvancedBinanceTrader(config, None, symbol_specific=False)
            opt_type = "Standard"

        try:
            bal, tot = trader.get_account_info()
            update_balance(tot, tot)
        except Exception as e:
            logging.warning(f"Balance-Fehler: {e}")

        print(f"\n{'='*70}")
        print("🎯 ENHANCED BINANCE TRADING BOT")
        print(f"{'='*70}")
        print(f"📊 Symbole: {', '.join(trader.symbols)}")
        print(f"⚙️ Parameter-Typ: {opt_type}")
        print(f"{'='*70}\n")

        print("🚀 Trading Bot wird gestartet...")
        _ = trader.start()
        input("🛑 ENTER zum Stoppen...")
    except KeyboardInterrupt:
        print("\n⚠️ Bot wird gestoppt...")
    except Exception as e:
        logging.error(f"Main-Fehler: {e}", exc_info=True)
        print(f"❌ Unerwarteter Fehler: {e}")
    finally:
        if trader:
            print("🛑 Bot wird beendet...")
            # SICHERHEITSABFRAGE: Alle offenen Positionen glattstellen
            try:
                trader.close_all_positions()  # Muss in deinem Trader implementiert sein
                print("✅ Alle offenen Positionen wurden verkauft.")
            except Exception as e:
                print(f"❌ Fehler beim Schließen der Positionen: {e}")
            trader.stop()
            optimization_status['running'] = False
            print("✅ Bot sicher beendet")

def _get_default_parameters(config):
    """
    Standard-Parameter wenn keine Optimierungen verfügbar sind
    (Fallback-Funktion für main.py)
    """
    parameter_ranges = config.get('parameter_ranges', {})
    default_params = {}
    for key, values in parameter_ranges.items():
        if isinstance(values, list) and values:
            default_params[key] = values[len(values)//2]
        else:
            fallback = {
                'rsi_period':14,'rsi_oversold':30,'rsi_overbought':70,
                'ema_fast':12,'ema_slow':26,
                'take_profit_pct':0.015,'stop_loss_pct':0.02,
                'trailing_sl_pct':0.006,'volume_threshold':2.0,
                'bb_period':20,'confidence_threshold':0.7
            }
            default_params[key] = fallback.get(key,0.5)
    return {
        'parameters': default_params, 'score':0.5, 'method':'default',
        'total_tests':0,'best_win_rate':0.0,'best_profit':0.0,'optimization_time':0.0
    }

if __name__ == "__main__":
    main()










