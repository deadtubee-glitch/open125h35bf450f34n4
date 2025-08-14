"""
Enhanced Trading Module
Extended functionality with ML integration and performance tracking
"""

import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.base_trader import AdvancedBinanceTrader
from ml.ml_system import MLDaytradingSystem
from analysis.performance import PerformanceAnalyzer
from analysis.sentiment import NewsSentimentAnalyzer

from utils.flask_app import (
    start_flask_server,
    update_live_status,
    record_trade,
    update_balance,
    update_performance_metrics,
    update_optimization_status
)

logger = logging.getLogger(__name__)

class EnhancedAdvancedBinanceTrader(AdvancedBinanceTrader):
    """Enhanced version with performance tracking and ML integration"""

    def __init__(self, config, optimized_params=None, symbol_specific=False):
        # Symbol-spezifische Parameter-Behandlung
        self.symbol_specific_mode = symbol_specific
        
        if symbol_specific and isinstance(optimized_params, dict):
            # optimized_params ist {symbol: params} für symbol-spezifische Mode
            self.symbol_specific_params = optimized_params
            # Keine globalen Parameter an parent weitergeben
            super().__init__(config, None)
            logger.info(f"✅ Symbol-spezifische Parameter für {len(self.symbol_specific_params)} Symbole geladen")
        else:
            # Standard-Modus: globale Parameter
            self.symbol_specific_params = {}
            super().__init__(config, optimized_params)
            if optimized_params:
                logger.info("✅ Standard optimierte Parameter geladen")

        # Performance analyzer
        self.performance_analyzer = PerformanceAnalyzer()
        self.last_performance_report = datetime.now()

        # ML system - safe initialization
        self.ml_system = None
        try:
            self.ml_system = MLDaytradingSystem(config)
            logger.info("🧠 ML system successfully initialized")
        except Exception as e:
            logger.warning(f"⚠️ ML system initialization failed: {e}")
            class DummyML:
                def __init__(self):
                    self.ml_enabled = False
                    self.feature_window = 100
                def save_ml_state(self): pass
                def update_live_data(self, symbol, df): pass
                def get_ml_prediction(self, symbol, features):
                    return {
                        'signal':'NEUTRAL','confidence':0.0,'predictions':{},'probabilities':{},
                        'ensemble_score':0.5,'model_count':0
                    }
                def extract_features(self, df, symbol): return None
            self.ml_system = DummyML()

        # Sentiment analyzer - safe initialization
        self.sentiment_analyzer = None
        try:
            self.sentiment_analyzer = NewsSentimentAnalyzer(config)
            logger.info("📰 Sentiment analyzer successfully initialized")
        except Exception as e:
            logger.warning(f"⚠️ Sentiment analyzer initialization failed: {e}")
            class DummySentiment:
                def __init__(self): self.sentiment_enabled=False
                def get_sentiment_signal(self, symbol):
                    return {
                        'signal':'NEUTRAL','sentiment':'neutral','strength':0.0,
                        'sentiment_score':0.0,'confidence':0.0,'articles_count':0
                    }
            self.sentiment_analyzer = DummySentiment()

        # Threading for parallel symbol processing
        self.max_symbol_threads = min(4, len(self.symbols))

        # Start Flask server once
        try:
            start_flask_server()
            logger.info("🌐 Flask server started")
        except Exception as e:
            logger.warning(f"⚠️ Flask server start failed: {e}")

        logger.info("✅ Enhanced trading bot initialized")

    def get_trading_params_for_symbol(self, symbol):
        """Gibt symbol-spezifische Parameter zurück oder fallback auf Standard"""
        if self.symbol_specific_mode and symbol in self.symbol_specific_params:
            return self.symbol_specific_params[symbol]
        return self.trading_params  # Fallback auf Standard

    def generate_trading_signal(self, indicators, symbol, current_price, ml_features=None):
        """Generate trading signals mit symbol-spezifischen Parametern"""
        # Hole symbol-spezifische Parameter
        raw_params = self.get_trading_params_for_symbol(symbol)
        
        # 🔧 FIX: Alle Parameter zu Float konvertieren (löst den BTCUSDT-Fehler)
        params = {
            'rsi_oversold': float(raw_params['rsi_oversold']),
            'rsi_overbought': float(raw_params['rsi_overbought']),
            'volume_threshold': float(raw_params['volume_threshold']),
            'confidence_threshold': float(raw_params.get('confidence_threshold', 0.6)),
            'take_profit_pct': float(raw_params.get('take_profit_pct', 0.03)),
            'stop_loss_pct': float(raw_params.get('stop_loss_pct', 0.02)),
            'trailing_sl_pct': float(raw_params.get('trailing_sl_pct', 0.015))
        }
        
        # Debug für BTCUSDT Type-Fix
        if symbol == 'BTCUSDT':
            logger.info(f"🔧 {symbol} Type Fix: "
                       f"rsi_oversold={params['rsi_oversold']} (type: {type(params['rsi_oversold'])})")
        
        signals={'action':'HOLD','confidence':0.0,'reasons':[]}
        buy_signals=0; confidence=0.0
    
        # Technical indicators mit symbol-spezifischen Parametern (jetzt crash-safe)
        if indicators['rsi'] < params['rsi_oversold']:
            buy_signals += 2; confidence += 0.2
            signals['reasons'].append(f"RSI Oversold ({indicators['rsi']:.1f} < {params['rsi_oversold']})")
        elif indicators['rsi'] > params['rsi_overbought']:
            buy_signals -= 2
            signals['reasons'].append(f"RSI Overbought ({indicators['rsi']:.1f} > {params['rsi_overbought']})")
    
        if indicators['ema_fast'] > indicators['ema_slow']:
            buy_signals += 1; confidence += 0.15
            signals['reasons'].append("EMA Bullish")
        else:
            buy_signals -= 1
            signals['reasons'].append("EMA Bearish")
    
        if indicators['macd'] > indicators['macd_signal']:
            buy_signals += 1; confidence += 0.1
            signals['reasons'].append("MACD Bullish")
    
        if current_price < indicators['bb_lower']:
            buy_signals += 1; confidence += 0.1
            signals['reasons'].append("BB Lower Touch")
        elif current_price > indicators['bb_upper']:
            buy_signals -= 1
            signals['reasons'].append("BB Upper Touch")
    
        if indicators['volume_ratio'] > params['volume_threshold']:
            confidence += 0.05
            signals['reasons'].append("High Volume")
    
        # ML integration
        ml_enabled_actual = False  # Track ob ML wirklich aktiv ist
        if self.ml_system.ml_enabled and ml_features:
            ml_enabled_actual = True
            ml_pred=self.ml_system.get_ml_prediction(symbol,ml_features)
            s=ml_pred['signal']; c=ml_pred['confidence']; score=ml_pred['ensemble_score']
            
            if s == 'STRONG_BUY':
                buy_signals += 3; confidence += 0.3 * c
                signals['reasons'].append(f"ML Strong Buy ({score:.2f})")
            elif s == 'BUY':
                buy_signals += 2; confidence += 0.2 * c
                signals['reasons'].append(f"ML Buy ({score:.2f})")
            elif s == 'STRONG_SELL':
                buy_signals -= 3
                signals['reasons'].append(f"ML Strong Sell ({score:.2f})")
            elif s == 'SELL':
                buy_signals -= 2
                signals['reasons'].append(f"ML Sell ({score:.2f})")
            
            confidence = min(confidence + (c * 0.1), 1.0)
            signals['ml_prediction'] = ml_pred
    
        # Sentiment integration
        nsig=self.sentiment_analyzer.get_sentiment_signal(symbol)
        if nsig['signal'] == 'BULLISH':
            buy_signals += 2; confidence += 0.2 * nsig['strength']
            signals['reasons'].append(f"Bullish News ({nsig['sentiment_score']:.2f})")
        elif nsig['signal'] == 'WEAK_BULLISH':
            buy_signals += 1; confidence += 0.1 * nsig['strength']
            signals['reasons'].append("Positive News")
        elif nsig['signal'] == 'BEARISH':
            buy_signals -= 2; confidence -= 0.1
            signals['reasons'].append(f"Bearish News ({nsig['sentiment_score']:.2f})")
        elif nsig['signal'] == 'WEAK_BEARISH':
            buy_signals -= 1
            signals['reasons'].append("Negative News")
    
        # 🔍 DEBUG-LOGGING (mit sicheren Float-Parametern)
        confidence_threshold = params['confidence_threshold']  # Bereits Float
        
        # Detaillierte Debug-Ausgabe für Problemdiagnose
        logger.info(f"🔍 {symbol} Signal-Debug:")
        logger.info(f"   RSI: {indicators['rsi']:.1f} (Oversold<{params['rsi_oversold']}, Overbought>{params['rsi_overbought']})")
        logger.info(f"   EMA: Fast={indicators['ema_fast']:.4f}, Slow={indicators['ema_slow']:.4f}")
        logger.info(f"   MACD: {indicators['macd']:.4f} vs Signal={indicators['macd_signal']:.4f}")
        logger.info(f"   Volume Ratio: {indicators['volume_ratio']:.2f} (Threshold={params['volume_threshold']})")
        logger.info(f"   ML: Enabled={ml_enabled_actual}, Features={'Yes' if ml_features else 'No'}")
        if ml_enabled_actual and ml_features:
            logger.info(f"   ML Prediction: {ml_pred.get('signal', 'NONE')}, Confidence={ml_pred.get('confidence', 0):.3f}")
        logger.info(f"   Sentiment: {nsig['signal']}, Score={nsig.get('sentiment_score', 0):.2f}")
        logger.info(f"   🎯 RESULT: buy_signals={buy_signals}, confidence={confidence:.3f}, threshold={confidence_threshold}")
        logger.info(f"   Reasons: {', '.join(signals['reasons']) if signals['reasons'] else 'None'}")
    
        # 🧪 TEST MODE für niedrigere Schwelle (richtig definiert)
        test_mode = False
        if symbol in ['LINKUSDT', 'AVAXUSDT'] and buy_signals >= 1 and confidence > 0.15:
            test_mode = True
            logger.info(f"🧪 TEST MODE ACTIVE for {symbol}: buy_signals={buy_signals}, confidence={confidence:.3f}")
            
        # Final decision mit symbol-spezifischen Schwellenwerten
        if buy_signals >= 1 and confidence >= confidence_threshold:
            signals['action'] = 'BUY'
            signals['confidence'] = confidence
            logger.info(f"✅ {symbol}: SIGNAL TRIGGERED - Action=BUY, Confidence={confidence:.1%}")
        elif test_mode:  # Test-Modus Override (jetzt definiert)
            signals['action'] = 'BUY'
            signals['confidence'] = confidence
            logger.info(f"🧪 {symbol}: TEST MODE BUY - Action=BUY, Confidence={confidence:.1%}")
        elif buy_signals <= -2:
            signals['action'] = 'SELL'
            signals['confidence'] = confidence
            logger.info(f"📉 {symbol}: SELL SIGNAL - Action=SELL, Confidence={confidence:.1%}")
        else:
            logger.info(f"⏸️ {symbol}: No signal - buy_signals={buy_signals} (need ≥1), "
                       f"confidence={confidence:.1%} (need ≥{confidence_threshold:.1%})")
    
        signals['sentiment'] = nsig
        
        # Zusätzliche Debug-Daten für Dashboard
        signals['debug_info'] = {
            'buy_signals': buy_signals,
            'confidence_threshold': confidence_threshold,
            'ml_enabled': ml_enabled_actual,
            'ml_features_available': bool(ml_features),
            'rsi_status': 'Oversold' if indicators['rsi'] < params['rsi_oversold'] else 
                         'Overbought' if indicators['rsi'] > params['rsi_overbought'] else 'Neutral',
            'test_mode_active': test_mode  # Jetzt definiert
        }
    
        return signals




    def should_exit_position(self, symbol, current_price, indicators):
        """Prüft Exit-Bedingungen mit symbol-spezifischen Parametern"""
        if symbol not in self.positions:
            return False, ""
        
        # Hole symbol-spezifische Parameter
        params = self.get_trading_params_for_symbol(symbol)
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        profit_pct = (current_price - entry_price) / entry_price
        
        # SOFORTIGER NOTFALL-STOP bei großen Verlusten
        if profit_pct <= -0.03:  # 3% harter Stop
            return True, f"NOTFALL-STOP ({profit_pct*100:.2f}%)"
        
        # Take Profit mit symbol-spezifischen Werten
        if profit_pct >= (params['take_profit_pct'] * 0.8):  # 80% vom Ziel
            return True, f"Take Profit ({profit_pct*100:.2f}%)"
        
        # Dynamischer Stop Loss mit symbol-spezifischen Parametern
        base_stop = params['stop_loss_pct']
        
        # RSI-basierter Stop
        if indicators['rsi'] > params['rsi_overbought'] and profit_pct > 0:
            return True, f"RSI Momentum Exit ({profit_pct*100:.2f}%)"
        
        # Standard Stop Loss
        if profit_pct <= -base_stop:
            return True, f"Stop Loss ({profit_pct*100:.2f}%)"
        
        # Trailing Stop
        if symbol in self.trailing_stops:
            trailing_stop_price = self.trailing_stops[symbol]
            if profit_pct > 0.005:  # 0.5% Gewinn
                if current_price <= trailing_stop_price:
                    return True, f"Trailing Stop ({profit_pct*100:.2f}%)"
        
        return False, ""

    def update_trailing_stop(self, symbol, current_price):
        """Aktualisiert Trailing Stop mit symbol-spezifischen Parametern"""
        if symbol not in self.positions:
            return
        
        # Hole symbol-spezifische Parameter    
        params = self.get_trading_params_for_symbol(symbol)
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        profit_pct = (current_price - entry_price) / entry_price
        
        # Dynamischer Trailing Stop
        if profit_pct > 0.02:  # Bei >2% Gewinn: engerer Trailing Stop
            trailing_distance = 0.008  # 0.8%
        elif profit_pct > 0.01:  # Bei >1% Gewinn: normaler Trailing Stop
            trailing_distance = 0.012  # 1.2%
        else:
            trailing_distance = params['trailing_sl_pct']  # Symbol-spezifisch
        
        new_trailing_stop = current_price * (1 - trailing_distance)
        
        if symbol not in self.trailing_stops:
            if profit_pct > 0.005:  # 0.5% Mindestgewinn
                self.trailing_stops[symbol] = new_trailing_stop
            return
        
        current_trailing_stop = self.trailing_stops[symbol]
        if new_trailing_stop > current_trailing_stop:
            self.trailing_stops[symbol] = new_trailing_stop
            logger.info(f"{symbol}: Trailing Stop aktualisiert: {new_trailing_stop:.4f} "
                       f"(Gewinn: {profit_pct*100:.2f}%)")

    def place_buy_order(self, symbol, current_price, signals, balance):
        """Platziert Kauforder mit symbol-spezifischen Parametern"""
        try:
            # Hole symbol-spezifische Parameter
            params = self.get_trading_params_for_symbol(symbol)
            
            if not self.check_capital_protection():
                return False
            
            # Cooldown prüfen
            if symbol in self.last_trade_time:
                time_since_last = (datetime.now() - self.last_trade_time[symbol]).total_seconds()
                if time_since_last < self.cooldown_period:
                    return False
            
            if len(self.positions) >= self.max_concurrent_positions:
                return False
            
            # Symbol-Info abrufen
            symbol_info = self.get_symbol_info(symbol)
            min_qty = symbol_info['min_qty']
            step_size = symbol_info['step_size']
            
            # Position Size berechnen
            risk_amount = balance * self.max_risk_per_trade
            raw_quantity = (risk_amount / current_price) * 0.99
            quantity = self.round_quantity(raw_quantity, step_size)
            
            if quantity < min_qty:
                logger.warning(f"{symbol}: Quantity {quantity:.6f} under minimum {min_qty:.6f}")
                quantity = min_qty
                
            required_usdt = quantity * current_price * 1.01
            if required_usdt > balance:
                logger.warning(f"{symbol}: Not enough balance")
                return False
            
            order_value = quantity * current_price
            if order_value < 10:
                logger.warning(f"{symbol}: Order value {order_value:.2f} under minimum 10 USDT")
                return False
            
            logger.info(f"{symbol}: Kaufsignal - Confidence: {signals['confidence']:.2f}")
            logger.info(f"Parameter-Set: TP={params['take_profit_pct']*100:.1f}%, SL={params['stop_loss_pct']*100:.1f}%")
            logger.info(f"Signale: {', '.join(signals['reasons'])}")
            logger.info(f"Order: {quantity:.6f} für {order_value:.2f} USDT")
            
            # Market Buy Order
            order = self.client.order_market_buy(
                symbol=symbol,
                quantity=f"{quantity:.{symbol_info['base_precision']}f}"
            )
            
            if order and order.get('status') == 'FILLED':
                fills = order.get('fills', [])
                if fills:
                    total_qty = sum(float(fill['qty']) for fill in fills)
                    total_cost = sum(float(fill['qty']) * float(fill['price']) for fill in fills)
                    avg_price = total_cost / total_qty if total_qty > 0 else current_price
                else:
                    avg_price = current_price
                
                executed_qty = float(order.get('executedQty', quantity))
                
                # Position speichern mit verwendeten Parametern
                self.positions[symbol] = {
                    'entry_price': avg_price,
                    'quantity': executed_qty,
                    'order_id': order['orderId'],
                    'timestamp': datetime.now(),
                    'confidence': signals['confidence'],
                    'reasons': signals['reasons'],
                    'used_params': params  # Speichere verwendete Parameter
                }
                
                # Trailing Stop mit symbol-spezifischen Parametern
                initial_stop = avg_price * (1 - params['trailing_sl_pct'])
                self.trailing_stops[symbol] = initial_stop
                
                self.last_trade_time[symbol] = datetime.now()
                
                logger.info(f"✅ KAUF ERFOLGREICH {symbol}: Preis: {avg_price:.4f}, Menge: {executed_qty:.6f}")
                logger.info(f"Symbol-spezifischer Trailing Stop: {initial_stop:.4f}")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Kauforder fehlgeschlagen {symbol}: {e}")
        
        return False

    def _get_bb_position(self, price, indicators):
        """Determine Bollinger Band position"""
        bb_upper=indicators.get('bb_upper',price)
        bb_lower=indicators.get('bb_lower',price)
        if price>bb_upper: return 'ABOVE_UPPER'
        if price<bb_lower: return 'BELOW_LOWER'
        return 'MIDDLE'

    def process_symbol(self, symbol):
        """Process a symbol with ML integration and enhanced debugging"""
        try:
            if self.emergency_stop: 
                return
    
            df = self.get_klines_data(symbol, self.ml_system.feature_window if self.ml_system.ml_enabled else 100)
            if df is None or len(df) < 50: 
                return
    
            if self.ml_system.ml_enabled:
                self.ml_system.update_live_data(symbol, df)
                if len(df) > 1:
                    change = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1)
                    self.ml_system.validate_predictions(symbol, change)
    
            indicators = self.calculate_indicators(df)
            if indicators is None: 
                return
    
            price = df['close'].iloc[-1]
            balances, total = self.get_account_info()
            usdt_bal = balances.get('USDT', {}).get('free', 0)
            in_pos = (symbol in self.positions)
    
            # Extract ML features
            ml_feat = None
            signals = {'action': 'HOLD', 'confidence': 0.0, 'reasons': []}
            
            if not in_pos:
                if self.ml_system.ml_enabled:
                    try: 
                        ml_feat = self.ml_system.extract_features(df, symbol)
                        
                        # 🔍 ML DEBUG - Schritt 2 aus der Antwort
                        logger.info(f"🔍 {symbol} ML Debug: Features available={bool(ml_feat)}, "
                                   f"Features type={type(ml_feat)}")
                        
                        # Prüfe ML-System Type
                        if hasattr(self.ml_system, '__class__'):
                            logger.info(f"ML System Type: {self.ml_system.__class__.__name__}")
                            
                        # Prüfe ML-Models geladen
                        if hasattr(self.ml_system, 'models_loaded'):
                            logger.info(f"ML Models loaded: {self.ml_system.models_loaded}")
                        elif hasattr(self.ml_system, 'ml_enabled'):
                            logger.info(f"ML Status: {self.ml_system.ml_enabled}")
                            
                    except Exception as e: 
                        logger.warning(f"ML extraction failed {symbol}: {e}")
                        ml_feat = None
                
                # 🔍 VOLUME DEBUG - Schritt 3 aus der Antwort
                if 'volume_ratio' in indicators:
                    current_volume = df['volume'].iloc[-1] if len(df) > 0 else 0
                    volume_sma = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 0
                    logger.info(f"🔍 {symbol} Volume Debug: current={current_volume:.2f}, "
                               f"sma={volume_sma:.2f}, ratio={indicators['volume_ratio']:.4f}")
                
                signals = self.generate_trading_signal(indicators, symbol, price, ml_feat)
                
                # 🧪 TEST MODE für ein Symbol (optional)
                if symbol == 'BTCUSDT' and signals['confidence'] > 0.1:  # Test nur wenn basic confidence da ist
                    logger.info(f"🧪 TEST MODE ACTIVE for {symbol}: Original confidence={signals['confidence']:.3f}")
                    # Senke temporär die Schwelle nur für dieses Symbol
                    temp_threshold = 0.15
                    params = self.get_trading_params_for_symbol(symbol)
                    if signals['confidence'] >= temp_threshold:
                        logger.info(f"🧪 TEST: {symbol} würde mit Threshold {temp_threshold} kaufen!")
                
                if (signals['action'] == 'BUY' and 
                    usdt_bal > self.min_usdt_balance and 
                    self.check_capital_protection()):
                    self.place_buy_order(symbol, price, signals, usdt_bal)
            else:
                self.update_trailing_stop(symbol, price)
                exit_ok, reason = self.should_exit_position(symbol, price, indicators)
                if exit_ok: 
                    self.enhanced_place_sell_order(symbol, reason)
    
            # Build enhanced status
            status = self.get_enhanced_status_data(symbol, indicators, price, signals, in_pos)
    
            # Symbol-spezifische Parameter zu Status hinzufügen
            if self.symbol_specific_mode and symbol in self.symbol_specific_params:
                params = self.symbol_specific_params[symbol]
                status.update({
                    'symbol_optimized': True,
                    'take_profit_pct': params.get('take_profit_pct', 0) * 100,
                    'stop_loss_pct': params.get('stop_loss_pct', 0) * 100,
                    'rsi_oversold_threshold': params.get('rsi_oversold', 35),
                    'confidence_threshold': params.get('confidence_threshold', 0.6)
                })
            else:
                status.update({'symbol_optimized': False})
    
            # ML Status für Dashboard
            if self.ml_system.ml_enabled and ml_feat is not None:
                try:
                    mp = self.ml_system.get_ml_prediction(symbol, ml_feat)
                    
                    # 🔍 ML PREDICTION DEBUG
                    logger.info(f"🧠 {symbol} ML Prediction Details: "
                               f"signal={mp.get('signal', 'NONE')}, "
                               f"confidence={mp.get('confidence', 0):.3f}, "
                               f"ensemble_score={mp.get('ensemble_score', 0):.3f}")
                    
                    status.update({
                        'ml_signal': mp.get('signal', 'NEUTRAL'),
                        'ml_confidence': round(mp.get('confidence', 0.0), 3),
                        'ml_score': round(mp.get('ensemble_score', 0.0), 3),
                        'ml_models': mp.get('model_count', 0)
                    })
                except Exception as e:
                    logger.warning(f"ML pred failed {symbol}: {e}")
                    status.update({
                        'ml_signal': 'NEUTRAL',
                        'ml_confidence': 0.0,
                        'ml_score': 0.0,
                        'ml_models': 0
                    })
            else:
                # ML nicht verfügbar oder keine Features
                status.update({
                    'ml_signal': 'NEUTRAL',
                    'ml_confidence': 0.0,
                    'ml_score': 0.0,
                    'ml_models': 0
                })
    
            # Balance info
            balances, total = self.get_account_info()
            status.update({
                "usdt_balance": round(usdt_bal, 2),
                "total_balance": round(total, 2),
                "daily_pnl": round(self.daily_pnl, 2),
                "capital_ratio": round(total / self.initial_balance, 3) if self.initial_balance > 0 else 1.0
            })
    
            # 🔍 FINAL STATUS DEBUG
            logger.info(f"📊 {symbol} Final Status: Action={signals.get('action', 'HOLD')}, "
                       f"Confidence={signals.get('confidence', 0):.1%}, "
                       f"ML={status.get('ml_signal', 'N/A')}, "
                       f"Volume_Ratio={indicators.get('volume_ratio', 0):.2f}")
    
            update_live_status(symbol, status)
    
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
            
            # Exception-Details für besseres Debugging
            import traceback
            logger.error(f"❌ {symbol} Traceback: {traceback.format_exc()}")


    def get_enhanced_status_data(self,symbol,ind,price,signals,in_pos):
        """Generate enhanced status data with more trading details"""
        sd={
            "price":price,
            "signal":signals.get('action','HOLD'),
            "confidence":signals.get('confidence',0.0),
            "reasons":signals.get('reasons',[]),
            "rsi":round(ind.get('rsi',50),1),
            "ema_fast":round(ind.get('ema_fast',0),4),
            "ema_slow":round(ind.get('ema_slow',0),4),
            "volume_status":"HIGH" if ind.get('volume_ratio',1)>2 else "NORMAL",
            "bb_position":self._get_bb_position(price,ind),
            "in_position":in_pos,
            "last_update":datetime.now().strftime("%H:%M:%S")
        }
        if in_pos and symbol in self.positions:
            pos=self.positions[symbol]
            entry=pos.get('timestamp',datetime.now())
            dur=(datetime.now()-entry).total_seconds()/60
            pv=pos.get('quantity',0)*pos.get('entry_price',0)
            cv=pos.get('quantity',0)*price
            upnl=cv-pv
            sd.update({
                "entry_price":pos.get('entry_price'),
                "entry_reason":pos.get('entry_reason','N/A'),
                "position_value":pv,
                "unrealized_pnl":upnl,
                "position_duration":f"{dur:.1f}m",
                "trailing_stop":self.trailing_stops.get(symbol)
            })
        try:
            t24=self.client.get_24hr_ticker(symbol=symbol)
            sd["price_change_24h"]=float(t24['priceChangePercent'])
        except: pass
        return sd

    def process_symbols_threaded(self):
        """Process symbols in parallel with threading"""
        try:
            with ThreadPoolExecutor(max_workers=self.max_symbol_threads) as executor:
                futures={executor.submit(self.process_symbol,s):s for s in self.symbols}
                for _ in as_completed(futures,timeout=30): pass
        except Exception as e:
            logger.error(f"❌ Error in threaded symbol processing: {e}")
            for s in self.symbols: self.process_symbol(s)

    def enhanced_place_sell_order(self,symbol,exit_reason):
        """Enhanced sell order with dashboard tracking"""
        try:
            if symbol not in self.positions: return False
            pos=self.positions[symbol]
            entry=pos['timestamp']; ep=pos['entry_price']
            bal,_=self.get_account_info()
            free=bal.get(symbol.replace('USDT',''),{}).get('free',0)
            if free<=0: 
                logger.warning(f"{symbol}: no balance"); return False
            price=self.get_current_price(symbol) or ep*1.01
            profit_pct=(price-ep)/ep
            profit_usdt=profit_pct*ep*pos['quantity']
            dur=(datetime.now()-entry).total_seconds()/60
            df=self.get_klines_data(symbol,50)
            mc={}
            if df is not None and len(df)>0:
                ind=self.calculate_indicators(df)
                mc={k:ind.get(k) for k in ('rsi','ema_fast','ema_slow','volume_ratio')}
                mc['current_price']=price
            
            # Hole verwendete Parameter für Trade-Tracking
            used_params = pos.get('used_params', self.get_trading_params_for_symbol(symbol))
            
            td={
                'timestamp':datetime.now(),'symbol':symbol,'entry_price':ep,
                'exit_price':price,'profit_pct':profit_pct,'profit_usdt':profit_usdt,
                'duration_minutes':dur,'exit_reason':exit_reason,
                'position_size_usdt':ep*pos['quantity'],
                'risk_pct':used_params.get('stop_loss_pct',0.02),
                'take_profit_target':used_params.get('take_profit_pct',0.03),
                'symbol_optimized': self.symbol_specific_mode and symbol in self.symbol_specific_params,
                'market_conditions':mc
            }
            self.performance_analyzer.record_trade(td)
            record_trade(td)
            update_balance(*self.get_account_info())
            try: update_performance_metrics(self.performance_analyzer.get_dashboard_metrics())
            except: pass
            
            # Erweiterte Logging mit Parameter-Info
            param_info = ""
            if self.symbol_specific_mode and symbol in self.symbol_specific_params:
                param_info = f" [Optimiert: TP={used_params.get('take_profit_pct',0)*100:.1f}%]"
            
            logger.info(f"{'✅' if profit_pct>0 else '❌'} {symbol}: {profit_pct*100:+.2f}%{param_info}")
            del self.positions[symbol]
            if symbol in self.trailing_stops: del self.trailing_stops[symbol]
            return True
        except Exception as e:
            logger.error(f"❌ Enhanced sell order failed {symbol}: {e}")
            return False

    # Alle anderen Methoden bleiben unverändert...
    def trading_loop(self):
        """Enhanced trading loop with performance reports"""
        logger.info("🚀 Enhanced trading loop started")
        lf=0; lp=0
        while self.running and not self.emergency_stop:
            try:
                now=time.time()
                if now-lp>=3:
                    if self.positions: self.check_positions_fast()
                    lp=now
                if now-lf>=12:
                    d=datetime.now()
                    if d.hour==0 and d.minute==0:
                        self.daily_pnl=0; self.daily_start_balance=self.current_balance; self.trades_today=0
                        logger.info("Daily reset")
                    if not self.check_capital_protection():
                        logger.warning("Paused by capital protection"); time.sleep(60); continue
                    if self.max_symbol_threads>1: self.process_symbols_threaded()
                    else:
                        for s in self.symbols:
                            if not self.running or self.emergency_stop: break
                            self.process_symbol(s); time.sleep(0.5)
                    self.performance_analyzer.print_detailed_report()
                    if self.positions:
                        logger.info(f"Active positions: {list(self.positions.keys())}")
                        logger.info(f"Daily PnL: {self.daily_pnl:.2f}")
                    wr=(self.session_stats['winning_trades']/max(self.session_stats['total_trades'],1))*100
                    logger.info(f"Session: {self.session_stats['total_trades']} trades, {wr:.1f}% win")
                    lf=now
                time.sleep(0.5)
            except KeyboardInterrupt:
                logger.info("Stopped by user"); break
            except Exception as e:
                logger.error(f"Error in loop: {e}"); time.sleep(5)

    def start(self):
        """Start the enhanced trading bot"""
        if not self.running:
            self.running=True
            try:
                bs,tot=self.get_account_info()
                logger.info(f"Balance: {tot:.2f}")
                update_balance(tot,tot)
                
                # Zeige Parameter-Modus
                if self.symbol_specific_mode:
                    logger.info(f"🎯 Symbol-spezifische Parameter aktiv für {len(self.symbol_specific_params)} Symbole")
                else:
                    logger.info("📝 Standard/Globale Parameter aktiv")
                    
            except Exception as e: logger.warning(f"init balance failed: {e}")
            import threading
            threading.Thread(target=self.trading_loop,daemon=True).start()
            logger.info("🚀 Bot started; dashboard at http://localhost:5000")
        else:
            logger.warning("Bot already running")

    def stop(self):
        """Stop the enhanced trading bot"""
        self.running=False
        try:
            if getattr(self.ml_system,'ml_enabled',False):
                self.ml_system.save_ml_state(); logger.info("ML state saved")
        except Exception as e:
            logger.warning(f"ML stop error: {e}")
        logger.info("🛑 Bot stopped")

    def check_positions_fast(self):
        """Fast position check for exits"""
        try:
            for s in list(self.positions):
                price=self.get_current_price(s)
                if price:
                    self.update_trailing_stop(s,price)
                    exit_ok,reason=self.should_exit_position(s,price,self.calculate_indicators(self.get_klines_data(s,20)))
                    if exit_ok: self.enhanced_place_sell_order(s,reason)
        except Exception as e:
            logger.error(f"Error in fast check: {e}")

    def get_current_price(self, symbol):
        try:
            return float(self.client.get_symbol_ticker(symbol=symbol)['price'])
        except Exception as e:
            logger.warning(f"price fetch failed {symbol}: {e}")
            return None

