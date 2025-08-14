import logging
import numpy as np
import pandas as pd
import threading
import time
import math
from binance.client import Client
from binance.exceptions import BinanceAPIException
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AdvancedBinanceTrader:
    def __init__(self, config, optimized_params=None):
        self.config = config
        self.client = Client(config['api']['key'], config['api']['secret'], testnet=config['api'].get('testnet', True))
        logging.info("TESTNET MODUS" if config['api'].get('testnet', True) else "LIVE TRADING")

        self.trading_params = optimized_params or self.get_default_params()

        self.symbols = config['trading']['symbols']
        self.interval = config['trading']['interval']
        self.max_risk_per_trade = config['trading']['max_risk_per_trade']
        self.min_usdt_balance = config['trading']['min_usdt_balance']
        self.max_concurrent_positions = config['trading']['max_concurrent_positions']
        self.capital_protection_threshold = config['trading']['capital_protection_threshold']
        self.emergency_stop_loss = config['trading']['emergency_stop_loss']
        self.daily_loss_limit = config['trading']['daily_loss_limit']
        self.cooldown_period = config['trading']['cooldown_period']

        self.positions = {}
        self.trailing_stops = {}
        self.last_trade_time = {}
        self.initial_balance = 0
        self.daily_start_balance = 0
        self.current_balance = 0
        self.running = False
        self.emergency_stop = False

        self.trades_today = 0
        self.daily_pnl = 0
        self.session_stats = {'total_trades':0,'winning_trades':0,'total_pnl':0,'max_drawdown':0}

    def get_default_params(self):
        return {
            'rsi_period':14,'rsi_oversold':35,'rsi_overbought':70,
            'ema_fast':8,'ema_slow':21,'take_profit_pct':0.03,
            'stop_loss_pct':0.02,'trailing_sl_pct':0.015,
            'volume_threshold':1.5,'bb_period':20,'confidence_threshold':0.6
        }

    def check_capital_protection(self):
        if self.initial_balance==0: return True
        current_ratio=self.current_balance/self.initial_balance
        daily_loss_pct=(self.daily_start_balance-self.current_balance)/self.daily_start_balance if self.daily_start_balance>0 else 0

        if current_ratio<=(1-self.emergency_stop_loss):
            logging.critical("EMERGENCY STOP!")
            self.emergency_stop=True; return False
        if current_ratio<=self.capital_protection_threshold:
            logging.warning("CAPITAL PROTECTION engaged"); return False
        if daily_loss_pct>=self.daily_loss_limit:
            logging.warning("DAILY LOSS LIMIT reached"); return False
        return True

    def get_account_info(self):
        try:
            account=self.client.get_account()
            balances={}; total_usdt=0.0
            for b in account['balances']:
                val=float(b['free'])+float(b['locked'])
                if val>0:
                    balances[b['asset']]={'free':float(b['free']),'locked':float(b['locked']),'total':val}
                    if b['asset']=='USDT': total_usdt+=val
                    elif b['asset'] in ['BTC','ETH','BNB','ADA','SOL']:
                        try:
                            price=float(self.client.get_symbol_ticker(symbol=f"{b['asset']}USDT")['price'])
                            total_usdt+=val*price
                        except: pass
            if self.initial_balance==0:
                self.initial_balance=total_usdt; self.daily_start_balance=total_usdt
                logging.info(f"Initial Balance: {self.initial_balance:.2f}")
            self.current_balance=total_usdt
            return balances, total_usdt
        except Exception as e:
            logging.error(f"Account info error: {e}")
            return {}, 0.0

    def get_klines_data(self, symbol, limit=100):
        try:
            klines=self.client.get_klines(symbol=symbol,interval=self.interval,limit=limit)
            df=pd.DataFrame(klines,columns=['open_time','open','high','low','close','volume','close_time','quote_asset_volume','number_of_trades','taker_buy_base','taker_buy_quote','ignore'])
            for c in ['open','high','low','close','volume']: df[c]=df[c].astype(float)
            df['timestamp']=pd.to_datetime(df['open_time'],unit='ms')
            return df
        except Exception as e:
            logging.error(f"Klines data error {symbol}: {e}")
            return None

    def calculate_indicators(self, df):
        try:
            min_req=max(self.trading_params['rsi_period'],self.trading_params['ema_slow'],self.trading_params['bb_period'],20)
            if len(df)<min_req:
                logging.warning("Not enough data")
                return None

            indicators={}
            r=RSIIndicator(df['close'],window=self.trading_params['rsi_period']).rsi().iloc[-1]
            indicators['rsi']=50.0 if np.isnan(r) else r

            ef=EMAIndicator(df['close'],window=self.trading_params['ema_fast']).ema_indicator().iloc[-1]
            es=EMAIndicator(df['close'],window=self.trading_params['ema_slow']).ema_indicator().iloc[-1]
            indicators['ema_fast']=ef if not np.isnan(ef) else df['close'].iloc[-1]
            indicators['ema_slow']=es if not np.isnan(es) else df['close'].iloc[-1]

            m=MACD(df['close'])
            indicators['macd']=0.0 if np.isnan(m.macd().iloc[-1]) else m.macd().iloc[-1]
            indicators['macd_signal']=0.0 if np.isnan(m.macd_signal().iloc[-1]) else m.macd_signal().iloc[-1]

            bb=BollingerBands(df['close'],window=self.trading_params['bb_period'])
            up=bb.bollinger_hband().iloc[-1]; lo=bb.bollinger_lband().iloc[-1]; ma=bb.bollinger_mavg().iloc[-1]
            cp=df['close'].iloc[-1]
            indicators['bb_upper']=up if not np.isnan(up) else cp*1.02
            indicators['bb_lower']=lo if not np.isnan(lo) else cp*0.98
            indicators['bb_middle']=ma if not np.isnan(ma) else cp

            cv=df['volume'].iloc[-1]; sma=df['volume'].rolling(20).mean().iloc[-1]
            if pd.isna(sma) or sma==0: indicators['volume_ratio']=cv/(df['volume'].tail(5).median() or 1)
            else: indicators['volume_ratio']=cv/sma
            if np.isnan(indicators['volume_ratio']) or np.isinf(indicators['volume_ratio']):
                indicators['volume_ratio']=1.0

            return indicators
        except Exception as e:
            logging.error(f"Indicator error: {e}")
            return {
                'rsi':50.0,'ema_fast':df['close'].iloc[-1],'ema_slow':df['close'].iloc[-1],
                'macd':0.0,'macd_signal':0.0,'bb_upper':df['close'].iloc[-1]*1.02,
                'bb_lower':df['close'].iloc[-1]*0.98,'bb_middle':df['close'].iloc[-1],'volume_ratio':1.0
            }

    def generate_trading_signal(self, indicators, symbol, current_price, ml_features=None):
        """Generiert Trading-Signale mit ML-Integration"""
        signals = {
            'action': 'HOLD',
            'confidence': 0.0,
            'reasons': []
        }
        
        buy_signals = 0
        confidence = 0.0
        
        # TECHNISCHE INDIKATOREN (bestehende Logik)
        if indicators['rsi'] < self.trading_params['rsi_oversold']:
            buy_signals += 2
            confidence += 0.2
            signals['reasons'].append(f"RSI Oversold ({indicators['rsi']:.1f})")
        elif indicators['rsi'] > self.trading_params['rsi_overbought']:
            buy_signals -= 2
            signals['reasons'].append(f"RSI Overbought ({indicators['rsi']:.1f})")
        
        if indicators['ema_fast'] > indicators['ema_slow']:
            buy_signals += 1
            confidence += 0.15
            signals['reasons'].append("EMA Bullish")
        else:
            buy_signals -= 1
            signals['reasons'].append("EMA Bearish")
        
        if indicators['macd'] > indicators['macd_signal']:
            buy_signals += 1
            confidence += 0.1
            signals['reasons'].append("MACD Bullish")
        
        if current_price < indicators['bb_lower']:
            buy_signals += 1
            confidence += 0.1
            signals['reasons'].append("BB Lower Touch")
        elif current_price > indicators['bb_upper']:
            buy_signals -= 1
            signals['reasons'].append("BB Upper Touch")
        
        if indicators['volume_ratio'] > self.trading_params['volume_threshold']:
            confidence += 0.05
            signals['reasons'].append("High Volume")
        
        # ML-INTEGRATION
        if self.ml_system.ml_enabled and ml_features:
            ml_prediction = self.ml_system.get_ml_prediction(symbol, ml_features)
            
            ml_signal = ml_prediction['signal']
            ml_confidence = ml_prediction['confidence']
            ml_score = ml_prediction['ensemble_score']
            
            # ML-Signal Gewichtung
            if ml_signal == 'STRONG_BUY':
                buy_signals += 3
                confidence += 0.3 * ml_confidence
                signals['reasons'].append(f"ML Strong Buy ({ml_score:.2f})")
            elif ml_signal == 'BUY':
                buy_signals += 2
                confidence += 0.2 * ml_confidence
                signals['reasons'].append(f"ML Buy ({ml_score:.2f})")
            elif ml_signal == 'STRONG_SELL':
                buy_signals -= 3
                signals['reasons'].append(f"ML Strong Sell ({ml_score:.2f})")
            elif ml_signal == 'SELL':
                buy_signals -= 2
                signals['reasons'].append(f"ML Sell ({ml_score:.2f})")
            
            # ML-Konfidenz in Gesamtkonfidenz einbeziehen
            confidence = min(confidence + (ml_confidence * 0.1), 1.0)
            
            signals['ml_prediction'] = ml_prediction
        
        # FINALE ENTSCHEIDUNG
        confidence_threshold = self.trading_params.get('confidence_threshold', 0.6)
        
        if buy_signals >= 4 and confidence >= confidence_threshold:
            signals['action'] = 'BUY'
            signals['confidence'] = confidence
        elif buy_signals <= -3:
            signals['action'] = 'SELL'  # Für Exit-Signale
            signals['confidence'] = confidence
        
        return signals

    def should_exit_position(self, symbol, current_price, indicators):
        """Prüft Exit-Bedingungen mit verbessertem Stop-Loss"""
        if symbol not in self.positions:
            return False, ""
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        profit_pct = (current_price - entry_price) / entry_price
        
        # SOFORTIGER NOTFALL-STOP bei großen Verlusten
        if profit_pct <= -0.03:  # 3% harter Stop
            return True, f"NOTFALL-STOP ({profit_pct*100:.2f}%)"
        
        # Take Profit (etwas früher)
        if profit_pct >= (self.trading_params['take_profit_pct'] * 0.8):  # 80% vom Ziel
            return True, f"Take Profit ({profit_pct*100:.2f}%)"
        
        # Dynamischer Stop Loss basierend auf Volatilität
        base_stop = self.trading_params['stop_loss_pct']
        
        # RSI-basierter Stop (früher aussteigen wenn Momentum dreht)
        if indicators['rsi'] > 65 and profit_pct > 0:
            return True, f"RSI Momentum Exit ({profit_pct*100:.2f}%)"
        
        # Enger Stop bei negativem MACD
        if indicators['macd'] < indicators['macd_signal'] and profit_pct < 0:
            tighter_stop = base_stop * 0.7  # 30% engerer Stop
            if profit_pct <= -tighter_stop:
                return True, f"MACD Stop ({profit_pct*100:.2f}%)"
        
        # Standard Stop Loss
        if profit_pct <= -base_stop:
            return True, f"Stop Loss ({profit_pct*100:.2f}%)"
        
        # Trailing Stop mit verbesserter Logik
        if symbol in self.trailing_stops:
            trailing_stop_price = self.trailing_stops[symbol]
            
            # Aktiviere Trailing Stop erst bei 0.5% Gewinn
            if profit_pct > 0.005:  # 0.5% Gewinn
                if current_price <= trailing_stop_price:
                    return True, f"Trailing Stop ({profit_pct*100:.2f}%)"
        
        # Zeit-basierter Stop (nach 30 Minuten bei Verlust)
        position_age = (datetime.now() - position['timestamp']).total_seconds() / 60  # Minuten
        if position_age > 30 and profit_pct < -0.005:  # Nach 30min bei -0.5%
            return True, f"Zeit-Stop ({profit_pct*100:.2f}%, {position_age:.0f}min)"
        
        return False, ""


    def update_trailing_stop(self, symbol, current_price):
        """Aktualisiert Trailing Stop mit verbesserter Logik"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        entry_price = position['entry_price']
        profit_pct = (current_price - entry_price) / entry_price
        
        # Dynamischer Trailing Stop basierend auf Gewinn
        if profit_pct > 0.02:  # Bei >2% Gewinn: engerer Trailing Stop
            trailing_distance = 0.008  # 0.8%
        elif profit_pct > 0.01:  # Bei >1% Gewinn: normaler Trailing Stop
            trailing_distance = 0.012  # 1.2%
        else:
            trailing_distance = self.trading_params['trailing_sl_pct']  # Standard
        
        new_trailing_stop = current_price * (1 - trailing_distance)
        
        if symbol not in self.trailing_stops:
            # Initial nur setzen wenn im Gewinn
            if profit_pct > 0.005:  # 0.5% Mindestgewinn
                self.trailing_stops[symbol] = new_trailing_stop
            return
        
        current_trailing_stop = self.trailing_stops[symbol]
        
        # Nur nach oben anpassen (bei steigenden Preisen)
        if new_trailing_stop > current_trailing_stop:
            self.trailing_stops[symbol] = new_trailing_stop
            logging.info(f"{symbol}: Trailing Stop aktualisiert: {new_trailing_stop:.4f} "
                        f"(Gewinn: {profit_pct*100:.2f}%, Abstand: {trailing_distance*100:.1f}%)")

    def get_symbol_info(self, symbol):
            """Holt Symbol-Informationen für LOT_SIZE Filter"""
            try:
                info = self.client.get_symbol_info(symbol)
                filters = {f['filterType']: f for f in info['filters']}
                
                lot_size = filters.get('LOT_SIZE', {})
                min_qty = float(lot_size.get('minQty', '0.00001'))
                step_size = float(lot_size.get('stepSize', '0.00001'))
                
                return {
                    'min_qty': min_qty,
                    'step_size': step_size,
                    'base_precision': info['baseAssetPrecision'],
                    'quote_precision': info['quotePrecision']
                }
            except Exception as e:
                logging.error(f"Fehler beim Abrufen Symbol-Info für {symbol}: {e}")
                return {'min_qty': 0.001, 'step_size': 0.001, 'base_precision': 6, 'quote_precision': 2}

    def round_quantity(self, quantity, step_size):
        """Rundet Menge auf gültigen Step-Size"""
        return math.floor(quantity / step_size) * step_size

    def place_buy_order(self, symbol, current_price, signals, balance):
        """Platziert Kauforder mit LOT_SIZE Korrektur"""
        try:
            # Kapitalschutz prüfen
            if not self.check_capital_protection():
                return False
            
            # Cooldown prüfen
            if symbol in self.last_trade_time:
                time_since_last = (datetime.now() - self.last_trade_time[symbol]).total_seconds()
                if time_since_last < self.cooldown_period:
                    return False
            
            # Maximale Positionen prüfen
            if len(self.positions) >= self.max_concurrent_positions:
                return False
            
            # Symbol-Info abrufen für LOT_SIZE
            symbol_info = self.get_symbol_info(symbol)
            min_qty = symbol_info['min_qty']
            step_size = symbol_info['step_size']
            
            asset = symbol.replace('USDT', '')
            
            # Position Size berechnen
            risk_amount = balance * self.max_risk_per_trade
            raw_quantity = (risk_amount / current_price) * 0.99  # 1% Reserve für Gebühren
            
            # Auf gültigen LOT_SIZE runden
            quantity = self.round_quantity(raw_quantity, step_size)
            
            # Mindestmenge prüfen
            if quantity < min_qty:
                logging.warning(f"{symbol}: Menge {quantity:.6f} unter Minimum {min_qty:.6f}")
                # Versuche Mindestmenge
                quantity = min_qty
                
            # Prüfen ob genug Balance für Mindestmenge
            required_usdt = quantity * current_price * 1.01  # +1% für Gebühren
            if required_usdt > balance:
                logging.warning(f"{symbol}: Nicht genug Balance für Mindestorder. Benötigt: {required_usdt:.2f}, Verfügbar: {balance:.2f}")
                return False
            
            # Mindest-Order-Wert prüfen (Binance Minimum ~10 USDT)
            order_value = quantity * current_price
            if order_value < 10:
                logging.warning(f"{symbol}: Order-Wert {order_value:.2f} unter Minimum 10 USDT")
                return False
            
            logging.info(f"{symbol}: Kaufsignal - Confidence: {signals['confidence']:.2f}")
            logging.info(f"Signale: {', '.join(signals['reasons'])}")
            logging.info(f"Order: {quantity:.6f} {asset} für {order_value:.2f} USDT")
            
            # Market Buy Order
            order = self.client.order_market_buy(
                symbol=symbol,
                quantity=f"{quantity:.{symbol_info['base_precision']}f}"
            )
            
            if order and order.get('status') == 'FILLED':
                # Durchschnittspreis berechnen
                fills = order.get('fills', [])
                if fills:
                    total_qty = sum(float(fill['qty']) for fill in fills)
                    total_cost = sum(float(fill['qty']) * float(fill['price']) for fill in fills)
                    avg_price = total_cost / total_qty if total_qty > 0 else current_price
                else:
                    avg_price = current_price
                
                executed_qty = float(order.get('executedQty', quantity))
                
                # Position speichern
                self.positions[symbol] = {
                    'entry_price': avg_price,
                    'quantity': executed_qty,
                    'order_id': order['orderId'],
                    'timestamp': datetime.now(),
                    'confidence': signals['confidence'],
                    'reasons': signals['reasons']
                }
                
                # Trailing Stop initialisieren
                initial_stop = avg_price * (1 - self.trading_params['trailing_sl_pct'])
                self.trailing_stops[symbol] = initial_stop
                
                # Cooldown setzen
                self.last_trade_time[symbol] = datetime.now()
                
                logging.info(f"✅ KAUF ERFOLGREICH {symbol}: Preis: {avg_price:.4f}, Menge: {executed_qty:.6f}")
                logging.info(f"Initial Trailing Stop: {initial_stop:.4f}")
                
                return True
                
        except BinanceAPIException as e:
            logging.error(f"❌ Kauforder fehlgeschlagen {symbol}: {e}")
            if "LOT_SIZE" in str(e):
                logging.error(f"LOT_SIZE Problem - Min: {symbol_info.get('min_qty', 'unknown')}, Step: {symbol_info.get('step_size', 'unknown')}")
        except Exception as e:
            logging.error(f"❌ Unerwarteter Fehler beim Kauf {symbol}: {e}")
        
        return False

    def place_sell_order(self, symbol, exit_reason):
        """Platziert Verkaufsorder mit Performance-Tracking"""
        try:
            if symbol not in self.positions:
                return False
            
            position = self.positions[symbol]
            asset = symbol.replace('USDT', '')
            
            # Aktuelle Balance prüfen
            balances, _ = self.get_account_info()
            balance = balances.get(asset, {}).get('free', 0)
            
            if balance <= 0:
                logging.warning(f"{symbol}: Keine {asset} Balance zum Verkaufen")
                return False
            
            # Market Sell Order
            order = self.client.order_market_sell(
                symbol=symbol,
                quantity=balance
            )
            
            if order and order.get('status') == 'FILLED':
                # Exit-Preis berechnen
                fills = order.get('fills', [])
                if fills:
                    exit_price = float(fills[0]['price'])
                else:
                    # Fallback: aktueller Marktpreis
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    exit_price = float(ticker['price'])
                
                # Performance berechnen
                entry_price = position['entry_price']
                profit_pct = (exit_price - entry_price) / entry_price
                profit_usdt = profit_pct * entry_price * position['quantity']
                
                # Trade-Statistiken aktualisieren
                self.session_stats['total_trades'] += 1
                self.session_stats['total_pnl'] += profit_usdt
                self.daily_pnl += profit_usdt
                
                if profit_pct > 0:
                    self.session_stats['winning_trades'] += 1
                
                # Drawdown berechnen
                if self.current_balance > 0:
                    drawdown = (self.initial_balance - self.current_balance) / self.initial_balance
                    self.session_stats['max_drawdown'] = max(self.session_stats['max_drawdown'], drawdown)
                
                # Logging
                if profit_pct > 0:
                    logging.info(f"GEWINN {symbol}: +{profit_pct*100:.2f}% ({profit_usdt:.2f} USDT) - {exit_reason}")
                else:
                    logging.info(f"VERLUST {symbol}: {profit_pct*100:.2f}% ({profit_usdt:.2f} USDT) - {exit_reason}")
                
                # Position schließen
                del self.positions[symbol]
                if symbol in self.trailing_stops:
                    del self.trailing_stops[symbol]
                
                return True
                
        except Exception as e:
            logging.error(f"Verkauf fehlgeschlagen {symbol}: {e}")
        
        return False

    def process_symbol(self, symbol):
        """Verarbeitet ein Symbol mit ML-Integration"""
        try:
            if self.emergency_stop:
                return
            
            df = self.get_klines_data(symbol, self.ml_system.feature_window if self.ml_system.ml_enabled else 100)
            if df is None or len(df) < 50:
                return
            
            # ML Live-Data Update
            if self.ml_system.ml_enabled:
                self.ml_system.update_live_data(symbol, df)
                
                # Vorhersagen validieren
                if len(df) > 1:
                    recent_change = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1)
                    self.ml_system.validate_predictions(symbol, recent_change)
            
            indicators = self.calculate_indicators(df)
            if indicators is None:
                return
            
            current_price = df['close'].iloc[-1]
            balances, total_value = self.get_account_info()
            usdt_balance = balances.get('USDT', {}).get('free', 0)
            
            in_position = symbol in self.positions
            
            if not in_position:
                # ML-Features extrahieren
                ml_features = None
                if self.ml_system.ml_enabled:
                    ml_features = self.ml_system.extract_features(df, symbol)
                
                signals = self.generate_trading_signal(indicators, symbol, current_price, ml_features)
                
                if (signals['action'] == 'BUY' and 
                    usdt_balance > self.min_usdt_balance and 
                    self.check_capital_protection()):
                    self.place_buy_order(symbol, current_price, signals, usdt_balance)
            else:
                self.update_trailing_stop(symbol, current_price)
                should_exit, exit_reason = self.should_exit_position(symbol, current_price, indicators)
                if should_exit:
                    self.place_sell_order(symbol, exit_reason)
            
            # Status Update mit ML-Info
            ml_info = {}
            if self.ml_system.ml_enabled and ml_features:
                ml_prediction = self.ml_system.get_ml_prediction(symbol, ml_features)
                ml_info = {
                    'ml_signal': ml_prediction['signal'],
                    'ml_confidence': round(ml_prediction['confidence'], 3),
                    'ml_score': round(ml_prediction['ensemble_score'], 3),
                    'ml_models': ml_prediction.get('model_count', 0)  # ← Sicher mit .get()

                }
            
            live_status[symbol] = {
                "price": current_price,
                "rsi": round(indicators['rsi'], 2) if not np.isnan(indicators['rsi']) else 0,
                "ema_fast": round(indicators['ema_fast'], 4),
                "ema_slow": round(indicators['ema_slow'], 4),
                "in_position": in_position,
                "entry_price": self.positions.get(symbol, {}).get('entry_price'),
                "trailing_stop": self.trailing_stops.get(symbol),
                "usdt_balance": round(usdt_balance, 2),
                "total_balance": round(total_value, 2),
                "daily_pnl": round(self.daily_pnl, 2),
                "capital_ratio": round(total_value / self.initial_balance, 3) if self.initial_balance > 0 else 1.0,
                "last_update": datetime.now().strftime("%H:%M:%S"),
                **ml_info
            }
            
        except Exception as e:
            logging.error(f"Fehler bei der Verarbeitung von {symbol}: {e}")

    def trading_loop(self):
        """Haupt-Trading Loop mit schnellerer Position-Überwachung"""
        logging.info("Trading Loop gestartet mit verbessertem Stop-Loss...")
        
        # Verschiedene Timer für verschiedene Aufgaben
        last_full_scan = 0
        last_position_check = 0
        
        while self.running and not self.emergency_stop:
            try:
                current_time = time.time()
                
                # SCHNELLE POSITIONS-ÜBERWACHUNG (alle 5 Sekunden)
                if current_time - last_position_check >= 5:
                    if self.positions:
                        self.check_positions_fast()
                    last_position_check = current_time
                
                # VOLLSTÄNDIGER SCAN (alle 15 Sekunden)
                if current_time - last_full_scan >= 15:
                    # Täglicher Reset-Check
                    now = datetime.now()
                    if now.hour == 0 and now.minute == 0:
                        self.daily_pnl = 0
                        self.daily_start_balance = self.current_balance
                        self.trades_today = 0
                        logging.info("Täglicher Reset durchgeführt")
                    
                    # Kapitalschutz prüfen
                    if not self.check_capital_protection():
                        logging.warning("Trading pausiert aufgrund Kapitalschutz")
                        time.sleep(60)  # 1 Minute Pause
                        continue
                    
                    # Symbole verarbeiten
                    for symbol in self.symbols:
                        if not self.running or self.emergency_stop:
                            break
                        self.process_symbol(symbol)
                        time.sleep(1)  # Kurze Pause zwischen Symbolen
                    
                    # Status-Update
                    if self.positions:
                        logging.info(f"Aktive Positionen: {list(self.positions.keys())}")
                        logging.info(f"Tages-PnL: {self.daily_pnl:.2f} USDT")
                    
                    # Performance-Update
                    win_rate = (self.session_stats['winning_trades'] / max(self.session_stats['total_trades'], 1)) * 100
                    logging.info(f"Session Stats: {self.session_stats['total_trades']} Trades, {win_rate:.1f}% Win Rate")
                    
                    last_full_scan = current_time
                
                # Kurze Pause zwischen Checks
                time.sleep(1)
                    
            except KeyboardInterrupt:
                logging.info("Trading gestoppt durch Benutzer")
                break
            except Exception as e:
                logging.error(f"Fehler im Trading Loop: {e}")
                time.sleep(10)  # 10 Sekunden Pause bei Fehlern
                
    def check_positions_fast(self):
        """Schnelle Überprüfung aller Positionen (nur Exit-Signale)"""
        try:
            for symbol in list(self.positions.keys()):
                # Aktueller Preis
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                
                position = self.positions[symbol]
                entry_price = position['entry_price']
                profit_pct = (current_price - entry_price) / entry_price
                
                # KRITISCHE STOPS (ohne Indikatoren)
                should_exit = False
                exit_reason = ""
                
                # Notfall-Stop
                if profit_pct <= -0.03:  # 3%
                    should_exit = True
                    exit_reason = f"NOTFALL-STOP ({profit_pct*100:.2f}%)"
                
                # Trailing Stop Check
                elif symbol in self.trailing_stops and current_price <= self.trailing_stops[symbol]:
                    should_exit = True
                    exit_reason = f"Trailing Stop ({profit_pct*100:.2f}%)"
                
                # Standard Stop
                elif profit_pct <= -self.trading_params['stop_loss_pct']:
                    should_exit = True
                    exit_reason = f"Stop Loss ({profit_pct*100:.2f}%)"
                
                # Take Profit
                elif profit_pct >= self.trading_params['take_profit_pct']:
                    should_exit = True
                    exit_reason = f"Take Profit ({profit_pct*100:.2f}%)"
                
                if should_exit:
                    logging.warning(f"🚨 SCHNELLER EXIT {symbol}: {exit_reason}")
                    self.place_sell_order(symbol, exit_reason)
                else:
                    # Trailing Stop aktualisieren
                    self.update_trailing_stop(symbol, current_price)
                    
        except Exception as e:
            logging.error(f"Fehler bei schneller Positions-Check: {e}")

    def start_flask_server(self):
        """Startet Flask Server für Monitoring"""

    def start(self):
        """Startet den Trading Bot"""
        if not self.running:
            self.running = True
            
            # Flask Server starten
            self.start_flask_server()
            
            # Initial Balance setzen
            balances, total_value = self.get_account_info()
            logging.info(f"Account Balance: {total_value:.2f} USDT")
            
            # Trading Loop starten
            trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
            trading_thread.start()
            
            logging.info("Trading Bot gestartet mit optimierten Parametern!")
            logging.info(f"Parameter: {self.trading_params}")
            logging.info(f"Kapitalschutz: {self.capital_protection_threshold*100}%")
            logging.info(f"Emergency Stop: {self.emergency_stop_loss*100}%")
            
            return trading_thread
        else:
            logging.warning("Bot läuft bereits!")

    def stop(self, force_exit_all=True):
        """Stoppt den Trading Bot mit automatischem Position-Exit"""
        logging.info("🛑 Bot shutdown initiated...")
        self.running = False
        
        # Automatischer Verkauf aller Positionen
        if force_exit_all and hasattr(self, 'positions') and self.positions:
            logging.info(f"🔄 Auto-exiting {len(self.positions)} open positions...")
            self.emergency_exit_all_positions()
        
        # ML-System safe stoppen (Ihre bestehende Logik)
        try:
            if (hasattr(self, 'ml_system') and 
                self.ml_system and 
                hasattr(self.ml_system, 'ml_enabled') and 
                self.ml_system.ml_enabled):
                self.ml_system.save_ml_state()
                logging.info("💾 ML-Zustand gespeichert")
        except Exception as e:
            logging.warning(f"⚠️ ML-System Stop Fehler: {e}")
        
        logging.info("🛑 Trading Bot gestoppt")

    def graceful_shutdown(self, timeout_seconds=30):
        """Graceful shutdown mit natürlichen Exits"""
        import time
        
        logging.info(f"🔄 Graceful shutdown started (timeout: {timeout_seconds}s)")
        
        start_time = time.time()
        self.running = False
        
        # Warten auf natürliche Exits
        while (hasattr(self, 'positions') and self.positions and 
               (time.time() - start_time) < timeout_seconds):
            logging.info(f"⏳ Waiting for positions to close naturally... ({len(self.positions)} remaining)")
            time.sleep(2)
            
            # Prüfe natürliche Exits
            for symbol in list(self.positions.keys()):
                try:
                    price = self.get_current_price(symbol)
                    if price and hasattr(self, 'should_exit_position'):
                        # Falls should_exit_position verfügbar ist
                        indicators = self.calculate_indicators(self.get_klines_data(symbol, 20))
                        if indicators:
                            exit_ok, reason = self.should_exit_position(symbol, price, indicators)
                            if exit_ok:
                                self.force_sell_position(symbol, f"GRACEFUL_EXIT_{reason}")
                except Exception as e:
                    logging.warning(f"⚠️ Graceful exit check failed {symbol}: {e}")
        
        # Timeout: Force exit
        if hasattr(self, 'positions') and self.positions:
            logging.warning(f"⏰ Timeout reached, force exiting {len(self.positions)} positions")
            self.emergency_exit_all_positions()
        
        # ML-System safe stoppen
        try:
            if (hasattr(self, 'ml_system') and 
                self.ml_system and 
                hasattr(self.ml_system, 'ml_enabled') and 
                self.ml_system.ml_enabled):
                self.ml_system.save_ml_state()
                logging.info("💾 ML-Zustand gespeichert")
        except Exception as e:
            logging.warning(f"⚠️ ML-System Stop Fehler: {e}")
        
        logging.info("✅ Graceful shutdown completed")
    
    def emergency_exit_all_positions(self):
        """Verkauft alle offenen Positionen sofort"""
        logging.info("🚨 EMERGENCY EXIT: Selling all positions...")
        
        if not hasattr(self, 'positions') or not self.positions:
            logging.info("ℹ️ No positions to exit")
            return []
        
        exit_results = []
        
        for symbol in list(self.positions.keys()):
            try:
                # Aktuelle Position-Info
                position = self.positions[symbol]
                current_price = self.get_current_price(symbol)
                
                if current_price:
                    # Berechne aktuellen P&L
                    entry_price = position['entry_price']
                    profit_pct = (current_price - entry_price) / entry_price
                    
                    logging.info(f"🔄 Emergency exit {symbol}: "
                               f"Entry={entry_price:.4f}, "
                               f"Current={current_price:.4f}, "
                               f"P&L={profit_pct*100:+.2f}%")
                    
                    # Verkauf ausführen
                    success = self.force_sell_position(symbol, "EMERGENCY_SHUTDOWN")
                    
                    exit_results.append({
                        'symbol': symbol,
                        'success': success,
                        'profit_pct': profit_pct,
                        'reason': 'EMERGENCY_SHUTDOWN'
                    })
                    
                else:
                    logging.error(f"❌ Could not get price for {symbol}")
                    
            except Exception as e:
                logging.error(f"❌ Emergency exit failed for {symbol}: {e}")
                exit_results.append({
                    'symbol': symbol,
                    'success': False,
                    'error': str(e)
                })
        
        # Exit-Summary
        successful_exits = sum(1 for r in exit_results if r.get('success', False))
        total_positions = len(exit_results)
        
        logging.info(f"🏁 Emergency exit completed: "
                   f"{successful_exits}/{total_positions} positions closed")
        
        # Detaillierte Ausgabe
        for result in exit_results:
            if result.get('success'):
                logging.info(f"✅ {result['symbol']}: {result['profit_pct']*100:+.2f}%")
            else:
                logging.error(f"❌ {result['symbol']}: {result.get('error', 'Unknown error')}")
        
        return exit_results
    
    def force_sell_position(self, symbol, exit_reason):
        """Forciert den Verkauf einer Position"""
        try:
            if symbol not in self.positions:
                logging.warning(f"⚠️ {symbol}: Position not found")
                return False
            
            position = self.positions[symbol]
            
            # Balance prüfen
            balances, _ = self.get_account_info()
            base_asset = symbol.replace('USDT', '')
            available_qty = balances.get(base_asset, {}).get('free', 0)
            
            if available_qty <= 0:
                logging.warning(f"⚠️ {symbol}: No balance to sell ({available_qty})")
                # Position trotzdem aus dem Tracking entfernen
                del self.positions[symbol]
                if hasattr(self, 'trailing_stops') and symbol in self.trailing_stops:
                    del self.trailing_stops[symbol]
                return False
            
            # Symbol-Info für Precision
            symbol_info = self.get_symbol_info(symbol)
            base_precision = symbol_info.get('base_precision', 6)
            
            # Verkaufsmenge anpassen
            sell_quantity = min(available_qty, position['quantity'])
            sell_quantity = round(sell_quantity, base_precision)
            
            logging.info(f"🔄 Force selling {symbol}: {sell_quantity:.{base_precision}f}")
            
            # Market Sell Order
            order = self.client.order_market_sell(
                symbol=symbol,
                quantity=f"{sell_quantity:.{base_precision}f}"
            )
            
            if order and order.get('status') == 'FILLED':
                # Exit-Preis berechnen
                fills = order.get('fills', [])
                if fills:
                    total_qty = sum(float(fill['qty']) for fill in fills)
                    total_value = sum(float(fill['qty']) * float(fill['price']) for fill in fills)
                    avg_exit_price = total_value / total_qty if total_qty > 0 else 0
                else:
                    avg_exit_price = float(order.get('price', 0))
                
                # Performance tracking (falls verfügbar)
                if hasattr(self, 'enhanced_place_sell_order'):
                    self.enhanced_place_sell_order(symbol, exit_reason)
                else:
                    # Fallback: Basic position cleanup
                    del self.positions[symbol]
                    if hasattr(self, 'trailing_stops') and symbol in self.trailing_stops:
                        del self.trailing_stops[symbol]
                
                logging.info(f"✅ {symbol} force sold at {avg_exit_price:.4f}")
                return True
            else:
                logging.error(f"❌ {symbol}: Force sell order failed")
                return False
                
        except Exception as e:
            logging.error(f"❌ Force sell failed {symbol}: {e}")
            
            # Fallback: Position aus Tracking entfernen
            if hasattr(self, 'positions') and symbol in self.positions:
                del self.positions[symbol]
            if hasattr(self, 'trailing_stops') and symbol in self.trailing_stops:
                del self.trailing_stops[symbol]
            
            return False
    
    def get_current_price(self, symbol):
        """Holt aktuellen Preis (falls nicht vorhanden)"""
        try:
            if hasattr(self, 'client'):
                return float(self.client.get_symbol_ticker(symbol=symbol)['price'])
            else:
                logging.error("❌ No client available for price fetch")
                return None
        except Exception as e:
            logging.warning(f"⚠️ Price fetch failed {symbol}: {e}")
            return None

