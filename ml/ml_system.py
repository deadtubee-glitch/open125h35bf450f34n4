import logging
import numpy as np
import pandas as pd
import sqlite3
import os
import pickle
import joblib
from datetime import datetime, timedelta
from collections import deque
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
import warnings



warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# MACHINE LEARNING DAYTRADING SYSTEM
class MLDaytradingSystem:
    """Intelligentes ML-System für Daytrading mit persistentem Lernen"""
    
    def __init__(self, config):
        self.config = config
        self.ml_enabled = config.get('advanced', {}).get('use_machine_learning', False)
        
        # Datenbank für persistente Speicherung
        from pathlib import Path
        base = Path(__file__).parent.parent  # Projekt-Root
        self.db_path     = base / 'data' / 'ml_trading_data.db'
        self.models_path = base / 'ml' / 'models'
        os.makedirs(self.models_path, exist_ok=True)
        
        # ML-Parameter
        self.lookback_periods = [5, 10, 20, 50]  # Verschiedene Zeitfenster
        self.prediction_horizon = 5  # Vorhersage für nächste 5 Minuten
        self.min_samples_for_training = 500  # Mindestdaten für Training
        self.retrain_interval = 100  # Nach 100 neuen Samples neu trainieren
        
        # Feature Engineering Parameter
        self.feature_window = 100  # Letzte 100 Kerzen für Features
        
        # Modelle (Ensemble für bessere Vorhersagen)
        self.models = {
            'xgboost': None,
            'random_forest': None,
            'neural_network': None,
            'gradient_boost': None
        }
        
        # Scaler für Feature-Normalisierung
        self.scalers = {}
        
        # Performance Tracking
        self.model_performance = {}
        self.prediction_history = deque(maxlen=1000)
        
        # Live Data Buffer
        self.live_data_buffer = {}  # Symbol -> DataFrame
        self.sample_counter = 0
        
        # Initialisierung
        self.init_database()
        self.load_models()
        
        logging.info(f"ML Daytrading System {'aktiviert' if self.ml_enabled else 'deaktiviert'}")
        if self.ml_enabled:
            self.log_ml_status()

    def init_database(self):
        """Initialisiert SQLite Datenbank für ML-Daten"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabelle für Trainingsdaten
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timestamp DATETIME,
                    
                    -- Preis Features
                    open_price REAL, high_price REAL, low_price REAL, close_price REAL,
                    volume REAL, price_change_pct REAL,
                    
                    -- Technische Indikatoren
                    rsi_14 REAL, rsi_7 REAL, rsi_21 REAL,
                    ema_8 REAL, ema_21 REAL, ema_50 REAL,
                    sma_10 REAL, sma_20 REAL, sma_50 REAL,
                    macd REAL, macd_signal REAL, macd_histogram REAL,
                    bb_upper REAL, bb_middle REAL, bb_lower REAL,
                    bb_width REAL, bb_position REAL,
                    
                    -- Volume Features
                    volume_sma_20 REAL, volume_ratio REAL,
                    volume_weighted_price REAL,
                    
                    -- Volatilität Features
                    atr_14 REAL, volatility_10 REAL, volatility_20 REAL,
                    
                    -- Price Action Features
                    higher_high REAL, higher_low REAL, lower_high REAL, lower_low REAL,
                    support_level REAL, resistance_level REAL,
                    breakout_signal REAL, breakdown_signal REAL,
                    
                    -- Momentum Features
                    price_momentum_5 REAL, price_momentum_10 REAL,
                    volume_momentum_5 REAL, volume_momentum_10 REAL,
                    
                    -- Market Microstructure
                    bid_ask_spread REAL, order_imbalance REAL,
                    price_acceleration REAL,
                    
                    -- Target Variables (was passierte als nächstes)
                    price_change_1min REAL, price_change_5min REAL,
                    direction_1min INTEGER, direction_5min INTEGER,
                    
                    -- Trading Results (falls Trade stattfand)
                    trade_taken INTEGER, trade_result REAL, trade_duration REAL
                )
            ''')
            
            # Tabelle für Model Performance
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    symbol TEXT,
                    timestamp DATETIME,
                    accuracy REAL,
                    precision_buy REAL,
                    precision_sell REAL,
                    recall_buy REAL,
                    recall_sell REAL,
                    f1_score REAL,
                    samples_trained INTEGER
                )
            ''')
            
            # Tabelle für Live Predictions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timestamp DATETIME,
                    model_name TEXT,
                    prediction_class INTEGER,
                    prediction_probability REAL,
                    actual_result INTEGER,
                    correct_prediction INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            logging.info("ML-Datenbank initialisiert")
            
        except Exception as e:
            logging.error(f"Fehler bei DB-Initialisierung: {e}")

    def extract_features(self, df, symbol):
        """Extrahiert umfangreiche Features für ML"""
        try:
            if len(df) < self.feature_window:
                return None
                
            features = {}
            
            # Basic Price Features
            features['open'] = df['open'].iloc[-1]
            features['high'] = df['high'].iloc[-1]
            features['low'] = df['low'].iloc[-1]
            features['close'] = df['close'].iloc[-1]
            features['volume'] = df['volume'].iloc[-1]
            
            # Price Changes
            features['price_change_1'] = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) if len(df) > 1 else 0
            features['price_change_5'] = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) if len(df) > 5 else 0
            features['price_change_10'] = (df['close'].iloc[-1] / df['close'].iloc[-11] - 1) if len(df) > 10 else 0
            
            # RSI für verschiedene Perioden
            for period in [7, 14, 21]:
                if len(df) > period:
                    rsi = RSIIndicator(df['close'], window=period).rsi()
                    features[f'rsi_{period}'] = rsi.iloc[-1] if not rsi.isna().iloc[-1] else 50
                else:
                    features[f'rsi_{period}'] = 50
            
            # EMAs
            for period in [8, 21, 50]:
                if len(df) > period:
                    ema = EMAIndicator(df['close'], window=period).ema_indicator()
                    features[f'ema_{period}'] = ema.iloc[-1] if not ema.isna().iloc[-1] else df['close'].iloc[-1]
                    features[f'ema_{period}_slope'] = (ema.iloc[-1] / ema.iloc[-2] - 1) if len(ema) > 1 and not ema.isna().iloc[-2] else 0
                else:
                    features[f'ema_{period}'] = df['close'].iloc[-1]
                    features[f'ema_{period}_slope'] = 0
            
            # SMAs
            for period in [10, 20, 50]:
                if len(df) > period:
                    sma = df['close'].rolling(period).mean()
                    features[f'sma_{period}'] = sma.iloc[-1] if not sma.isna().iloc[-1] else df['close'].iloc[-1]
                    features[f'price_to_sma_{period}'] = df['close'].iloc[-1] / sma.iloc[-1] if not sma.isna().iloc[-1] else 1
                else:
                    features[f'sma_{period}'] = df['close'].iloc[-1]
                    features[f'price_to_sma_{period}'] = 1
            
            # MACD
            if len(df) > 26:
                macd_indicator = MACD(df['close'])
                macd = macd_indicator.macd()
                macd_signal = macd_indicator.macd_signal()
                features['macd'] = macd.iloc[-1] if not macd.isna().iloc[-1] else 0
                features['macd_signal'] = macd_signal.iloc[-1] if not macd_signal.isna().iloc[-1] else 0
                features['macd_histogram'] = (macd.iloc[-1] - macd_signal.iloc[-1]) if not (macd.isna().iloc[-1] or macd_signal.isna().iloc[-1]) else 0
            else:
                features['macd'] = 0
                features['macd_signal'] = 0
                features['macd_histogram'] = 0
            
            # Bollinger Bands
            if len(df) > 20:
                bb = BollingerBands(df['close'], window=20)
                bb_upper = bb.bollinger_hband()
                bb_lower = bb.bollinger_lband()
                bb_middle = bb.bollinger_mavg()
                
                features['bb_upper'] = bb_upper.iloc[-1] if not bb_upper.isna().iloc[-1] else df['close'].iloc[-1] * 1.02
                features['bb_lower'] = bb_lower.iloc[-1] if not bb_lower.isna().iloc[-1] else df['close'].iloc[-1] * 0.98
                features['bb_middle'] = bb_middle.iloc[-1] if not bb_middle.isna().iloc[-1] else df['close'].iloc[-1]
                features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
                features['bb_position'] = (df['close'].iloc[-1] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            else:
                current_price = df['close'].iloc[-1]
                features['bb_upper'] = current_price * 1.02
                features['bb_lower'] = current_price * 0.98
                features['bb_middle'] = current_price
                features['bb_width'] = 0.04
                features['bb_position'] = 0.5
            
            # Volume Features
            if len(df) > 20:
                volume_sma = df['volume'].rolling(20).mean()
                features['volume_sma_20'] = volume_sma.iloc[-1] if not volume_sma.isna().iloc[-1] else df['volume'].iloc[-1]
                features['volume_ratio'] = df['volume'].iloc[-1] / features['volume_sma_20'] if features['volume_sma_20'] > 0 else 1
            else:
                features['volume_sma_20'] = df['volume'].iloc[-1]
                features['volume_ratio'] = 1
            
            # Volatilität (ATR)
            if len(df) > 14:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(14).mean()
                features['atr_14'] = atr.iloc[-1] if not atr.isna().iloc[-1] else 0
            else:
                features['atr_14'] = 0
            
            # Price Action Patterns
            highs = df['high'].rolling(5).max()
            lows = df['low'].rolling(5).min()
            
            features['higher_high'] = 1 if len(df) > 5 and df['high'].iloc[-1] > highs.iloc[-6] else 0
            features['higher_low'] = 1 if len(df) > 5 and df['low'].iloc[-1] > lows.iloc[-6] else 0
            features['lower_high'] = 1 if len(df) > 5 and df['high'].iloc[-1] < highs.iloc[-6] else 0
            features['lower_low'] = 1 if len(df) > 5 and df['low'].iloc[-1] < lows.iloc[-6] else 0
            
            # Support/Resistance Levels
            recent_highs = df['high'].tail(20)
            recent_lows = df['low'].tail(20)
            features['resistance_level'] = recent_highs.max() if len(recent_highs) > 0 else df['high'].iloc[-1]
            features['support_level'] = recent_lows.min() if len(recent_lows) > 0 else df['low'].iloc[-1]
            features['distance_to_resistance'] = (features['resistance_level'] - df['close'].iloc[-1]) / df['close'].iloc[-1]
            features['distance_to_support'] = (df['close'].iloc[-1] - features['support_level']) / df['close'].iloc[-1]
            
            # Momentum Features
            for period in [5, 10]:
                if len(df) > period:
                    features[f'price_momentum_{period}'] = (df['close'].iloc[-1] / df['close'].iloc[-period-1] - 1)
                    features[f'volume_momentum_{period}'] = (df['volume'].iloc[-1] / df['volume'].iloc[-period-1] - 1) if df['volume'].iloc[-period-1] > 0 else 0
                else:
                    features[f'price_momentum_{period}'] = 0
                    features[f'volume_momentum_{period}'] = 0
            
            # Time-based Features
            current_time = datetime.now()
            features['hour'] = current_time.hour
            features['minute'] = current_time.minute
            features['day_of_week'] = current_time.weekday()
            features['is_market_open'] = 1 if 9 <= current_time.hour <= 16 else 0  # Haupthandelszeiten
            
            # Market Sentiment (vereinfacht)
            recent_returns = df['close'].pct_change().tail(10).dropna()
            features['sentiment_score'] = recent_returns.mean() if len(recent_returns) > 0 else 0
            features['sentiment_volatility'] = recent_returns.std() if len(recent_returns) > 1 else 0
            
            return features
            
        except Exception as e:
            logging.error(f"Fehler bei Feature-Extraktion für {symbol}: {e}")
            return None

    def calculate_targets(self, df, current_index):
        """Berechnet Target-Variablen (was passierte als nächstes)"""
        try:
            targets = {}
            
            if current_index >= len(df) - 5:
                return None  # Nicht genug zukünftige Daten
            
            current_price = df['close'].iloc[current_index]
            
            # Preisänderungen
            if current_index + 1 < len(df):
                future_price_1min = df['close'].iloc[current_index + 1]
                targets['price_change_1min'] = (future_price_1min / current_price - 1)
                targets['direction_1min'] = 1 if future_price_1min > current_price else 0
            
            if current_index + 5 < len(df):
                future_price_5min = df['close'].iloc[current_index + 5]
                targets['price_change_5min'] = (future_price_5min / current_price - 1)
                targets['direction_5min'] = 1 if future_price_5min > current_price else 0
                
                # Zusätzliche Targets für bessere Klassifikation
                change_5min = targets['price_change_5min']
                
                if change_5min > 0.005:  # >0.5% Anstieg
                    targets['class_5min'] = 2  # STRONG_BUY
                elif change_5min > 0.002:  # >0.2% Anstieg
                    targets['class_5min'] = 1  # BUY
                elif change_5min < -0.005:  # <-0.5% Rückgang
                    targets['class_5min'] = -2  # STRONG_SELL
                elif change_5min < -0.002:  # <-0.2% Rückgang
                    targets['class_5min'] = -1  # SELL
                else:
                    targets['class_5min'] = 0  # HOLD
                    
            return targets
            
        except Exception as e:
            logging.error(f"Fehler bei Target-Berechnung: {e}")
            return None

    def store_training_data(self, symbol, features, targets):
        """Speichert Trainingsdaten in Datenbank"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Daten für Insert vorbereiten
            data = [
                symbol,
                datetime.now().isoformat(),
                
                # Basic features
                features.get('open', 0), features.get('high', 0), features.get('low', 0), features.get('close', 0),
                features.get('volume', 0), features.get('price_change_1', 0),
                
                # RSI
                features.get('rsi_14', 50), features.get('rsi_7', 50), features.get('rsi_21', 50),
                
                # EMAs
                features.get('ema_8', 0), features.get('ema_21', 0), features.get('ema_50', 0),
                
                # SMAs
                features.get('sma_10', 0), features.get('sma_20', 0), features.get('sma_50', 0),
                
                # MACD
                features.get('macd', 0), features.get('macd_signal', 0), features.get('macd_histogram', 0),
                
                # Bollinger Bands
                features.get('bb_upper', 0), features.get('bb_middle', 0), features.get('bb_lower', 0),
                features.get('bb_width', 0), features.get('bb_position', 0),
                
                # Volume
                features.get('volume_sma_20', 0), features.get('volume_ratio', 1),
                features.get('close', 0),  # Volume weighted price placeholder
                
                # Volatility
                features.get('atr_14', 0), 0, 0,  # Volatility placeholders
                
                # Price Action
                features.get('higher_high', 0), features.get('higher_low', 0),
                features.get('lower_high', 0), features.get('lower_low', 0),
                features.get('support_level', 0), features.get('resistance_level', 0),
                0, 0,  # Breakout signals placeholder
                
                # Momentum
                features.get('price_momentum_5', 0), features.get('price_momentum_10', 0),
                features.get('volume_momentum_5', 0), features.get('volume_momentum_10', 0),
                
                # Market microstructure placeholders
                0, 0, 0,
                
                # Targets
                targets.get('price_change_1min', 0), targets.get('price_change_5min', 0),
                targets.get('direction_1min', 0), targets.get('direction_5min', 0),
                
                # Trade results (wird später aktualisiert)
                0, 0, 0
            ]
            
            cursor.execute('''
                INSERT INTO trading_features (
                    symbol, timestamp, open_price, high_price, low_price, close_price, volume, price_change_pct,
                    rsi_14, rsi_7, rsi_21, ema_8, ema_21, ema_50, sma_10, sma_20, sma_50,
                    macd, macd_signal, macd_histogram, bb_upper, bb_middle, bb_lower, bb_width, bb_position,
                    volume_sma_20, volume_ratio, volume_weighted_price, atr_14, volatility_10, volatility_20,
                    higher_high, higher_low, lower_high, lower_low, support_level, resistance_level,
                    breakout_signal, breakdown_signal, price_momentum_5, price_momentum_10,
                    volume_momentum_5, volume_momentum_10, bid_ask_spread, order_imbalance, price_acceleration,
                    price_change_1min, price_change_5min, direction_1min, direction_5min, trade_taken, trade_result, trade_duration
                ) VALUES (''' + ','.join(['?'] * len(data)) + ')', data)
            
            conn.commit()
            conn.close()
            
            self.sample_counter += 1
            
        except Exception as e:
            logging.error(f"Fehler beim Speichern der Trainingsdaten: {e}")

    def load_training_data(self, symbol=None, limit=5000):
        """Lädt Trainingsdaten aus Datenbank"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if symbol:
                query = """
                    SELECT * FROM trading_features 
                    WHERE symbol = ? AND direction_5min IS NOT NULL 
                    ORDER BY timestamp DESC LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=(symbol, limit))
            else:
                query = """
                    SELECT * FROM trading_features 
                    WHERE direction_5min IS NOT NULL 
                    ORDER BY timestamp DESC LIMIT ?
                """
                df = pd.read_sql_query(query, conn, params=(limit,))
            
            conn.close()
            return df
            
        except Exception as e:
            logging.error(f"Fehler beim Laden der Trainingsdaten: {e}")
            return pd.DataFrame()

    def prepare_ml_data(self, df):
        """Bereitet Daten für ML-Training vor"""
        if len(df) < self.min_samples_for_training:
            return None, None, None, None
        
        # Feature-Spalten definieren
        feature_columns = [
            'rsi_14', 'rsi_7', 'rsi_21', 'ema_8', 'ema_21', 'ema_50',
            'sma_10', 'sma_20', 'sma_50', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'volume_ratio', 'atr_14', 'higher_high', 'higher_low', 'lower_high', 'lower_low',
            'price_momentum_5', 'price_momentum_10', 'volume_momentum_5', 'volume_momentum_10'
        ]
        
        # Features und Targets extrahieren
        X = df[feature_columns].fillna(0)
        y = df['direction_5min'].astype(int)  # Binary: 0=down, 1=up
        
        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        return X_train, X_test, y_train, y_test

    def train_models(self, symbol):
        """Trainiert alle ML-Modelle"""
        try:
            # Trainingsdaten laden
            df = self.load_training_data(symbol)
            
            if len(df) < self.min_samples_for_training:
                logging.warning(f"Nicht genügend Daten für Training: {len(df)} < {self.min_samples_for_training}")
                return False
            
            # Daten vorbereiten
            X_train, X_test, y_train, y_test = self.prepare_ml_data(df)
            if X_train is None:
                return False
            
            # Feature Scaling
            if symbol not in self.scalers:
                self.scalers[symbol] = RobustScaler()
            
            X_train_scaled = self.scalers[symbol].fit_transform(X_train)
            X_test_scaled = self.scalers[symbol].transform(X_test)
            
            models_config = {
                'xgboost': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss'
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boost': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                'neural_network': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                )
            }
            
            # Modelle trainieren
            for model_name, model in models_config.items():
                try:
                    logging.info(f"Trainiere {model_name} für {symbol}...")
                    
                    # Training
                    if model_name == 'neural_network':
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Performance bewerten
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    
                    # Modell speichern
                    self.models[model_name] = model
                    model_path = os.path.join(self.models_path, f"{symbol}_{model_name}.joblib")
                    joblib.dump(model, model_path)
                    
                    # Scaler speichern
                    scaler_path = os.path.join(self.models_path, f"{symbol}_scaler.joblib")
                    joblib.dump(self.scalers[symbol], scaler_path)
                    
                    # Performance speichern
                    self.model_performance[f"{symbol}_{model_name}"] = {
                        'accuracy': accuracy,
                        'precision': report.get('1', {}).get('precision', 0),
                        'recall': report.get('1', {}).get('recall', 0),
                        'f1_score': report.get('1', {}).get('f1-score', 0),
                        'samples': len(X_train)
                    }
                    
                    # Performance in DB speichern
                    self.store_model_performance(model_name, symbol, accuracy, report, len(X_train))
                    
                    logging.info(f"{model_name} für {symbol}: Accuracy {accuracy:.3f}, F1-Score: {report.get('1', {}).get('f1-score', 0):.3f}")
                    
                except Exception as e:
                    logging.error(f"Fehler beim Training von {model_name}: {e}")
                    continue
            
            return True
            
        except Exception as e:
            logging.error(f"Fehler beim Model-Training: {e}")
            return False

    def store_model_performance(self, model_name, symbol, accuracy, report, samples):
        """Speichert Model-Performance in Datenbank"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance (
                    model_name, symbol, timestamp, accuracy, 
                    precision_buy, precision_sell, recall_buy, recall_sell,
                    f1_score, samples_trained
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_name, symbol, datetime.now().isoformat(),
                accuracy,
                report.get('1', {}).get('precision', 0),
                report.get('0', {}).get('precision', 0),
                report.get('1', {}).get('recall', 0),
                report.get('0', {}).get('recall', 0),
                report.get('macro avg', {}).get('f1-score', 0),
                samples
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Fehler beim Speichern der Performance: {e}")

    def load_models(self):
        """Lädt gespeicherte Modelle"""
        try:
            for model_file in os.listdir(self.models_path):
                if model_file.endswith('.joblib') and not 'scaler' in model_file:
                    symbol = model_file.split('_')[0]
                    model_name = model_file.replace(f'{symbol}_', '').replace('.joblib', '')
                    
                    model_path = os.path.join(self.models_path, model_file)
                    scaler_path = os.path.join(self.models_path, f"{symbol}_scaler.joblib")
                    
                    # Modell laden
                    model = joblib.load(model_path)
                    self.models[f"{symbol}_{model_name}"] = model
                    
                    # Scaler laden
                    if os.path.exists(scaler_path):
                        scaler = joblib.load(scaler_path)
                        self.scalers[symbol] = scaler
                    
                    logging.info(f"Modell geladen: {symbol}_{model_name}")
            
            logging.info(f"Insgesamt {len(self.models)} Modelle geladen")
            
        except Exception as e:
            logging.error(f"Fehler beim Laden der Modelle: {e}")

    def get_ml_prediction(self, symbol, features):
        """Generiert ML-basierte Trading-Vorhersagen"""
        if not self.ml_enabled or not features:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'predictions': {},
                'probabilities': {},    # ← NEU hinzugefügt
                'ensemble_score': 0.0,
                'model_count': 0       # ← NEU hinzugefügt - LÖSUNG!
            }
        
        try:
            # Feature-Array vorbereiten
            feature_names = [
                'rsi_14', 'rsi_7', 'rsi_21', 'ema_8', 'ema_21', 'ema_50',
                'sma_10', 'sma_20', 'sma_50', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
                'volume_ratio', 'atr_14', 'higher_high', 'higher_low', 'lower_high', 'lower_low',
                'price_momentum_5', 'price_momentum_10', 'volume_momentum_5', 'volume_momentum_10'
            ]
            
            X = np.array([[features.get(name, 0) for name in feature_names]])
            
            predictions = {}
            probabilities = {}
            
            # Vorhersagen von allen verfügbaren Modellen
            for model_key, model in self.models.items():
                if symbol in model_key:
                    try:
                        model_name = model_key.replace(f"{symbol}_", "")
                        
                        # Skalierung für Neural Network
                        if model_name == 'neural_network' and symbol in self.scalers:
                            X_scaled = self.scalers[symbol].transform(X)
                            pred = model.predict(X_scaled)[0]
                            prob = model.predict_proba(X_scaled)[0]
                        else:
                            pred = model.predict(X)[0]
                            prob = model.predict_proba(X)[0]
                        
                        predictions[model_name] = pred
                        probabilities[model_name] = prob[1]  # Wahrscheinlichkeit für Klasse 1 (UP)
                        
                    except Exception as e:
                        logging.warning(f"Vorhersage-Fehler für {model_key}: {e}")
                        continue
            
            if not predictions:
                return {
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'predictions': {},
                    'ensemble_score': 0.0
                }
            
            # Ensemble-Vorhersage (gewichteter Durchschnitt)
            model_weights = {
                'xgboost': 0.3,
                'random_forest': 0.25,
                'gradient_boost': 0.25,
                'neural_network': 0.2
            }
            
            weighted_score = 0
            total_weight = 0
            
            for model_name, prob in probabilities.items():
                weight = model_weights.get(model_name, 0.2)
                weighted_score += prob * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_score = weighted_score / total_weight
            else:
                ensemble_score = 0.5
            
            # Signal-Generierung basierend auf Ensemble-Score
            confidence = abs(ensemble_score - 0.5) * 2  # 0-1 Skala
            
            if ensemble_score > 0.65:
                signal = 'STRONG_BUY'
            elif ensemble_score > 0.55:
                signal = 'BUY'
            elif ensemble_score < 0.35:
                signal = 'STRONG_SELL'
            elif ensemble_score < 0.45:
                signal = 'SELL'
            else:
                signal = 'NEUTRAL'
            
            # Prediction für spätere Validierung speichern
            prediction_data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'ensemble_score': ensemble_score,
                'signal': signal,
                'confidence': confidence,
                'predictions': predictions,
                'probabilities': probabilities
            }
            
            self.prediction_history.append(prediction_data)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'predictions': predictions,
                'probabilities': probabilities,
                'ensemble_score': ensemble_score,
                'model_count': len(predictions)
            }
            
        except Exception as e:
            logging.error(f"ML-Vorhersage Fehler für {symbol}: {e}")
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'predictions': {},
                'probabilities': {},    # ← NEU hinzugefügt
                'ensemble_score': 0.0,
                'model_count': 0       # ← NEU hinzugefügt
            }

    def update_live_data(self, symbol, df):
        """Aktualisiert Live-Daten Buffer für kontinuierliches Lernen"""
        try:
            # Buffer aktualisieren
            self.live_data_buffer[symbol] = df.tail(self.feature_window).copy()
            
            # Features extrahieren und speichern (für späteres Training)
            if len(df) >= self.feature_window:
                features = self.extract_features(df, symbol)
                
                if features and len(df) > 10:  # Genug Daten für Targets
                    # Historical Targets berechnen (für Daten die alt genug sind)
                    if len(df) > self.prediction_horizon + 5:
                        historical_index = len(df) - self.prediction_horizon - 1
                        targets = self.calculate_targets(df, historical_index)
                        
                        if targets:
                            # Historical Features für denselben Punkt
                            historical_df = df.iloc[:historical_index+1]
                            historical_features = self.extract_features(historical_df, symbol)
                            
                            if historical_features:
                                self.store_training_data(symbol, historical_features, targets)
            
            # Prüfen ob Re-Training nötig ist
            if self.sample_counter % self.retrain_interval == 0 and self.sample_counter > 0:
                logging.info(f"Starte automatisches Re-Training nach {self.sample_counter} Samples")
                self.train_models(symbol)
                
        except Exception as e:
            logging.error(f"Fehler beim Live-Data Update für {symbol}: {e}")

    def validate_predictions(self, symbol, actual_price_change):
        """Validiert vergangene Vorhersagen und lernt daraus"""
        try:
            # Finde passende Vorhersagen aus der letzten Zeit
            current_time = datetime.now()
            
            for prediction in list(self.prediction_history):
                time_diff = (current_time - prediction['timestamp']).total_seconds() / 60
                
                # Vorhersagen die 5 Minuten alt sind validieren
                if 4.5 <= time_diff <= 5.5 and prediction['symbol'] == symbol:
                    
                    # Actual result bestimmen
                    actual_direction = 1 if actual_price_change > 0 else 0
                    
                    # Prediction war korrekt?
                    predicted_direction = 1 if prediction['ensemble_score'] > 0.5 else 0
                    correct = (predicted_direction == actual_direction)
                    
                    # In Datenbank speichern
                    self.store_prediction_result(prediction, actual_direction, correct)
                    
                    # Aus History entfernen
                    self.prediction_history.remove(prediction)
                    
                    logging.info(f"Vorhersage validiert: {symbol} - {'✅ Korrekt' if correct else '❌ Falsch'} "
                               f"(Score: {prediction['ensemble_score']:.3f}, Actual: {actual_price_change:.4f})")
                    
        except Exception as e:
            logging.error(f"Fehler bei Vorhersage-Validierung: {e}")

    def store_prediction_result(self, prediction, actual_result, correct):
        """Speichert Vorhersage-Ergebnisse für Performance-Tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for model_name, prob in prediction['probabilities'].items():
                cursor.execute('''
                    INSERT INTO live_predictions (
                        symbol, timestamp, model_name, prediction_class, 
                        prediction_probability, actual_result, correct_prediction
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction['symbol'],
                    prediction['timestamp'].isoformat(),
                    model_name,
                    1 if prob > 0.5 else 0,
                    prob,
                    actual_result,
                    1 if correct else 0
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Fehler beim Speichern des Vorhersage-Ergebnisses: {e}")

    def get_ml_performance_stats(self, symbol=None):
        """Holt Performance-Statistiken der ML-Modelle"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            if symbol:
                query = """
                    SELECT model_name, AVG(accuracy) as avg_accuracy, 
                           AVG(f1_score) as avg_f1, COUNT(*) as evaluations
                    FROM model_performance 
                    WHERE symbol = ? 
                    GROUP BY model_name
                    ORDER BY avg_accuracy DESC
                """
                df = pd.read_sql_query(query, conn, params=(symbol,))
            else:
                query = """
                    SELECT model_name, symbol, AVG(accuracy) as avg_accuracy,
                           AVG(f1_score) as avg_f1, COUNT(*) as evaluations
                    FROM model_performance 
                    GROUP BY model_name, symbol
                    ORDER BY avg_accuracy DESC
                """
                df = pd.read_sql_query(query, conn)
            
            conn.close()
            return df.to_dict('records')
            
        except Exception as e:
            logging.error(f"Fehler beim Abrufen der ML-Performance: {e}")
            return []

    def log_ml_status(self):
        """Loggt aktuellen ML-Status"""
        try:
            stats = self.get_ml_performance_stats()
            
            if stats:
                logging.info("=== ML-System Status ===")
                for stat in stats[:5]:  # Top 5
                    logging.info(f"{stat['model_name']} ({stat.get('symbol', 'ALL')}): "
                               f"Accuracy: {stat['avg_accuracy']:.3f}, "
                               f"F1-Score: {stat['avg_f1']:.3f}")
                logging.info("========================")
            else:
                logging.info("ML-System: Keine Trainingsdaten verfügbar - sammle Daten...")
                
        except Exception as e:
            logging.error(f"Fehler beim ML-Status Logging: {e}")

    def cleanup_old_data(self, days_to_keep=30):
        """Bereinigt alte Daten aus der Datenbank"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Alte Trainingsdaten löschen
            cursor.execute("DELETE FROM trading_features WHERE timestamp < ?", (cutoff_date,))
            deleted_features = cursor.rowcount
            
            # Alte Performance-Daten löschen
            cursor.execute("DELETE FROM model_performance WHERE timestamp < ?", (cutoff_date,))
            deleted_performance = cursor.rowcount
            
            # Alte Vorhersagen löschen
            cursor.execute("DELETE FROM live_predictions WHERE timestamp < ?", (cutoff_date,))
            deleted_predictions = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logging.info(f"Datenbank bereinigt: {deleted_features} Features, "
                        f"{deleted_performance} Performance-Einträge, "
                        f"{deleted_predictions} Vorhersagen gelöscht")
            
        except Exception as e:
            logging.error(f"Fehler bei Datenbank-Bereinigung: {e}")

    def save_ml_state(self):
        """Speichert aktuellen ML-Zustand beim Beenden"""
        try:
            # Alle Modelle und Scaler sind bereits persistent gespeichert
            
            # Prediction History als Backup speichern
            if self.prediction_history:
                backup_path = os.path.join(self.models_path, 'prediction_history_backup.pkl')
                with open(backup_path, 'wb') as f:
                    pickle.dump(list(self.prediction_history), f)
                
                logging.info(f"ML-Zustand gespeichert: {len(self.prediction_history)} aktive Vorhersagen")
            
            # Performance-Statistiken loggen
            self.log_ml_status()
            
        except Exception as e:
            logging.error(f"Fehler beim Speichern des ML-Zustands: {e}")

    def load_ml_state(self):
        """Lädt gespeicherten ML-Zustand beim Start"""
        try:
            backup_path = os.path.join(self.models_path, 'prediction_history_backup.pkl')
            
            if os.path.exists(backup_path):
                with open(backup_path, 'rb') as f:
                    history = pickle.load(f)
                    
                # Nur recent Vorhersagen laden (< 10 Minuten alt)
                current_time = datetime.now()
                for pred in history:
                    time_diff = (current_time - pred['timestamp']).total_seconds() / 60
                    if time_diff < 10:
                        self.prediction_history.append(pred)
                
                logging.info(f"ML-Zustand geladen: {len(self.prediction_history)} aktive Vorhersagen")
                
                # Backup-Datei löschen
                os.remove(backup_path)
            
        except Exception as e:
            logging.error(f"Fehler beim Laden des ML-Zustands: {e}")
