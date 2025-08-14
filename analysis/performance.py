import logging
from collections import defaultdict
import statistics
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Detaillierte Performance-Analyse mit allen gewünschten Metriken"""
    
    def __init__(self):
        self.trades = []
        self.market_conditions = []
        self.position_types = {'LONG': 0, 'SHORT': 0}
        self.risk_reward_ratios = []
        self.hourly_performance = defaultdict(list)
        self.market_volatility_trades = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
        self.volume_condition_trades = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
        self.rsi_condition_trades = {'OVERSOLD': [], 'NEUTRAL': [], 'OVERBOUGHT': []}
        self.trend_condition_trades = {'BULLISH': [], 'BEARISH': [], 'SIDEWAYS': []}
        self.last_report_time = datetime.now()
        
    def record_trade(self, trade_data):
        """Zeichnet einen Trade mit allen relevanten Daten auf"""
        try:
            # Basis Trade-Daten
            self.trades.append({
                'timestamp': trade_data['timestamp'],
                'symbol': trade_data['symbol'],
                'entry_price': trade_data['entry_price'],
                'exit_price': trade_data['exit_price'],
                'profit_pct': trade_data['profit_pct'],
                'profit_usdt': trade_data['profit_usdt'],
                'duration_minutes': trade_data.get('duration_minutes', 0),
                'exit_reason': trade_data['exit_reason'],
                'position_size_usdt': trade_data.get('position_size_usdt', 0),
                'position_type': 'LONG',  # Derzeit nur Long-Positionen
                'market_conditions': trade_data.get('market_conditions', {})
            })
            
            # Risk/Reward Ratio berechnen
            if trade_data['profit_pct'] > 0:  # Gewinn-Trade
                # Risk = potentieller Verlust bis Stop Loss
                risk_pct = trade_data.get('risk_pct', 0.02)  # Standard 2%
                reward_pct = trade_data['profit_pct']
                risk_reward_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
                self.risk_reward_ratios.append(risk_reward_ratio)
            
            # Position Type tracking
            self.position_types['LONG'] += 1
            
            # Marktbedingungen kategorisieren
            market_conditions = trade_data.get('market_conditions', {})
            self._categorize_market_conditions(trade_data, market_conditions)
            
            # Stündliche Performance
            hour = trade_data['timestamp'].hour
            self.hourly_performance[hour].append(trade_data['profit_pct'])
            
        except Exception as e:
            logging.error(f"Fehler beim Aufzeichnen des Trades: {e}")
    
    def _categorize_market_conditions(self, trade_data, conditions):
        """Kategorisiert Marktbedingungen für Analyse"""
        try:
            profit_pct = trade_data['profit_pct']
            
            # Volatilitäts-basierte Kategorisierung
            atr = conditions.get('atr', 0)
            if atr > 0.003:  # >0.3% ATR
                volatility_cat = 'HIGH'
            elif atr > 0.0015:  # >0.15% ATR
                volatility_cat = 'MEDIUM'
            else:
                volatility_cat = 'LOW'
            self.market_volatility_trades[volatility_cat].append(profit_pct)
            
            # Volume-basierte Kategorisierung
            volume_ratio = conditions.get('volume_ratio', 1.0)
            if volume_ratio > 2.0:
                volume_cat = 'HIGH'
            elif volume_ratio > 1.2:
                volume_cat = 'MEDIUM'
            else:
                volume_cat = 'LOW'
            self.volume_condition_trades[volume_cat].append(profit_pct)
            
            # RSI-basierte Kategorisierung
            rsi = conditions.get('rsi', 50)
            if rsi < 35:
                rsi_cat = 'OVERSOLD'
            elif rsi > 65:
                rsi_cat = 'OVERBOUGHT'
            else:
                rsi_cat = 'NEUTRAL'
            self.rsi_condition_trades[rsi_cat].append(profit_pct)
            
            # Trend-basierte Kategorisierung
            ema_fast = conditions.get('ema_fast', 0)
            ema_slow = conditions.get('ema_slow', 0)
            price = conditions.get('current_price', 0)
            
            if ema_fast > ema_slow and price > ema_fast:
                trend_cat = 'BULLISH'
            elif ema_fast < ema_slow and price < ema_fast:
                trend_cat = 'BEARISH'
            else:
                trend_cat = 'SIDEWAYS'
            self.trend_condition_trades[trend_cat].append(profit_pct)
            
        except Exception as e:
            logging.error(f"Fehler bei Marktbedingung-Kategorisierung: {e}")
    
    def should_print_report(self):
        """Prüft ob ein neuer Report gedruckt werden soll (alle 5 Minuten)"""
        now = datetime.now()
        time_diff = (now - self.last_report_time).total_seconds()
        return time_diff >= 300  # 5 Minuten = 300 Sekunden
    
    def print_detailed_report(self):
        """Druckt detaillierten Performance-Report im Terminal"""
        if not self.should_print_report() or not self.trades:
            return
        
        now = datetime.now()
        all_trades = self.trades  # Sicherstellen, dass all_trades definiert ist
        
        try:
            print("\n" + "="*80)
            print("📊 DETAILLIERTE PERFORMANCE ANALYSE")
            print("="*80)
            
            recent_trades = [t for t in self.trades if (datetime.now() - t['timestamp']).total_seconds() < 3600]  # Letzte Stunde
            all_trades = self.trades
            
            # 1. RISK/REWARD VERHÄLTNIS
            print("\n🎯 RISK/REWARD ANALYSE:")
            if self.risk_reward_ratios:
                avg_rr = statistics.mean(self.risk_reward_ratios)
                median_rr = statistics.median(self.risk_reward_ratios)
                max_rr = max(self.risk_reward_ratios)
                min_rr = min(self.risk_reward_ratios)
                
                print(f"   Durchschnittliches R/R: {avg_rr:.2f}:1")
                print(f"   Median R/R:            {median_rr:.2f}:1")
                print(f"   Bestes R/R:            {max_rr:.2f}:1")
                print(f"   Schlechtestes R/R:     {min_rr:.2f}:1")
                
                # R/R Kategorien
                excellent_rr = len([r for r in self.risk_reward_ratios if r >= 3.0])
                good_rr = len([r for r in self.risk_reward_ratios if 2.0 <= r < 3.0])
                ok_rr = len([r for r in self.risk_reward_ratios if 1.0 <= r < 2.0])
                poor_rr = len([r for r in self.risk_reward_ratios if r < 1.0])
                
                print(f"   Exzellent (≥3:1):      {excellent_rr} Trades ({excellent_rr/len(self.risk_reward_ratios)*100:.1f}%)")
                print(f"   Gut (2-3:1):           {good_rr} Trades ({good_rr/len(self.risk_reward_ratios)*100:.1f}%)")
                print(f"   OK (1-2:1):            {ok_rr} Trades ({ok_rr/len(self.risk_reward_ratios)*100:.1f}%)")
                print(f"   Schlecht (<1:1):       {poor_rr} Trades ({poor_rr/len(self.risk_reward_ratios)*100:.1f}%)")
            else:
                print("   Keine Gewinn-Trades für R/R Berechnung vorhanden")
            
            # 2. MARKTBEDINGUNGEN ANALYSE
            print("\n📈 BESTE MARKTBEDINGUNGEN:")
            
            # Volatilitäts-Performance
            print("   Nach Volatilität:")
            for vol_type, profits in self.market_volatility_trades.items():
                if profits:
                    avg_profit = statistics.mean(profits) * 100
                    win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
                    print(f"     {vol_type:>8} Volatilität: {avg_profit:+.2f}% avg, {win_rate:.1f}% Win Rate ({len(profits)} Trades)")
            
            # Volume-Performance
            print("   Nach Volume:")
            for vol_type, profits in self.volume_condition_trades.items():
                if profits:
                    avg_profit = statistics.mean(profits) * 100
                    win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
                    print(f"     {vol_type:>8} Volume:     {avg_profit:+.2f}% avg, {win_rate:.1f}% Win Rate ({len(profits)} Trades)")
            
            # RSI-Performance
            print("   Nach RSI-Bereich:")
            for rsi_type, profits in self.rsi_condition_trades.items():
                if profits:
                    avg_profit = statistics.mean(profits) * 100
                    win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
                    print(f"     {rsi_type:>10} RSI:      {avg_profit:+.2f}% avg, {win_rate:.1f}% Win Rate ({len(profits)} Trades)")
            
            # Trend-Performance
            print("   Nach Trend-Richtung:")
            for trend_type, profits in self.trend_condition_trades.items():
                if profits:
                    avg_profit = statistics.mean(profits) * 100
                    win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
                    print(f"     {trend_type:>10} Trend:    {avg_profit:+.2f}% avg, {win_rate:.1f}% Win Rate ({len(profits)} Trades)")
            
            # 3. POSITION TYPES (Long vs Short)
            print("\n📍 POSITION VERTEILUNG:")
            total_positions = sum(self.position_types.values())
            if total_positions > 0:
                for pos_type, count in self.position_types.items():
                    percentage = count / total_positions * 100
                    
                    # Performance nach Position Type
                    pos_trades = [t for t in all_trades if t['position_type'] == pos_type]
                    if pos_trades:
                        avg_profit = statistics.mean([t['profit_pct'] for t in pos_trades]) * 100
                        win_rate = len([t for t in pos_trades if t['profit_pct'] > 0]) / len(pos_trades) * 100
                        print(f"   {pos_type:>5} Positionen: {count:>3} ({percentage:.1f}%) - Avg: {avg_profit:+.2f}%, Win Rate: {win_rate:.1f}%")
            
            # 4. ZEITBASIERTE PERFORMANCE
            print("\n⏰ PERFORMANCE NACH UHRZEIT:")
            if self.hourly_performance:
                for hour in sorted(self.hourly_performance.keys()):
                    profits = self.hourly_performance[hour]
                    if len(profits) >= 2:  # Mindestens 2 Trades
                        avg_profit = statistics.mean(profits) * 100
                        win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
                        print(f"     {hour:02d}:00-{hour+1:02d}:00 Uhr: {avg_profit:+.2f}% avg, {win_rate:.1f}% Win Rate ({len(profits)} Trades)")
            
            # 5. TRADE DURATION ANALYSE
            print("\n⌚ TRADE DAUER ANALYSE:")
            durations = [t['duration_minutes'] for t in all_trades if t['duration_minutes'] > 0]
            if durations:
                avg_duration = statistics.mean(durations)
                median_duration = statistics.median(durations)
                
                # Kategorien
                quick_trades = len([d for d in durations if d <= 5])    # ≤5 min
                medium_trades = len([d for d in durations if 5 < d <= 30])  # 5-30 min
                long_trades = len([d for d in durations if d > 30])    # >30 min
                
                print(f"   Durchschnittlich:  {avg_duration:.1f} Minuten")
                print(f"   Median:           {median_duration:.1f} Minuten")
                print(f"   Schnell (≤5min):  {quick_trades} Trades ({quick_trades/len(durations)*100:.1f}%)")
                print(f"   Mittel (5-30min): {medium_trades} Trades ({medium_trades/len(durations)*100:.1f}%)")
                print(f"   Lang (>30min):    {long_trades} Trades ({long_trades/len(durations)*100:.1f}%)")
            
            # 6. SYMBOLE PERFORMANCE
            print("\n💰 PERFORMANCE NACH SYMBOL:")
            symbol_stats = defaultdict(lambda: {'trades': [], 'profits': []})
            for trade in all_trades:
                symbol_stats[trade['symbol']]['trades'].append(trade)
                symbol_stats[trade['symbol']]['profits'].append(trade['profit_pct'])
            
            for symbol, data in sorted(symbol_stats.items()):
                if data['profits']:
                    avg_profit = statistics.mean(data['profits']) * 100
                    win_rate = len([p for p in data['profits'] if p > 0]) / len(data['profits']) * 100
                    total_profit_usdt = sum([t['profit_usdt'] for t in data['trades']])
                    print(f"   {symbol:>8}: {avg_profit:+.2f}% avg, {win_rate:.1f}% Win Rate, {total_profit_usdt:+.2f} USDT total ({len(data['trades'])} Trades)")
            
            # 7. EXIT REASONS ANALYSE
            print("\n🚪 EXIT GRÜNDE ANALYSE:")
            exit_reasons = defaultdict(list)
            for trade in all_trades:
                exit_reasons[trade['exit_reason']].append(trade['profit_pct'])
            
            for reason, profits in sorted(exit_reasons.items(), key=lambda x: len(x[1]), reverse=True):
                avg_profit = statistics.mean(profits) * 100
                win_rate = len([p for p in profits if p > 0]) / len(profits) * 100
                print(f"   {reason:>15}: {avg_profit:+.2f}% avg, {win_rate:.1f}% Win Rate ({len(profits)} mal)")
            
            print("\n" + "="*80)
            print(f"Report Zeit: {datetime.now().strftime('%H:%M:%S')} | Nächster Report in 5 Minuten")
            print("="*80 + "\n")
            
            self.last_report_time = now
            
        except Exception as e:
            logging.error(f"Fehler beim Erstellen des Performance-Reports: {e}")
    def get_dashboard_metrics(self):
        """Generiert Metriken für das Dashboard"""
        if not self.trades:
            return {}
        
        profits = [t['profit_pct'] for t in self.trades]
        profits_usdt = [t['profit_usdt'] for t in self.trades]
        
        return {
            'total_trades': len(self.trades),
            'total_profit_usdt': sum(profits_usdt),
            'avg_profit_pct': statistics.mean(profits) * 100,
            'win_rate': len([p for p in profits if p > 0]) / len(profits) * 100,
            'best_trade': max(profits) * 100,
            'worst_trade': min(profits) * 100,
            'avg_trade_duration': statistics.mean([t.get('duration_minutes', 0) for t in self.trades]),
            'risk_reward_ratio': statistics.mean(self.risk_reward_ratios) if self.risk_reward_ratios else 0,
            'hourly_performance': dict(self.hourly_performance),
            'symbol_performance': self._get_symbol_performance(),
            'recent_trades': self.trades[-10:]  # Last 10 trades
        }

    def _get_symbol_performance(self):
        """Symbolspezifische Performance"""
        symbol_stats = {}
        for trade in self.trades:
            symbol = trade['symbol']
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {'trades': 0, 'profit': 0, 'wins': 0}
            
            symbol_stats[symbol]['trades'] += 1
            symbol_stats[symbol]['profit'] += trade['profit_usdt']
            if trade['profit_pct'] > 0:
                symbol_stats[symbol]['wins'] += 1
        
        # Calculate win rates
        for symbol, stats in symbol_stats.items():
            stats['win_rate'] = (stats['wins'] / stats['trades']) * 100
        
        return symbol_stats

# INTEGRATION IN DEN HAUPTBOT - In der __init__ Methode des AdvancedBinanceTrader:
