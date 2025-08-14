"""
Enhanced Flask Web Interface Module
Comprehensive dashboard for trading bot monitoring
"""

from flask import Flask, jsonify, render_template_string
import threading
import logging
from datetime import datetime, timedelta
import json

# Global variables for enhanced status tracking
app = Flask(__name__)
live_status = {}
test_results = {}
optimization_status = {'running': False, 'progress': 0, 'best_params': None}
trade_history = []
performance_metrics = {}
profit_tracking = {
    'daily_profits': [],
    'hourly_profits': {},
    'symbol_profits': {},
    'total_profit': 0.0,
    'total_trades': 0,
    'winning_trades': 0,
    'current_balance': 0.0,
    'start_balance': 0.0
}
MACRO_DISPLAY_VALUE = """
{% macro display_value(value, precision=2) -%}
  {%- if value is number -%}
    {{ value|round(precision) }}
  {%- else -%}
    {{ value }}
  {%- endif -%}
{%- endmacro %}
"""

# HTML Template für das Dashboard
DASHBOARD_HTML = MACRO_DISPLAY_VALUE + """
<!DOCTYPE html>
<html>
<head>
    <title>🤖 Advanced Trading Bot Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f1419; color: #e6e6e6; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #1e2328 0%, #2b3139 100%); padding: 30px; border-radius: 12px; margin-bottom: 30px; text-align: center; }
        .header h1 { color: #f0b90b; font-size: 2.5em; margin-bottom: 10px; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .status-card { background: #1e2328; border-radius: 12px; padding: 25px; border-left: 4px solid #f0b90b; }
        .status-card h3 { color: #f0b90b; margin-bottom: 15px; font-size: 1.3em; }
        .profit-positive { color: #02c076 !important; font-weight: bold; }
        .profit-negative { color: #f6465d !important; font-weight: bold; }
        .metric-value { font-size: 2em; font-weight: bold; margin: 10px 0; }
        .chart-container { background: #1e2328; border-radius: 12px; padding: 25px; margin: 20px 0; }
        .trades-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        .trades-table th, .trades-table td { padding: 12px; text-align: left; border-bottom: 1px solid #2b3139; }
        .trades-table th { background: #2b3139; color: #f0b90b; font-weight: bold; }
        .symbol-status { display: flex; justify-content: space-between; align-items: center; margin: 10px 0; padding: 15px; background: #2b3139; border-radius: 8px; }
        .signal-strong-buy { background: linear-gradient(90deg, #02c076, #03a66d); color: white; padding: 5px 10px; border-radius: 5px; font-weight: bold; }
        .signal-buy { background: linear-gradient(90deg, #17a2b8, #138496); color: white; padding: 5px 10px; border-radius: 5px; }
        .signal-neutral { background: linear-gradient(90deg, #6c757d, #545b62); color: white; padding: 5px 10px; border-radius: 5px; }
        .signal-sell { background: linear-gradient(90deg, #dc3545, #c82333); color: white; padding: 5px 10px; border-radius: 5px; }
        .refresh-btn { background: #f0b90b; color: #0f1419; border: none; padding: 12px 25px; border-radius: 8px; cursor: pointer; font-weight: bold; margin: 10px; }
        .refresh-btn:hover { background: #d4a50a; }
        .parameter-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 20px; }
        .parameter-item { background: #2b3139; padding: 15px; border-radius: 8px; }
        .parameter-item strong { color: #f0b90b; }
        .ml-status { display: inline-block; padding: 5px 12px; border-radius: 20px; font-size: 0.9em; font-weight: bold; }
        .ml-active { background: #02c076; color: white; }
        .ml-inactive { background: #f6465d; color: white; }
        .performance-highlight { background: linear-gradient(135deg, #f0b90b, #d4a50a); color: #0f1419; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .active-position {border-left: 4px solid #02c076 !important; background: linear-gradient(90deg, #1a5c47, #2b3139) !important; }
        .signal-hold { background: linear-gradient(90deg, #95a5a6, #7f8c8d); color: white; padding: 5px 10px; border-radius: 5px; }
        .position-profit { color: #02c076; font-weight: bold; }
        .position-loss { color: #f6465d; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Advanced Trading Bot Dashboard</h1>
            <p>Live-Monitoring • ML-Integration • Performance-Tracking</p>
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Dashboard</button>
        </div>

        <!-- Performance Overview -->
        <div class="status-grid">
            <div class="status-card">
                <h3>💰 Profit & Loss</h3>
                <div class="metric-value profit-{{ 'positive' if profit_tracking.total_profit > 0 else 'negative' }}">
                    {{ "+" if profit_tracking.total_profit > 0 else "" }}{{ display_value(profit_tracking.total_profit) }} USDT
                </div>
                <p>Balance: {{ display_value(profit_tracking.current_balance) }} USDT</p>
                <p>ROI: {{ display_value(((profit_tracking.current_balance / profit_tracking.start_balance - 1) * 100)) if profit_tracking.start_balance > 0 else 0 }}%</p>
            </div>
            
            <div class="status-card">
                <h3>📊 Trade Statistics</h3>
                <div class="metric-value">{{ profit_tracking.total_trades }} Trades</div>
                <p>Win Rate: {{ display_value(((profit_tracking.winning_trades / profit_tracking.total_trades) * 100)) if profit_tracking.total_trades > 0 else 0 }}%</p>
                <p>Winning: {{ profit_tracking.winning_trades }} | Losing: {{ profit_tracking.total_trades - profit_tracking.winning_trades }}</p>
            </div>

            <div class="status-card">
                <h3>🧠 ML System Status</h3>
                <span class="ml-status ml-{{ 'active' if optimization_status.get('ml_enabled', False) else 'inactive' }}">
                    {{ 'ACTIVE' if optimization_status.get('ml_enabled', False) else 'INACTIVE' }}
                </span>
                <p>Models: {{ optimization_status.get('ml_models', 0) }}</p>
                <p>Last Prediction: {{ optimization_status.get('last_ml_time', 'N/A') }}</p>
            </div>

            <div class="status-card">
                <h3>⚙️ Optimization Status</h3>
                <div class="metric-value">{{ optimization_status.get('progress', 0) }}%</div>
                <p>{{ 'Running...' if optimization_status.get('running', False) else 'Completed' }}</p>
                <p>Best Score: {{ display_value(optimization_status.get('best_score', 0)) }}</p>
            </div>
        </div>

        <!-- Live Trading Status -->
        <div class="chart-container">
            <h3>📈 Live Trading Status & Active Positions</h3>
            
            <!-- Active Positions Section -->
            {% if live_status.values() | selectattr('in_position', 'equalto', true) | list %}
            <div style="background: #0d4f3c; padding: 20px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #02c076;">
                <h4 style="color: #02c076; margin-bottom: 15px;">🎯 AKTIVE POSITIONEN</h4>
                {% for symbol, data in live_status.items() %}
                {% if data.get('in_position', False) %}
                <div class="symbol-status" style="background: #1a5c47; border-left: 3px solid #02c076;">
                    <div>
                        <strong style="color: #02c076;">{{ symbol }} - IN POSITION</strong>
                        <span style="color: #f0b90b;">{{ display_value(data.price) }} USDT</span>
                    </div>
                    <div>
                        <span style="color: #02c076; font-weight: bold;">
                            Entry: {{ display_value(data.entry_price) }} → 
                            P/L: {{ display_value(((data.price / data.entry_price - 1) * 100)) }}%
                            ({{ display_value((((data.price / data.entry_price - 1) * data.get('position_value', 0)))) }} USDT)
                        </span>
                        {% if data.get('trailing_stop') %}
                        <span style="color: #f39c12; margin-left: 15px;">
                            Trailing Stop: {{ display_value(data.trailing_stop) }}
                        </span>
                        {% endif %}
                    </div>
                    <div>
                        <span style="font-size: 0.9em; color: #bdc3c7;">
                            Duration: {{ data.get('position_duration', 'N/A') }} | 
                            RSI: {{ display_value(data.get('rsi', 0)) }} | 
                            Reason: {{ data.get('entry_reason', 'N/A') }}
                        </span>
                    </div>
                </div>
                {% endif %}
                {% endfor %}
            </div>
            {% else %}
            <div style="background: #2c3e50; padding: 15px; border-radius: 8px; margin-bottom: 20px; text-align: center; color: #7f8c8d;">
                <h4>💤 Keine aktiven Positionen</h4>
                <p>Bot wartet auf Handelssignale...</p>
            </div>
            {% endif %}
            
            <!-- All Symbols Status -->
            <h4 style="color: #f0b90b; margin-bottom: 15px;">📊 Alle Symbole - Signal Status</h4>
            {% for symbol, data in live_status.items() %}
            <div class="symbol-status {% if data.get('in_position', False) %}active-position{% endif %}">
                <div>
                    <strong>{{ symbol }}</strong>
                    <span style="color: #f0b90b;">{{ display_value(data.price) }} USDT</span>
                    {% if data.get('price_change_24h') %}
                    <span style="color: {{ '#02c076' if data.price_change_24h > 0 else '#f6465d' }}; margin-left: 10px;">
                        {{ "+" if data.price_change_24h > 0 else "" }}{{ display_value(data.price_change_24h) }}%
                    </span>
                    {% endif %}
                </div>
                <div>
                    <span class="signal-{{ data.get('signal', 'neutral').lower().replace('_', '-') }}">
                        {{ data.get('signal', 'HOLD') }}
                    </span>
                    <span style="margin-left: 10px; color: #bdc3c7;">
                        Confidence: {{ display_value((data.get('confidence', 0) * 100)) }}%
                    </span>
                    {% if data.get('in_position', False) %}
                    <span style="color: #02c076; font-weight: bold; margin-left: 10px;">
                        🎯 TRADING ({{ display_value(((data.price / data.entry_price - 1) * 100)) }}%)
                    </span>
                    {% else %}
                    <span style="color: #7f8c8d; margin-left: 10px;">
                        {% if data.get('signal', 'HOLD') == 'HOLD' %}
                        💤 Waiting for signal
                        {% else %}
                        ⏳ Signal detected, checking conditions
                        {% endif %}
                    </span>
                    {% endif %}
                </div>
                <div>
                    RSI: {{ display_value(data.get('rsi', 0)) }} | 
                    ML: {{ display_value(data.get('ml_confidence', 0)) }} |
                    Volume: {{ data.get('volume_status', 'NORMAL') }}
                    {% if data.get('reasons') %}
                    <br><small style="color: #95a5a6;">{{ data.get('reasons', [])|join(', ') }}</small>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>

        <!-- Best Parameters -->
        {% if optimization_status.get('best_params') %}
        <div class="chart-container">
            <h3>🏆 Optimized Parameters ({{ display_value(optimization_status.get('best_score', 0)) }} Score)</h3>
            <div class="performance-highlight">
                <p><strong>Profit Expectation:</strong> {{ display_value((optimization_status.get('expected_profit', 0) * 100)) }}% per trade</p>
                <p><strong>Win Rate:</strong> {{ display_value((optimization_status.get('expected_winrate', 0) * 100)) }}%</p>
                <p><strong>Risk/Reward:</strong> {{ display_value(optimization_status.get('avg_risk_reward', 0)) }}:1</p>
            </div>
            <div class="parameter-grid">
                {% for key, value in optimization_status.get('best_params', {}).items() %}
                <div class="parameter-item">
                    <strong>{{ key.replace('_', ' ').title() }}:</strong> {{ value }}
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        <!-- Recent Trades -->
        {% if trade_history %}
        <div class="chart-container">
            <h3>📋 Recent Trades</h3>
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>Profit %</th>
                        <th>Profit USDT</th>
                        <th>Duration</th>
                        <th>Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {% for trade in trade_history[-10:] %}
                    <tr>
                        <td>{{ trade.timestamp.strftime('%H:%M:%S') if trade.get('timestamp') else 'N/A' }}</td>
                        <td>{{ trade.symbol }}</td>
                        <td>{{ display_value(trade.entry_price) }}</td>
                        <td>{{ display_value(trade.exit_price) }}</td>
                        <td class="profit-{{ 'positive' if trade.profit_pct > 0 else 'negative' }}">
                            {{ display_value((trade.profit_pct * 100)) }}%
                        </td>
                        <td class="profit-{{ 'positive' if trade.profit_usdt > 0 else 'negative' }}">
                            {{ "+" if trade.profit_usdt > 0 else "" }}{{ display_value(trade.profit_usdt) }}
                        </td>
                        <td>{{ display_value(trade.get('duration_minutes', 0)) }}m</td>
                        <td>{{ trade.exit_reason }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <!-- Footer -->
        <div style="text-align: center; margin-top: 50px; padding: 20px; color: #6c757d;">
            <p>Last Update: <span id="lastUpdate">{{ datetime.now().strftime('%Y-%m-%d %H:%M:%S') }}</span></p>
            <p>🤖 Enhanced Trading Bot • Auto-refresh every 10 seconds</p>
        </div>
    </div>

    <script>
        // Auto-refresh every 10 seconds
        setTimeout(() => location.reload(), 10000);
        
        // Update timestamp
        document.getElementById('lastUpdate').textContent = new Date().toLocaleString();
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard with comprehensive trading information"""
    return render_template_string(DASHBOARD_HTML, 
                                live_status=live_status,
                                optimization_status=optimization_status,
                                profit_tracking=profit_tracking,
                                trade_history=trade_history,
                                performance_metrics=performance_metrics,
                                datetime=datetime)

@app.route("/status")
def status():
    """Get current trading status for all symbols"""
    return jsonify(live_status)

@app.route("/test-results")
def test_results_endpoint():
    """Get backtesting and optimization results"""
    return jsonify(test_results)

@app.route("/optimization")
def optimization():
    """Get current optimization status"""
    return jsonify(optimization_status)

@app.route("/best-params")
def best_params():
    """Get the best optimized parameters"""
    return jsonify(optimization_status.get('best_params', {}))

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "modules_loaded": len([k for k in locals() if not k.startswith('_')]),
        "active_positions": len([s for s in live_status.values() if s.get('in_position', False)])
    })

@app.route('/api/status')
def api_status():
    """API endpoint for current trading status"""
    return jsonify({
        'live_status': live_status,
        'profit_tracking': profit_tracking,
        'performance_metrics': performance_metrics,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/trades')
def api_trades():
    """API endpoint for trade history"""
    return jsonify({
        'trades': trade_history[-50:],  # Last 50 trades
        'total_trades': len(trade_history),
        'profit_summary': profit_tracking
    })

@app.route('/api/optimization')
def api_optimization():
    """API endpoint for optimization results"""
    return jsonify(optimization_status)

@app.route('/api/performance')
def api_performance():
    """API endpoint for detailed performance metrics"""
    return jsonify(performance_metrics)

def start_flask_server():
    """Start Flask server in a separate thread"""
    def run_flask():
        try:
            app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        except Exception as e:
            logging.error(f"Flask server error: {e}")
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logging.info("🌐 Enhanced Flask Dashboard started on http://localhost:5000")

# Helper functions for updating status
def update_live_status(symbol, data):
    """Update live status for a symbol with enhanced data"""
    live_status[symbol] = data

def update_test_results(results):
    """Update test results"""
    global test_results
    test_results.update(results)

def update_optimization_status(status):
    """Update optimization status with enhanced metrics"""
    global optimization_status
    optimization_status.update(status)

def record_trade(trade_data):
    """Record a new trade with profit tracking"""
    global trade_history, profit_tracking
    
    # Add to trade history
    trade_history.append(trade_data)
    
    # Update profit tracking
    profit_tracking['total_trades'] += 1
    profit_tracking['total_profit'] += trade_data.get('profit_usdt', 0)
    
    if trade_data.get('profit_pct', 0) > 0:
        profit_tracking['winning_trades'] += 1
    
    # Symbol-specific profits
    symbol = trade_data.get('symbol', 'UNKNOWN')
    if symbol not in profit_tracking['symbol_profits']:
        profit_tracking['symbol_profits'][symbol] = {'profit': 0, 'trades': 0}
    
    profit_tracking['symbol_profits'][symbol]['profit'] += trade_data.get('profit_usdt', 0)
    profit_tracking['symbol_profits'][symbol]['trades'] += 1
    
    # Hourly profits
    hour = trade_data.get('timestamp', datetime.now()).hour
    if hour not in profit_tracking['hourly_profits']:
        profit_tracking['hourly_profits'][hour] = 0
    profit_tracking['hourly_profits'][hour] += trade_data.get('profit_usdt', 0)

def update_balance(current_balance, start_balance=None):
    """Update current balance"""
    global profit_tracking
    profit_tracking['current_balance'] = current_balance
    if start_balance:
        profit_tracking['start_balance'] = start_balance

def update_performance_metrics(metrics):
    """Update performance metrics"""
    global performance_metrics
    performance_metrics.update(metrics)




