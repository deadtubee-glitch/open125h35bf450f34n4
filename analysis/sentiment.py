import logging
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    """Analysiert News-Sentiment für Crypto-Trading Entscheidungen"""
    
    def __init__(self, config):
        self.config = config
        self.sentiment_enabled = config.get('advanced', {}).get('news_sentiment', False)
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # News-API Keys (kostenlose Optionen)
        self.news_sources = {
            'newsapi': {
                'url': 'https://newsapi.org/v2/everything',
                'key': config.get('news_api', {}).get('newsapi_key', ''),
                'enabled': True
            },
            'cryptonews': {
                'url': 'https://cryptonews-api.com/api/v1/category',
                'key': config.get('news_api', {}).get('cryptonews_key', ''),
                'enabled': True
            }
        }
        
        # Sentiment Cache (vermeidet zu viele API calls)
        self.sentiment_cache = {}
        self.cache_duration = 300  # 5 Minuten Cache
        
        # Symbol-spezifische Keywords
        self.symbol_keywords = {
            'BTCUSDT': ['bitcoin', 'BTC', 'cryptocurrency', 'crypto market'],
            'ETHUSDT': ['ethereum', 'ETH', 'smart contracts', 'DeFi'],
            'BNBUSDT': ['binance', 'BNB', 'binance coin', 'binance smart chain'],
            'ADAUSDT': ['cardano', 'ADA', 'proof of stake'],
            'SOLUSDT': ['solana', 'SOL', 'blockchain']
        }
        
        logging.info(f"News Sentiment Analyzer {'aktiviert' if self.sentiment_enabled else 'deaktiviert'}")

    def get_crypto_news(self, symbol, hours_back=24):
        """Holt aktuelle Crypto-News für Symbol"""
        if not self.sentiment_enabled:
            return []  # niemals None
    
        try:
            keywords = self.symbol_keywords.get(symbol, [symbol.replace('USDT','')])
            news_articles = []
    
            # Nur falls API-Key vorhanden
            api_key = self.news_sources['newsapi']['key']
            if api_key:
                news_articles.extend(self._fetch_from_newsapi(keywords, hours_back))
    
            # Fallback auf öffentliche Quellen
            if len(news_articles) < 5:
                news_articles.extend(self._fetch_from_public_sources(keywords))
    
            return news_articles[:20]  # immer Liste
        except Exception as e:
            logging.error(f"Fehler beim News-Abruf für {symbol}: {e}")
            return []  # Fallback-Liste
    
    def _fetch_from_public_sources(self, keywords):
        """Fallback für öffentliche News-Quellen"""
        try:
            # Dummy-Implementierung für Stabilität
            return [{
                'title': f'Crypto Market Update - {keywords[0] if keywords else "General"}',
                'description': 'Market analysis and trends',
                'content': 'General market sentiment analysis',
                'url': 'https://example.com',
                'publishedAt': datetime.now().isoformat(),
                'source': 'Public'
            }]
        except:
            return []


    def _fetch_from_newsapi(self, keywords, hours_back):
        """Holt News von NewsAPI.org"""
        try:
            api_key = self.news_sources['newsapi']['key']
            if not api_key:
                return []
                
            query = ' OR '.join(keywords)
            from_date = (datetime.now() - timedelta(hours=hours_back)).isoformat()
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'popularity',
                'language': 'en',
                'apiKey': api_key
            }
            
            response = requests.get(self.news_sources['newsapi']['url'], params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = []
                
                for article in data.get('articles', [])[:10]:
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'publishedAt': article.get('publishedAt', ''),
                        'source': 'NewsAPI'
                    })
                
                return articles
                
        except Exception as e:
            logging.warning(f"NewsAPI Fehler: {e}")
            return []

    def _fetch_from_cryptonews(self, keywords):
        """Holt News von Crypto-spezifischen Quellen"""
        try:
            # Öffentliche Crypto-News APIs (ohne API Key)
            urls = [
                'https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories=BTC,ETH,Trading',
                'https://api.coingecko.com/api/v3/news'
            ]
            
            articles = []
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # CryptoCompare Format
                        if 'Data' in data:
                            for item in data['Data'][:5]:
                                articles.append({
                                    'title': item.get('title', ''),
                                    'description': item.get('body', '')[:200],
                                    'content': item.get('body', ''),
                                    'url': item.get('url', ''),
                                    'publishedAt': item.get('published_on', ''),
                                    'source': 'CryptoCompare'
                                })
                        
                        # CoinGecko Format
                        elif isinstance(data, list):
                            for item in data[:5]:
                                articles.append({
                                    'title': item.get('title', ''),
                                    'description': item.get('description', '')[:200],
                                    'content': item.get('content', ''),
                                    'url': item.get('url', ''),
                                    'publishedAt': item.get('updated_at', ''),
                                    'source': 'CoinGecko'
                                })
                                
                except Exception as e:
                    continue
                    
            return articles
            
        except Exception as e:
            logging.warning(f"Crypto News API Fehler: {e}")
            return []

    def _fetch_from_social_media(self, keywords):
        """Holt Sentiment aus sozialen Medien (öffentliche APIs)"""
        try:
            articles = []
            
            # Reddit Public API (ohne API Key)
            for keyword in keywords[:2]:  # Nur erste 2 Keywords
                try:
                    reddit_url = f"https://www.reddit.com/r/cryptocurrency/search.json?q={keyword}&sort=new&limit=5"
                    response = requests.get(reddit_url, headers={'User-Agent': 'CryptoBot 1.0'}, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        for post in data.get('data', {}).get('children', []):
                            post_data = post.get('data', {})
                            articles.append({
                                'title': post_data.get('title', ''),
                                'description': post_data.get('selftext', '')[:200],
                                'content': post_data.get('selftext', ''),
                                'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                'publishedAt': post_data.get('created_utc', ''),
                                'source': 'Reddit'
                            })
                except:
                    continue
                    
            return articles[:5]
            
        except Exception as e:
            logging.warning(f"Social Media API Fehler: {e}")
            return []

    def analyze_sentiment(self, articles):
        """Analysiert Sentiment der News-Artikel"""
        if not articles:
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'articles_analyzed': 0
            }
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in articles:
            try:
                # Text zusammenfassen
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if not text.strip():
                    continue
                
                # VADER Sentiment (besser für kurze Texte)
                vader_scores = self.vader_analyzer.polarity_scores(text)
                vader_sentiment = vader_scores['compound']  # -1 bis +1
                
                # TextBlob Sentiment (zusätzliche Validierung)
                try:
                    blob = TextBlob(text)
                    textblob_sentiment = blob.sentiment.polarity  # -1 bis +1
                except:
                    textblob_sentiment = 0
                
                # Gewichteter Durchschnitt
                combined_sentiment = (vader_sentiment * 0.7) + (textblob_sentiment * 0.3)
                sentiments.append(combined_sentiment)
                
                # Kategorisierung
                if combined_sentiment > 0.1:
                    positive_count += 1
                elif combined_sentiment < -0.1:
                    negative_count += 1
                else:
                    neutral_count += 1
                    
            except Exception as e:
                logging.warning(f"Sentiment-Analyse Fehler: {e}")
                continue
        
        if not sentiments:
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'articles_analyzed': 0
            }
        
        # Gesamtbewertung
        overall_sentiment = sum(sentiments) / len(sentiments)
        
        # Konfidenz basierend auf Konsistenz
        sentiment_std = np.std(sentiments) if len(sentiments) > 1 else 0
        confidence = max(0, 1 - (sentiment_std / 0.5))  # Je konsistenter, desto höher
        
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': confidence,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'articles_analyzed': len(sentiments),
            'sentiment_distribution': sentiments
        }

    def get_sentiment_for_symbol(self, symbol):
        """Holt und cached Sentiment für Symbol"""
        cache_key = f"{symbol}_{int(datetime.now().timestamp() / self.cache_duration)}"
        
        # Cache-Check
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        # News abrufen und analysieren
        articles = self.get_crypto_news(symbol)
        sentiment_data = self.analyze_sentiment(articles)
        
        # Cachen
        self.sentiment_cache[cache_key] = sentiment_data
        
        # Alte Cache-Einträge löschen
        current_time = int(datetime.now().timestamp() / self.cache_duration)
        old_keys = [k for k in self.sentiment_cache.keys() if not k.endswith(str(current_time))]
        for old_key in old_keys:
            del self.sentiment_cache[old_key]
        
        return sentiment_data

    def get_sentiment_signal(self, symbol):
        """Generiert Trading-Signal basierend auf Sentiment"""
        if not self.sentiment_enabled:
            return {'signal': 'NEUTRAL', 'strength': 0.0, 'reason': 'Sentiment deaktiviert'}
        
        sentiment_data = self.get_sentiment_for_symbol(symbol)
        
        overall_sentiment = sentiment_data['overall_sentiment']
        confidence = sentiment_data['confidence']
        articles_count = sentiment_data['articles_analyzed']
        
        # Mindestanzahl Artikel für verlässliche Signale
        if articles_count < 3:
            return {'signal': 'NEUTRAL', 'strength': 0.0, 'reason': 'Zu wenige News'}
        
        # Signal-Stärke basierend auf Sentiment und Konfidenz
        signal_strength = abs(overall_sentiment) * confidence
        
        # Signal-Generierung
        if overall_sentiment > 0.2 and confidence > 0.6:
            signal = 'BULLISH'
            reason = f"Positive News ({articles_count} Artikel, Sentiment: {overall_sentiment:.2f})"
        elif overall_sentiment < -0.2 and confidence > 0.6:
            signal = 'BEARISH' 
            reason = f"Negative News ({articles_count} Artikel, Sentiment: {overall_sentiment:.2f})"
        elif overall_sentiment > 0.1 and confidence > 0.4:
            signal = 'WEAK_BULLISH'
            reason = f"Schwach positive News ({articles_count} Artikel)"
        elif overall_sentiment < -0.1 and confidence > 0.4:
            signal = 'WEAK_BEARISH'
            reason = f"Schwach negative News ({articles_count} Artikel)"
        else:
            signal = 'NEUTRAL'
            reason = f"Gemischte/neutrale News ({articles_count} Artikel)"
        
        return {
            'signal': signal,
            'strength': signal_strength,
            'reason': reason,
            'sentiment_score': overall_sentiment,
            'confidence': confidence,
            'articles_count': articles_count,
            'details': sentiment_data
        }
