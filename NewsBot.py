import requests
import json
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
import re
from typing import Dict, List, Optional
import configparser
import os
from twelvedata import TDClient

class FinancialDataCollector:
    def __init__(self, config_file='config.ini'):
        """Initialize the data collector with Twelve Data API"""
        self.config = configparser.ConfigParser()
        if os.path.exists(config_file):
            self.config.read(config_file)
        
        # Initialize Twelve Data client
        self.td_client = None
        if self.config.has_option('TWELVE_DATA', 'api_key'):
            api_key = self.config['TWELVE_DATA']['fdfdccc2ba2048d59f9276ed9ff2dc06']
            self.td_client = TDClient(apikey=api_key)
            print(f"✓ Twelve Data API initialized")
        else:
            print("⚠ Twelve Data API key not found in config.ini")
        
        # Headers for web scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Currency pair mappings for Twelve Data (format: EUR/USD)
        self.currency_map = {
            'USD': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'USD/CAD', 'AUD/USD', 'NZD/USD'],
            'EUR': ['EUR/USD', 'EUR/GBP', 'EUR/JPY', 'EUR/CHF', 'EUR/CAD', 'EUR/AUD'],
            'GBP': ['GBP/USD', 'EUR/GBP', 'GBP/JPY', 'GBP/CHF', 'GBP/CAD', 'GBP/AUD'],
            'JPY': ['USD/JPY', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'CAD/JPY', 'NZD/JPY'],
            'AUD': ['AUD/USD', 'EUR/AUD', 'GBP/AUD', 'AUD/JPY', 'AUD/CAD', 'AUD/NZD'],
            'CAD': ['USD/CAD', 'EUR/CAD', 'GBP/CAD', 'CAD/JPY', 'AUD/CAD'],
            'CHF': ['USD/CHF', 'EUR/CHF', 'GBP/CHF', 'CHF/JPY'],
            'NZD': ['NZD/USD', 'EUR/NZD', 'GBP/NZD', 'NZD/JPY', 'AUD/NZD']
        }
    
    def get_forex_data_around_time(self, currency_pairs: List[str], event_time: datetime, 
                                   hours_before: int = 1, hours_after: int = 1) -> pd.DataFrame:
        """
        Get forex data around a specific news event time
        Shows price BEFORE and AFTER news to see actual impact!
        
        Args:
            currency_pairs: List of currency codes or pairs
            event_time: Time of the news event
            hours_before: Hours before event to start
            hours_after: Hours after event to end
            
        Returns:
            DataFrame with before/after prices and % change
        """
        if not self.td_client:
            print("⚠ Twelve Data API not configured")
            return pd.DataFrame()
        
        forex_data = []
        
        # Convert currency codes to Twelve Data format
        symbols_to_fetch = []
        for item in currency_pairs:
            if '/' in item:
                # Already in correct format
                symbols_to_fetch.append(item)
            elif len(item) == 3 and item.upper() in self.currency_map:
                # Currency code, get major pairs
                symbols_to_fetch.extend(self.currency_map[item.upper()][:3])  # Top 3 pairs
        
        # Remove duplicates
        symbols_to_fetch = list(set(symbols_to_fetch))
        
        # Calculate time window
        start_time = event_time - timedelta(hours=hours_before)
        end_time = event_time + timedelta(hours=hours_after)
        
        print(f"📊 Fetching data for {len(symbols_to_fetch)} pairs around {event_time.strftime('%H:%M')}...")
        print(f"   Window: {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}")
        
        for pair in symbols_to_fetch:
            try:
                # Get minute-level data for the time window
                ts = self.td_client.time_series(
                    symbol=pair,
                    interval="5min",
                    outputsize=50,
                    timezone="UTC",
                    start_date=start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    end_date=end_time.strftime('%Y-%m-%d %H:%M:%S')
                )
                
                df = ts.as_pandas()
                
                if not df.empty and len(df) > 1:
                    # Get price BEFORE news (first data point)
                    price_before = float(df.iloc[0]['close'])
                    time_before = df.index[0]
                    
                    # Get price AFTER news (last data point)
                    price_after = float(df.iloc[-1]['close'])
                    time_after = df.index[-1]
                    
                    # Calculate change
                    change_pips = price_after - price_before
                    change_percent = ((price_after - price_before) / price_before) * 100
                    
                    # Get high/low during event
                    high = float(df['high'].max())
                    low = float(df['low'].min())
                    volatility = ((high - low) / price_before) * 100
                    
                    forex_data.append({
                        'pair': pair,
                        'event_time': event_time,
                        'price_before': price_before,
                        'price_after': price_after,
                        'change_pips': change_pips,
                        'change_percent': change_percent,
                        'high_during_event': high,
                        'low_during_event': low,
                        'volatility_percent': volatility,
                        'time_before': time_before,
                        'time_after': time_after,
                        'data_points': len(df)
                    })
                    
                    emoji = "🟢" if change_percent >= 0 else "🔴"
                    print(f"  {emoji} {pair}: {price_before:.5f} → {price_after:.5f} ({change_percent:+.2f}%)")
                
                time.sleep(1)  # Rate limiting for Twelve Data
                
            except Exception as e:
                print(f"  ✗ Error fetching {pair}: {e}")
                continue
        
        return pd.DataFrame(forex_data)
    
    def parse_event_time(self, time_str: str, date: datetime = None) -> Optional[datetime]:
        """
        Parse event time from Forex Factory
        
        Args:
            time_str: Time string from Forex Factory (e.g., "8:30am", "2:00pm")
            date: Date of the event (defaults to today)
            
        Returns:
            datetime object or None if parsing fails
        """
        if not date:
            date = datetime.now()
        
        try:
            # Clean the time string
            time_str = time_str.strip().lower()
            
            # Parse time
            if 'am' in time_str or 'pm' in time_str:
                time_obj = datetime.strptime(time_str, '%I:%M%p').time()
            elif ':' in time_str:
                time_obj = datetime.strptime(time_str, '%H:%M').time()
            else:
                return None
            
            # Combine with date
            event_datetime = datetime.combine(date.date(), time_obj)
            
            return event_datetime
            
        except Exception as e:
            print(f"  ⚠ Could not parse time '{time_str}': {e}")
            return None
    
    def scrape_businessinsider_news(self, symbol: str = 'nvda') -> List[Dict]:
        """Scrape news from Business Insider Markets"""
        url = f"https://markets.businessinsider.com/news/{symbol}-stock"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            return self.parse_businessinsider_news(soup)
        except Exception as e:
            print(f"Error scraping Business Insider: {e}")
            return []
    
    def parse_businessinsider_news(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse Business Insider news from HTML"""
        news_items = []
        
        # Look for news articles
        articles = soup.find_all('article', class_=re.compile('.*news.*|.*story.*', re.I))
        if not articles:
            articles = soup.find_all('div', class_=re.compile('.*news.*|.*story.*', re.I))
        
        for article in articles[:10]:  # Limit to top 10 articles
            try:
                title_elem = article.find(['h2', 'h3', 'h4'])
                title = title_elem.get_text(strip=True) if title_elem else ''
                
                content_elem = article.find('p')
                content = content_elem.get_text(strip=True) if content_elem else ''
                
                link_elem = article.find('a')
                url = link_elem.get('href', '') if link_elem else ''
                if url and not url.startswith('http'):
                    url = 'https://markets.businessinsider.com' + url
                
                news_items.append({
                    'title': title,
                    'content': content,
                    'source': 'Business Insider',
                    'url': url,
                    'published_at': datetime.now().isoformat(),
                    'keywords': self.extract_keywords(title + ' ' + content)
                })
            except Exception as e:
                continue
        
        return news_items
    
    def scrape_forexfactory_calendar(self) -> List[Dict]:
        """Scrape economic calendar from Forex Factory"""
        url = "https://www.forexfactory.com/calendar"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            return self.parse_forexfactory_calendar(soup)
        except Exception as e:
            print(f"Error scraping Forex Factory: {e}")
            return []
    
    def parse_forexfactory_calendar(self, soup: BeautifulSoup) -> List[Dict]:
        """Parse Forex Factory economic calendar with event times"""
        events = []
        
        # Find the calendar table
        calendar_table = soup.find('table', class_='calendar__table')
        if not calendar_table:
            return events
        
        # Find all event rows
        rows = calendar_table.find_all('tr', class_=re.compile('calendar__row'))
        
        current_date = datetime.now()
        
        for row in rows[:1000]:  # Get more events
            try:
                # Extract time
                time_elem = row.find('td', class_='time')
                time_str = time_elem.get_text(strip=True) if time_elem else ''
                
                # Extract event data
                currency = row.find('td', class_='currency')
                currency = currency.get_text(strip=True) if currency else ''
                
                # Extract impact (high/medium/low)
                impact = row.find('td', class_='impact')
                impact_class = impact.get('class') if impact else []
                impact_level = 'unknown'
                if 'high' in str(impact_class):
                    impact_level = 'high'
                elif 'medium' in str(impact_class):
                    impact_level = 'medium'
                elif 'low' in str(impact_class):
                    impact_level = 'low'
                
                event_elem = row.find('td', class_='event')
                event_name = event_elem.get_text(strip=True) if event_elem else ''
                
                actual = row.find('td', class_='actual')
                actual_val = actual.get_text(strip=True) if actual else ''
                
                forecast = row.find('td', class_='forecast')
                forecast_val = forecast.get_text(strip=True) if forecast else ''
                
                previous = row.find('td', class_='previous')
                previous_val = previous.get_text(strip=True) if previous else ''
                
                if event_name and currency:
                    # Parse event time
                    event_time = self.parse_event_time(time_str, current_date) if time_str else None
                    
                    events.append({
                        'title': f"{currency} - {event_name}",
                        'content': f"Actual: {actual_val}, Forecast: {forecast_val}, Previous: {previous_val}",
                        'source': 'Forex Factory',
                        'url': 'https://www.forexfactory.com/calendar',
                        'published_at': datetime.now().isoformat(),
                        'currency': currency,
                        'event': event_name,
                        'event_time': event_time,
                        'event_time_str': time_str,
                        'impact': impact_level,
                        'actual': actual_val,
                        'forecast': forecast_val,
                        'previous': previous_val,
                        'keywords': self.extract_keywords(event_name)
                    })
            except Exception as e:
                continue
        
        return events
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using simple NLP"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Common financial keywords to look for
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
            'bullish', 'bearish', 'buy', 'sell', 'hold', 'upgrade', 'downgrade',
            'dividend', 'acquisition', 'merger', 'ipo', 'earnings beat',
            'earnings miss', 'guidance', 'forecast', 'analyst', 'price target',
            'inflation', 'federal reserve', 'interest rate', 'gdp', 'unemployment',
            'crypto', 'bitcoin', 'blockchain', 'fintech', 'artificial intelligence',
            'ai', 'machine learning', 'tech', 'semiconductor', 'supply chain', 'economy', 
            'inflation', 'unemployment', 'recession', 'revenue', 'profit', 'loss', 'growth',
            'decline', 'bullish', 'bearish', 'buy', 'sell', 'hold', 'upgrade', 'downgrade', 
            'dividend', 'acquisition', 'merger', 'ipo', 'earnings beat', 'earnings miss', 
            'guidance', 'forecast', 'analyst', 'price target', 'fed', 'federal reserve', 
            'interest rate', 'gdp', 'unemployment', 'crypto', 'bitcoin', 'blockchain',
            'fintech', 'artificial intelligence',
            'ai', 'machine learning', 'tech', 'semiconductor', 
            'supply chain', 'economy', 'inflation', 'unemployment', 'recession', 'revenue',
            'profit', 'loss', 'growth', 'decline', 'bullish', 'bearish', 'buy', 'sell', 
            'hold', 'upgrade', 'downgrade', 'dividend', 'acquisition', 'merger', 'ipo', 'earnings beat', 
            'earnings miss', 'guidance', 'forecast', 'analyst', 'price target',
        ]
        
        found_keywords = []
        for keyword in financial_keywords:
            if keyword in text:
                found_keywords.append(keyword)
        
        # Add any capitalized words (likely company names or tickers)
        words = text.split()
        for word in words:
            if len(word) > 1 and word.isupper():
                found_keywords.append(word)
        
        return list(set(found_keywords))  # Remove duplicates
    
    def collect_all_news(self, symbols: List[str] = ['AAPL', 'NVDA', 'TSLA']) -> pd.DataFrame:
        """Collect news from free sources (no API keys needed!)"""
        all_news = []
        
        print("Collecting news from all sources...")
        
        # Business Insider
        for symbol in symbols:
            symbol_lower = symbol.lower()
            print(f"Scraping Business Insider news for {symbol}...")
            news = self.scrape_businessinsider_news(symbol_lower)
            all_news.extend(news)
            time.sleep(2)
        
        # Forex Factory - Always scrape economic calendar
        print("Scraping Forex Factory calendar...")
        news = self.scrape_forexfactory_calendar()
        all_news.extend(news)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_news)
        
        if len(df) > 0:
            # Add metadata
            df['collected_at'] = datetime.now().isoformat()
            if 'content' in df.columns:
                df['content_length'] = df['content'].str.len()
        
        return df
    
    def collect_forex_news_and_data(self, hours_before: int = 1, hours_after: int = 1, 
                                   high_impact_only: bool = False) -> Dict:
        """
        Collect Forex Factory news and corresponding market data AROUND event times
        Shows price BEFORE and AFTER news to see actual impact!
        
        Args:
            hours_before: Hours before event to get data
            hours_after: Hours after event to get data
            high_impact_only: Only analyze high-impact events
            
        Returns:
            Dictionary with 'news', 'market_reactions', and summary data
        """
        print("\n" + "="*60)
        print("  FOREX NEWS & MARKET REACTION ANALYSIS")
        print("="*60 + "\n")
        
        if not self.td_client:
            print("❌ Twelve Data API not configured!")
            print("Please add your API key to config.ini")
            return {'news': pd.DataFrame(), 'market_reactions': pd.DataFrame()}
        
        # Get Forex Factory calendar
        print("📰 Collecting economic calendar from Forex Factory...")
        events = self.scrape_forexfactory_calendar()
        
        if not events:
            print("⚠ No news collected from Forex Factory")
            return {'news': pd.DataFrame(), 'market_reactions': pd.DataFrame()}
        
        news_df = pd.DataFrame(events)
        
        # Filter by impact if requested
        if high_impact_only:
            news_df = news_df[news_df['impact'] == 'high']
            print(f"📊 Filtered to {len(news_df)} high-impact events")
        else:
            print(f"📊 Found {len(news_df)} economic events")
        
        # Show summary
        if len(news_df) > 0:
            print(f"\nEvent breakdown:")
            print(f"  High impact: {sum(news_df['impact'] == 'high')}")
            print(f"  Medium impact: {sum(news_df['impact'] == 'medium')}")
            print(f"  Low impact: {sum(news_df['impact'] == 'low')}")
        
        # Analyze events with times
        all_reactions = []
        
        print(f"\n💹 Analyzing market reactions (±{hours_before}/{hours_after} hours around events)...\n")
        
        for idx, event in news_df.iterrows():
            if event['event_time'] is None:
                continue
            
            currency = event['currency']
            event_time = event['event_time']
            
            print(f"\n[{idx+1}] {event['title']}")
            print(f"    Time: {event_time.strftime('%Y-%m-%d %H:%M')} | Impact: {event['impact'].upper()}")
            
            # Get market data around this event time
            reactions = self.get_forex_data_around_time(
                [currency], 
                event_time,
                hours_before=hours_before,
                hours_after=hours_after
            )
            
            if not reactions.empty:
                # Add event info to reactions
                for col in ['title', 'event', 'impact', 'actual', 'forecast', 'previous']:
                    reactions[f'event_{col}'] = event[col]
                
                all_reactions.append(reactions)
            
            time.sleep(2)  # Rate limiting
        
        # Combine all reactions
        if all_reactions:
            market_df = pd.concat(all_reactions, ignore_index=True)
            
            print("\n" + "="*60)
            print("  SUMMARY")
            print("="*60)
            print(f"\n✓ Analyzed {len(market_df)} currency pair reactions")
            
            # Show biggest movers
            biggest_movers = market_df.nlargest(5, 'change_percent')
            print(f"\n📈 Biggest Positive Reactions:")
            for _, row in biggest_movers.iterrows():
                print(f"  🟢 {row['pair']}: {row['change_percent']:+.2f}% after {row['event_event']}")
            
            biggest_drops = market_df.nsmallest(5, 'change_percent')
            print(f"\n📉 Biggest Negative Reactions:")
            for _, row in biggest_drops.iterrows():
                print(f"  🔴 {row['pair']}: {row['change_percent']:+.2f}% after {row['event_event']}")
            
            # Most volatile
            most_volatile = market_df.nlargest(5, 'volatility_percent')
            print(f"\n⚡ Most Volatile Reactions:")
            for _, row in most_volatile.iterrows():
                print(f"  💥 {row['pair']}: {row['volatility_percent']:.2f}% volatility after {row['event_event']}")
            
            return {
                'news': news_df,
                'market_reactions': market_df,
                'summary': {
                    'total_events': len(news_df),
                    'analyzed_reactions': len(market_df),
                    'avg_change': market_df['change_percent'].mean(),
                    'avg_volatility': market_df['volatility_percent'].mean()
                }
            }
        else:
            print("\n⚠ No market data collected (events may be too old or in the future)")
            return {
                'news': news_df,
                'market_reactions': pd.DataFrame(),
                'summary': {}
            }
    
    def save_news(self, df: pd.DataFrame, filename: str = None):
        """Save news data to file"""
        if filename is None:
            filename = f"financial_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df.to_csv(filename, index=False)
        print(f"News saved to {filename}")
        
        # Also save as JSON for easy LLM consumption
        json_filename = filename.replace('.csv', '.json')
        df.to_json(json_filename, orient='records', indent=2)
        print(f"News also saved to {json_filename}")
    
    def prepare_for_llm(self, df: pd.DataFrame) -> str:
        """Prepare news data for LLM consumption"""
        # Sort by relevance (you can customize this)
        df_sorted = df.sort_values(['content_length'], ascending=False)
        
        # Create a formatted string for LLM
        llm_input = "Financial News Summary:\n\n"
        
        for idx, row in df_sorted.head(20).iterrows():  # Top 20 most relevant
            llm_input += f"Title: {row['title']}\n"
            llm_input += f"Source: {row['source']}\n"
            llm_input += f"Content: {row['content'][:500]}...\n"  # Truncate long content
            llm_input += f"Keywords: {', '.join(row.get('keywords', []))}\n"
            llm_input += f"URL: {row['url']}\n"
            llm_input += "-" * 50 + "\n\n"
        
        return llm_input

def create_config_file():
    """Create config.ini template"""
    config = configparser.ConfigParser()
    
    config['TWELVE_DATA'] = {
        'api_key': 'YOUR_TWELVE_DATA_API_KEY_HERE'
    }
    
    with open('config.ini', 'w') as f:
        config.write(f)
    
    print("✓ Created config.ini template")
    print("  Please add your Twelve Data API key!")
    print("  Get free key at: https://twelvedata.com/")

def main():
    """Main function - Forex news + market reaction analysis!"""
    print("\n" + "="*60)
    print("  NEWSBOT - FOREX NEWS & MARKET REACTIONS")
    print("="*60 + "\n")
    
    # Check for config file
    if not os.path.exists('config.ini'):
        print("⚠ config.ini not found!")
        create = input("Create config file template? (y/n): ").strip().lower()
        if create == 'y':
            create_config_file()
            print("\n👉 Edit config.ini with your Twelve Data API key, then run again!")
            return
        else:
            print("Cannot proceed without config file.")
            return
    
    # Initialize collector
    collector = FinancialDataCollector()
    
    if not collector.td_client:
        print("\n❌ Twelve Data API key not configured!")
        print("👉 Edit config.ini and add your API key")
        print("   Get free key at: https://twelvedata.com/")
        return
    
    print("Choose mode:")
    print("1. Collect stock news (Business Insider + Forex Factory)")
    print("2. Forex news + market reactions (⭐ RECOMMENDED - See before/after prices!)")
    print("3. High-impact events only (Faster, major news only)")
    
    choice = input("\nChoose (1-3): ").strip()
    
    if choice in ['2', '3']:
        # Forex reaction analysis
        high_impact_only = (choice == '3')
        
        # Get time window
        print("\nTime window for analysis:")
        hours_before = input("Hours before event (default 1): ").strip()
        hours_before = int(hours_before) if hours_before else 1
        
        hours_after = input("Hours after event (default 1): ").strip()
        hours_after = int(hours_after) if hours_after else 1
        
        result = collector.collect_forex_news_and_data(
            hours_before=hours_before,
            hours_after=hours_after,
            high_impact_only=high_impact_only
        )
        
        # Save news
        if len(result['news']) > 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            news_file = f"forex_news_{timestamp}.csv"
            result['news'].to_csv(news_file, index=False)
            print(f"\n💾 News saved to {news_file}")
        
        # Save market reactions
        if len(result['market_reactions']) > 0:
            reactions_file = f"forex_reactions_{timestamp}.csv"
            result['market_reactions'].to_csv(reactions_file, index=False)
            print(f"💾 Market reactions saved to {reactions_file}")
            
            # Save summary
            if result['summary']:
                print(f"\n📊 Overall Statistics:")
                print(f"  Events analyzed: {result['summary']['total_events']}")
                print(f"  Pair reactions: {result['summary']['analyzed_reactions']}")
                print(f"  Avg change: {result['summary']['avg_change']:.2f}%")
                print(f"  Avg volatility: {result['summary']['avg_volatility']:.2f}%")
        
    else:
        # Stock news collection (mode 1)
        symbols_input = input("\nEnter symbols (comma-separated, default: AAPL,NVDA,TSLA): ").strip()
        if symbols_input:
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
        else:
            symbols = ['AAPL', 'NVDA', 'TSLA']
        
        news_df = collector.collect_all_news(symbols)
        
        if len(news_df) > 0:
            print(f"\n✓ Collected {len(news_df)} news articles")
            if 'source' in news_df.columns:
                print(f"\nSources: {news_df['source'].value_counts().to_dict()}")
            
            # Save news
            collector.save_news(news_df)
            
            # Prepare for AI
            llm_input = collector.prepare_for_llm(news_df)
            
            with open('llm_input.txt', 'w', encoding='utf-8') as f:
                f.write(llm_input)
            
            print("✓ LLM input saved to llm_input.txt")
            
            print("\nSample news:")
            if 'title' in news_df.columns:
                for idx, row in news_df.head(5).iterrows():
                    print(f"  - {row['title'][:70]}...")
        else:
            print("\n⚠ No news collected.")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
