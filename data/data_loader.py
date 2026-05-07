import yfinance as yf
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, tickers):
        """
        Initialize DataLoader with a list of tickers.
        Supports stocks ('AAPL'), Crypto ('BTC-USD'), Forex ('EURUSD=X').
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        self.tickers = tickers

    def fetch_historical_data(self, start_date, end_date, interval='1d'):
        """
        Fetch historical data for the initialized tickers.
        Intervals can be: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        """
        logger.info(f"Fetching historical data for {self.tickers} from {start_date} to {end_date} (interval: {interval})")
        data = yf.download(self.tickers, start=start_date, end=end_date, interval=interval, progress=False)
        
        # Determine if multi-level columns exist (happens when >1 ticker)
        if len(self.tickers) > 1:
            # We can optionally stack or format it. Default yfinance behavior 
            # returns a MultiIndex column DataFrame (e.g. data['Close']['AAPL'])
            pass
            
        return data

    def fetch_real_time_mock(self, interval='1m'):
        """
        Mock real-time data fetching. 
        In production, this would connect to a WebSocket (e.g., Binance, Alpaca).
        """
        logger.info(f"Fetching real-time mock data for {self.tickers}")
        # Fetch the very last available period
        data = yf.download(self.tickers, period='1d', interval=interval, progress=False)
        if not data.empty:
            # Return the last row as the "real-time" tick
            return data.iloc[[-1]]
        return pd.DataFrame()

    def fetch_latest_window(self, lookback_window: int = 10, interval: str = '1m') -> pd.DataFrame:
        """
        Fetches the exact recent N periods of real-time data explicitly for RL state formulation.
        Provides a complete streaming lookback window identical to what is used during backtest training.
        """
        logger.info(f"Streaming latest {lookback_window} periods of {interval} data for {self.tickers}...")
        
        # 1. ATTEMPT NATIVE METATRADER 5 DATA SOURCING
        import os
        from dotenv import load_dotenv
        load_dotenv()
        mt5_login, mt5_pass, mt5_server = os.getenv("MT5_LOGIN"), os.getenv("MT5_PASS"), os.getenv("MT5_SERVER")
        
        if mt5_login and mt5_pass and mt5_server:
            try:
                import MetaTrader5 as mt5
                if mt5.initialize(login=int(mt5_login), password=mt5_pass, server=mt5_server):
                    df_dict = {}
                    tf_map = {'1m': mt5.TIMEFRAME_M1, '5m': mt5.TIMEFRAME_M5, '15m': mt5.TIMEFRAME_M15, '1h': mt5.TIMEFRAME_H1, '1d': mt5.TIMEFRAME_D1}
                    mt5_tf = tf_map.get(interval, mt5.TIMEFRAME_M1)
                    
                    for ticker in self.tickers:
                        # Ensure robust mapping from env strings to MT5 symbols
                        symbol_map = {
                            'GC=F': 'XAUUSD',
                            'SI=F': 'XAGUSD',
                            'BTC-USD': 'BTCUSD',
                            'XAU-USD': 'XAUUSD',
                            'XAG-USD': 'XAGUSD',
                            'XAUUSD=X': 'XAUUSD',
                            'XAGUSD=X': 'XAGUSD'
                        }
                        symbol = symbol_map.get(ticker, ticker.replace("-", "").replace("=X", ""))
                        
                        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, lookback_window)
                        
                        if rates is not None and len(rates) > 0:
                            df_dict[ticker] = [r[4] for r in rates] # index 4 is the 'close' price
                        else:
                            logger.error(f"MT5 returned no data for {symbol}.")
                    
                    if len(df_dict) == len(self.tickers):
                        final_df = pd.DataFrame(df_dict)
                        if len(final_df) >= lookback_window:
                            logger.info("Successfully fetched live streaming data from native MT5 terminal.")
                            return final_df
            except ImportError:
                logger.warning("MetaTrader5 package missing. Falling back to yfinance.")
            except Exception as e:
                logger.warning(f"MT5 Native Data sourcing failed: {e}. Falling back to yfinance.")
                
        # 2. FALLBACK TO YFINANCE
        # '5d' period ensures enough data points are grabbed even across low volume or weekend hours
        try:
            data = yf.download(self.tickers, period='5d', interval=interval, progress=False)
            
            # YFinance heavily utilizes multi-index blocks when >1 ticker is provided
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.levels[0]:
                    data = data['Close']
            else:
                if 'Close' in data.columns:
                    data = data[['Close']]
                    data.columns = self.tickers
                    
            # Return specifically the final tail end rows
            # Use ffill and bfill to prevent misaligned 1m timestamps between multiple futures from dropping rows
            latest_rows = data.ffill().bfill().dropna(axis=0, how='any')
            if len(latest_rows) >= lookback_window:
                return latest_rows.iloc[-lookback_window:]
            else:
                logger.warning(f"Live yfinance window too sparse. Found {len(latest_rows)} rows cleanly.")
                return latest_rows
        except Exception as e:
            logger.error(f"Live API Fetch Failure: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    loader = DataLoader(['AAPL', 'BTC-USD'])
    df = loader.fetch_historical_data(start_date='2025-01-01', end_date='2025-12-31')
    print("Historical Data Sample:")
    print(df.head())
    
    real_time_df = loader.fetch_real_time_mock()
    print("\nReal-time Mock Data:")
    print(real_time_df)
