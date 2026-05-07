import time
import logging
import numpy as np
import pandas as pd
import sys
import os
import csv
from datetime import datetime
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from execution.signal_generator import UnifiedSignalGenerator
from execution.trade_executor import MockOrderExecutor
from rl.agent import TFAgent
from rl.env import PortfolioEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_live_trader(tickers=['AAPL', 'MSFT', 'BTC-USD'], interval='1m', lookback=10):
    load_dotenv()
    
    # Override default tickers if dashboard user specified them
    env_assets = os.getenv("TRACKED_ASSETS")
    if env_assets:
        env_assets = env_assets.strip("'").strip('"')
        tickers = [t.strip() for t in env_assets.split(',')]
        
    logger.info(f"Initializing Real-Time Autonomous Live Trader for active assets: {tickers}")
    
    data_loader = DataLoader(tickers)
    signal_generator = UnifiedSignalGenerator(use_ml_weights=False) # Fallback equal weights or trained externally
    
    load_dotenv()
    mt5_login = os.getenv("MT5_LOGIN")
    mt5_pass = os.getenv("MT5_PASS")
    mt5_server = os.getenv("MT5_SERVER")
    
    if mt5_login and mt5_pass and mt5_server:
        from execution.trade_executor import MetaTraderExecutor
        logger.info("MT5 Credentials located! Building MetaTraderExecutor bridge.")
        executor = MetaTraderExecutor(login=mt5_login, password=mt5_pass, server=mt5_server)
        
        # Immediate fallback if the OS doesn't support the native MT5 terminal bridge compilation
        if not getattr(executor, 'connected', False):
            logger.error("Native MT5 connection failed. Throwing explicit error.")
            raise ConnectionError("CRITICAL: Failed to connect to MT5 account. Please ensure your demo/real credentials are correct and the MetaTrader5 terminal is active.")
    else:
        logger.error("No MT5 credentials in .env. Throwing explicit error.")
        raise ValueError("CRITICAL: MT5 credentials missing. Please set MT5_LOGIN, MT5_PASS, and MT5_SERVER in your .env file to run the engine.")
        
    initial_balance = getattr(executor, 'balance', 100000.0)
    
    # Initialize UI logger CSVs
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    port_csv = os.path.join(data_dir, "live_portfolio.csv")
    trades_csv = os.path.join(data_dir, "live_trades.csv")
    
    if not os.path.exists(port_csv):
        with open(port_csv, 'w') as f:
            f.write("Timestamp,Portfolio Value,Initial Balance\n")
    if not os.path.exists(trades_csv):
        with open(trades_csv, 'w') as f:
            f.write("Exit Time,Action,Ticker,Amount,Entry Price ($),Exit Price ($),Profit ($),Confidence (%)\n")
            
    holdings_csv = os.path.join(data_dir, "live_holdings.csv")
    if not os.path.exists(holdings_csv):
        with open(holdings_csv, 'w') as f:
            f.write("Asset,Weight\n")
    
    # We must instantiate an agent to handle inference.
    # Normally we load weights here via `agent.load_weights()`.
    # For now, it will predict with initialized weights as a working structural implementation.
    
    # We create a dummy environment identical exactly to live so the agent's TF graph shape sets.
    dummy_prices = pd.DataFrame(np.zeros((lookback, len(tickers))), columns=tickers)
    dummy_signals = pd.DataFrame(np.zeros((lookback, len(tickers))), columns=tickers)
    dummy_env = PortfolioEnv(dummy_prices, dummy_signals, lookback_window=lookback)
    agent = TFAgent(dummy_env)
    
    logger.info(f"Looping continuously. Checking market every 60 seconds for targets...")
    
    last_trade_times = {t: 0.0 for t in tickers}
    TRADE_COOLDOWN_SECONDS = 300
    
    try:
        while True:
            # 1. Fetch exactly the current live market window
            live_prices_df = data_loader.fetch_latest_window(lookback_window=lookback, interval=interval)
            
            if live_prices_df.empty or len(live_prices_df) < lookback:
                logger.warning(f"Insufficient live data points fetched. Need {lookback}. Skipping interval.")
                time.sleep(60)
                continue
                
            # 2. Fetch live predictions (Assuming we pass recent prices inside ML sub-models)
            signals_list: list[list[float]] = []
            sentiment_scores: list[float] = []
            latest_confidences: dict[str, float] = {}
            for _ in range(lookback):
                row_sigs: list[float] = []
                for ticker in tickers:
                    # Provide stable mock inputs when live models aren't active to prevent continuous trading thrashing
                    sen = 0.0
                    sentiment_scores.append(sen)
                    sig_dict = signal_generator.generate_signal(
                        ml_pred=0.0, 
                        dl_pred=0.0, 
                        sentiment_score=sen
                    )
                    sig = float(sig_dict['signal'])
                    row_sigs.append(sig)
                    latest_confidences[ticker] = float(sig_dict['confidence'])
                signals_list.append(row_sigs)
                
            sen_array = np.array(sentiment_scores)
            current_sentiment = float(np.mean(sen_array[-len(tickers):]))
            
            live_signals_df = pd.DataFrame(signals_list, columns=tickers)
            
            # 3. Concatenate current 10-min state for RL Agent Inference
            prices_window = live_prices_df.values
            current_signals = live_signals_df.iloc[-1].values
            
            state = np.concatenate((prices_window.flatten(), current_signals)).astype(np.float32)
            
            # 4. Agent decides strict target allocation!
            action_probs, _ = agent.predict(state, deterministic=True)
            
            # Strict normalization (ensure weights absolutely sum to 1)
            target_weights = action_probs / np.sum(action_probs)
            logger.info(f"RL Agent Target Portfolio Allocation: {dict(zip(tickers, np.round(target_weights, 3)))}")
            
            # 5. Execute mathematical rebalancing difference based on actual positions
            current_equity = float(executor.balance)
            current_holdings: dict[str, float] = dict(executor.portfolio)
            current_prices = live_prices_df.iloc[-1]
            prices_dict: dict[str, float] = current_prices.to_dict()
            
            # Total capital = Cash + Assets Value
            total_holdings_value = float(sum(
                float(amt) * float(prices_dict.get(t, 0.0))
                for t, amt in current_holdings.items()
            ))
            total_portfolio_value = current_equity + total_holdings_value
            
            # 10% Global Drawdown Stop Engine
            if total_portfolio_value <= 0.90 * float(initial_balance):
                logger.critical(f"10% GLOBAL DRAWDOWN REACHED! Portfolio value: ${total_portfolio_value:.2f}. Halting engine.")
                
                # Execute emergency square-off
                for tk, amt in list(current_holdings.items()):
                    if float(amt) > 1e-6 and tk in prices_dict:
                        res = executor.execute_trade('SELL', tk, float(amt), float(prices_dict[tk]))
                        if type(res) is dict and res.get('success'):
                            with open(trades_csv, 'a') as f:
                                f.write(f"{datetime.now()},SELL (Emergency Drawdown),{tk},{amt:.6f},{executor.avg_cost.get(tk, 0):.2f},{float(prices_dict[tk]):.2f},{res.get('profit', 0.0):.2f},100.0%\n")
                
                # Update UI one last time
                with open(port_csv, 'a') as f:
                    f.write(f"{datetime.now()},{executor.balance},{initial_balance}\n")
                sys.exit(0) # Stop the script completely
                
            # 3% Single Trade Stop Loss and 5% Lock-in Take Profit
            for tk, amt in list(current_holdings.items()):
                if float(amt) > 1e-6 and tk in prices_dict:
                    avg_cost = float(executor.avg_cost.get(tk, prices_dict[tk]))
                    unrealized_profit = (float(prices_dict[tk]) - avg_cost) * float(amt)
                    
                    if unrealized_profit <= -0.03 * float(initial_balance):
                        logger.warning(f"3% Single Trade SL breached for {tk}. Realizing loss.")
                        res = executor.execute_trade('SELL', tk, float(amt), float(prices_dict[tk]))
                        if type(res) is dict and res.get('success'):
                            with open(trades_csv, 'a') as f:
                                f.write(f"{datetime.now()},SELL (Square Off - SL),{tk},{amt:.6f},{avg_cost:.2f},{float(prices_dict[tk]):.2f},{res.get('profit', 0.0):.2f},100.0%\n")
                            # Update current state explicitly so rebalancing logic doesn't rebuy immediately with old numbers
                            current_holdings[tk] = 0.0
                            current_equity += float(amt) * float(prices_dict[tk])
                            
                    elif unrealized_profit >= 0.05 * float(initial_balance):
                        logger.warning(f"5% Single Trade TP breached for {tk} (${unrealized_profit:.2f}). Securing floating profit!")
                        res = executor.execute_trade('SELL', tk, float(amt), float(prices_dict[tk]))
                        if type(res) is dict and res.get('success'):
                            with open(trades_csv, 'a') as f:
                                f.write(f"{datetime.now()},SELL (Square Off - TP),{tk},{amt:.6f},{avg_cost:.2f},{float(prices_dict[tk]):.2f},{res.get('profit', 0.0):.2f},100.0%\n")
                            current_holdings[tk] = 0.0
                            current_equity += float(amt) * float(prices_dict[tk])
            
            # Calculate needed buys and sells for the Delta
            for i, ticker in enumerate(tickers):
                if ticker not in prices_dict: 
                    continue # Skip if no price data arrived 

                target_value = float(target_weights[i]) * total_portfolio_value
                current_value = float(current_holdings.get(ticker, 0.0)) * float(prices_dict[ticker])
                delta_value = target_value - current_value
                
                # Dynamic Threshold: 0.5% of total portfolio value to prevent fractional micro-trading
                min_trade_threshold = max(50.0, total_portfolio_value * 0.005)
                
                if abs(delta_value) > min_trade_threshold:
                    current_time = time.time()
                    time_since_last = current_time - last_trade_times.get(ticker, 0.0)
                    
                    if time_since_last < TRADE_COOLDOWN_SECONDS:
                        logger.info(f"{ticker} is in cooldown for {int(TRADE_COOLDOWN_SECONDS - time_since_last)}s. Examining other assets.")
                        continue
                        
                    # Calculate predicted price bound based on recent volatility (SMA & StdDev)
                    if ticker in live_prices_df.columns:
                        recent_prices = live_prices_df[ticker].dropna()
                    else:
                        recent_prices = pd.Series([prices_dict[ticker]])
                        
                    if len(recent_prices) >= 2:
                        sma = float(recent_prices.mean())
                        std_dev = float(recent_prices.std())
                    else:
                        sma = float(prices_dict[ticker])
                        std_dev = 0.0
                        
                    # We want to buy at/below SMA value and sell at/above SMA value (organic target entries)
                    buy_bound = sma - (0.1 * std_dev)
                    sell_bound = sma + (0.1 * std_dev)
                    entry_price_val = float(prices_dict[ticker])
                    
                    if delta_value > min_trade_threshold:
                        if entry_price_val > buy_bound:
                            logger.info(f"Waiting for {ticker} accurate entry. Current ${entry_price_val:.2f} > Bound ${buy_bound:.2f}")
                            continue

                        # Contract Sizing: Account for margin/available balance
                        safe_allocation = min(delta_value, current_equity * 0.95) # Leave a 5% margin buffer
                        amount_to_buy = safe_allocation / entry_price_val
                        amount_to_buy = float(f"{amount_to_buy:.4f}")
                        
                        if amount_to_buy > 0.0001:
                            # Predefined exiting: 3% Stop Loss and 6% Take Profit attached to every order
                            sl_price = float(f"{entry_price_val * 0.97:.2f}")
                            tp_price = float(f"{entry_price_val * 1.06:.2f}")
                            res = executor.execute_trade('BUY', ticker, amount_to_buy, entry_price_val, sl=sl_price, tp=tp_price)
                            if type(res) is dict and res.get('success'):
                                confidence = float(latest_confidences.get(ticker, 0.0)) * 100
                                with open(trades_csv, 'a') as f:
                                    f.write(f"{datetime.now()},BUY,{ticker},{amount_to_buy:.6f},{res.get('entry_price', prices_dict[ticker]):.2f},,0.00,{confidence:.1f}%\n")
                                last_trade_times[ticker] = current_time
                                    
                    elif delta_value < -min_trade_threshold:
                        amount_to_sell = abs(delta_value) / entry_price_val
                        amount_to_sell = float(f"{amount_to_sell:.4f}")
                        
                        if amount_to_sell > 0.0001:
                            res = executor.execute_trade('SELL', ticker, amount_to_sell, entry_price_val)
                            if type(res) is dict and res.get('success'):
                                confidence = float(latest_confidences.get(ticker, 0.0)) * 100
                                with open(trades_csv, 'a') as f:
                                    f.write(f"{datetime.now()},SELL (Square Off),{ticker},{amount_to_sell:.6f},{res.get('entry_price', 0):.2f},{entry_price_val:.2f},{res.get('profit', 0.0):.2f},{confidence:.1f}%\n")
                                last_trade_times[ticker] = current_time
                            
            # Continuously update the live portfolio log so the UI graph moves
            with open(port_csv, 'a') as f:
                f.write(f"{datetime.now()},{total_portfolio_value},{initial_balance}\n")
                
            # Log exact dollar holdings dynamically for the Dashboard pie chart
            with open(holdings_csv, 'w') as f:
                f.write("Asset,Weight\n")
                if total_portfolio_value > 0:
                    f.write(f"CASH,{current_equity / total_portfolio_value:.4f}\n")
                for tk, amt in current_holdings.items():
                    if tk in prices_dict and total_portfolio_value > 0:
                        tk_val = float(amt) * float(prices_dict[tk])
                        f.write(f"{tk},{tk_val / total_portfolio_value:.4f}\n")
                        
            # Dump internal non-portfolio metrics for dashboard syncing
            metrics_csv = os.path.join(data_dir, "live_metrics.csv")
            with open(metrics_csv, 'w') as f:
                f.write("Metric,Value\n")
                f.write(f"Sentiment,{current_sentiment:.4f}\n")
                f.write(f"Weight_ML,{signal_generator.weights.get('ml', 0.33):.4f}\n")
                f.write(f"Weight_DL,{signal_generator.weights.get('dl', 0.33):.4f}\n")
                f.write(f"Weight_Sentiment,{signal_generator.weights.get('sentiment', 0.34):.4f}\n")
                    
            logger.info(f"Actual Portfolio Gross Value: ${total_portfolio_value:.2f}")
            logger.info("Loop 1 completed. Awaiting next cycle in 60 seconds...")
            time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Live execution loop halted intentionally by user.")

if __name__ == "__main__":
    run_live_trader()
