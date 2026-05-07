import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockOrderExecutor:
    """
    Simulated order execution engine that logs trades and updates portfolio balances locally.
    """
    def __init__(self, initial_balance=100000.0):
        self.balance = initial_balance
        self.portfolio: Dict[str, float] = {}
        self.avg_cost: Dict[str, float] = {} # Tracks the rolling average entry price
        logger.info(f"Mock Order Executor initialized. Balance: ${self.balance:.2f}")

    def execute_trade(self, action: str, ticker: str, amount: float, price: float, sl: float = 0.0, tp: float = 0.0) -> dict:
        cost = amount * price
        
        if action == 'BUY':
            if cost <= self.balance:
                self.balance -= cost
                
                # Calculate rolling average entry cost basis
                current_amount = self.portfolio.get(ticker, 0)
                current_avg = self.avg_cost.get(ticker, 0)
                new_total = current_amount + amount
                if new_total > 0:
                    self.avg_cost[ticker] = ((current_avg * current_amount) + (price * amount)) / new_total
                    
                self.portfolio[ticker] = new_total
                logger.info(f"EXECUTED BUY: {amount:.2f} {ticker} @ ${price:.2f} | Remaining Balance: ${self.balance:.2f}")
                return {'success': True, 'profit': 0.0, 'entry_price': price}
            else:
                logger.warning(f"FAILED BUY: Insufficient funds for {ticker}.")
                return {'success': False}
                
        elif action == 'SELL':
            if self.portfolio.get(ticker, 0) >= amount:
                self.balance += cost
                self.portfolio[ticker] -= amount
                
                # Calculate True Realized Profit based on average entry price
                entry_price = self.avg_cost.get(ticker, price)
                profit = (price - entry_price) * amount
                
                # Cleanup if squared off completely
                if self.portfolio[ticker] < 1e-6:
                    self.portfolio[ticker] = 0.0
                    self.avg_cost[ticker] = 0.0
                    
                logger.info(f"EXECUTED SELL: {amount:.2f} {ticker} @ ${price:.2f} | Profit: ${profit:.2f} | Remaining Balance: ${self.balance:.2f}")
                return {'success': True, 'profit': profit, 'entry_price': entry_price}
            else:
                logger.warning(f"FAILED SELL: Insufficient {ticker} holding {self.portfolio.get(ticker, 0)}.")
                return {'success': False}
        
        return {'success': False}

class MetaTraderExecutor:
    """
    Real execution bridge connecting to MetaTrader 5 terminal.
    NOTE: MetaTrader5 native Python library is Windows-only. 
    It will fail gracefully on MacOS returning False for trades.
    """
    def __init__(self, login, password, server):
        self.login = int(login)
        self.password = password
        self.server = server
        self.connected = False
        self._balance = 100000.0  # Fallback for local tracking
        self.portfolio: Dict[str, float] = {}
        self.avg_cost: Dict[str, float] = {}
        
        try:
            import MetaTrader5 as mt5
            logger.info("Initializing MetaTrader 5...")
            if not mt5.initialize(login=self.login, password=self.password, server=self.server):
                logger.error(f"MT5 Init Failed. Likely wrong credentials or operating system mismatch. Error: {mt5.last_error()}")
            else:
                self.connected = True
                
                # Try to sync real balance from MT5 account
                account_info = mt5.account_info()
                if account_info:
                    logger.info(f"MT5 Connected! Real Balance Synced: ${account_info.balance:.2f}")
                    if not account_info.trade_allowed:
                        logger.critical("MT5 initialized successfully, BUT Account Trading is DISABLED (trade_allowed=False). Did you use the Investor Password? Please use the Master Password in .env to place live trades. Trades will be REJECTED (10017).")
                else:
                    logger.warning("Could not retrieve account info.")
                    
        except ImportError:
            logger.error("MetaTrader5 package is not installed (or not supported on this OS). Cannot connect.")

    @property
    def balance(self):
        if self.connected:
            import MetaTrader5 as mt5
            acc = mt5.account_info()
            if acc:
                return acc.margin_free  # Dynamically return Free Margin for maximum loop leveraging
        return self._balance

    @balance.setter
    def balance(self, value):
        self._balance = value

    def execute_trade(self, action: str, ticker: str, amount: float, price: float, sl: float = 0.0, tp: float = 0.0) -> dict:
        if not self.connected:
            logger.error("MT5 not connected. Trade simulated locally only.")
            return {'success': False}
            
        import MetaTrader5 as mt5
        
        # Format mapping: yfinance tickers to MT5 symbols
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
        
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Symbol {symbol} not available in your MT5 broker.")
            return {'success': False}
            
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not available to get info from broker.")
            return {'success': False}
            
        acc = mt5.account_info()
        leverage = acc.leverage if acc else 100.0
        
        step = symbol_info.volume_step
        step_str = str(float(step))
        decimals = len(step_str.split('.')[1]) if '.' in step_str else 0
        
        # Safely fall back through filling modes to eliminate 'Unsupported filling mode' errors
        filling_modes = [
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_FOK,
            mt5.ORDER_FILLING_RETURN
        ]
        
        # Prioritize based on broker's symbol info if available
        if symbol_info.filling_mode & 1:  # SYMBOL_FILLING_FOK
            if mt5.ORDER_FILLING_FOK in filling_modes:
                filling_modes.insert(0, filling_modes.pop(filling_modes.index(mt5.ORDER_FILLING_FOK)))
        elif symbol_info.filling_mode & 2:  # SYMBOL_FILLING_IOC
            if mt5.ORDER_FILLING_IOC in filling_modes:
                filling_modes.insert(0, filling_modes.pop(filling_modes.index(mt5.ORDER_FILLING_IOC)))

        intended_usd = amount * price
        requests_to_send = []
        
        if action == 'SELL':
            current_abstract = self.portfolio.get(ticker, 0)
            if current_abstract > 1e-6:
                fraction_to_sell = amount / current_abstract
                if fraction_to_sell >= 0.99:
                    fraction_to_sell = 1.0 # Square off completely
                
                positions = mt5.positions_get(symbol=symbol)
                if positions and len(positions) > 0:
                    target_mt5_volume = sum(p.volume for p in positions) * fraction_to_sell
                    remaining_to_close = target_mt5_volume
                    
                    for p in positions:
                        if remaining_to_close <= 1e-8:
                            break
                        v_to_close = min(p.volume, remaining_to_close)
                        v_to_close_rounded = round(round(v_to_close / step) * step, decimals)
                        
                        if v_to_close_rounded < symbol_info.volume_min:
                            continue
                            
                        opposite_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                        close_price = mt5.symbol_info_tick(symbol).bid if opposite_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask

                        req = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": symbol,
                            "volume": float(v_to_close_rounded),
                            "type": opposite_type,
                            "position": p.ticket,
                            "price": close_price,
                            "deviation": 20,
                            "magic": 234000,
                            "comment": "Autonomous AI Square-Off",
                            "type_time": mt5.ORDER_TIME_GTC,
                        }
                        requests_to_send.append(req)
                        remaining_to_close -= v_to_close
            
            if not requests_to_send:
                logger.warning(f"SELL execution for {ticker} triggered natively but no matching positions found or volume too low.")
                return {'success': False}
                
        else:
            # BUY
            margin_1_lot = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, 1.0, price)
            if margin_1_lot and margin_1_lot > 0:
                notional_1_lot = margin_1_lot * leverage
                mt5_volume = intended_usd / notional_1_lot
            else:
                contract_size = getattr(symbol_info, 'trade_contract_size', 100000)
                mt5_volume = amount / (contract_size if contract_size > 1 else 1)
                
            mt5_volume = round(round(mt5_volume / step) * step, decimals)
            
            if mt5_volume < symbol_info.volume_min:
                logger.warning(f"Calculated mt5 trade volume {mt5_volume} is below the broker minimum {symbol_info.volume_min} for {symbol}. Trade aborted.")
                return {'success': False}
                
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(mt5_volume),
                "type": mt5.ORDER_TYPE_BUY,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "Autonomous AI Trade",
                "type_time": mt5.ORDER_TIME_GTC,
            }
            if sl > 0:
                req["sl"] = float(sl)
            if tp > 0:
                req["tp"] = float(tp)
            requests_to_send.append(req)

        results = []
        for request in requests_to_send:
            result = None
            for fill_mode in filling_modes:
                request["type_filling"] = fill_mode
                result = mt5.order_send(request)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    break
                    
                if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
                    continue
                    
                break
                
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"MT5 Trade Rejected: retcode={result.retcode if result else 'None'}, comment={result.comment if result else 'None'}, req={request}")
                return {'success': False}
            
            results.append(result)

        logger.info(f"MT5 LIVE {action} Executed locally. {len(results)} native orders placed. (Total nominal abstract logic amount {amount} {symbol})")
        
        # Keep local representation updated
        cost = amount * price
        
        if action == 'BUY':
            self.balance -= cost
            
            current_amount = self.portfolio.get(ticker, 0)
            current_avg = self.avg_cost.get(ticker, 0)
            new_total = current_amount + amount
            if new_total > 0:
                self.avg_cost[ticker] = ((current_avg * current_amount) + (price * amount)) / new_total
                
            self.portfolio[ticker] = new_total
            return {'success': True, 'profit': 0.0, 'entry_price': price}
            
        else:
            self.balance += cost
            self.portfolio[ticker] -= amount
            
            entry_price = self.avg_cost.get(ticker, price)
            profit = (price - entry_price) * amount
            
            if self.portfolio[ticker] < 1e-6:
                self.portfolio[ticker] = 0.0
                self.avg_cost[ticker] = 0.0
                
            return {'success': True, 'profit': profit, 'entry_price': entry_price}
