import asyncio
import time
from typing import Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from datetime import datetime

from utils.logger import logger
from config.enums import SymbolType, StrategyType
from database.manager import DatabaseManager
from connection.mt5_manager import ConnectionManager
from market.symbol_manager import SymbolManager
from market.data_provider import MarketDataProvider
from market.calendar_filter import CalendarFilter
from analysis.market_engine import MarketEngine
from risk.risk_manager import RiskManager
from strategy.strategy_manager import StrategyManager
from execution.order_executor import OrderExecutor
from ml.rl_engine import RLEngine
from reporting.performance_reporter import PerformanceReporter
from telegram_bot.bot import TelegramBot


class GoldManTradingBot:
    
    def __init__(self, telegram_token: str = None, mt5_login: int = None, 
                 mt5_password: str = None, mt5_server: str = None):
        self.db_manager = DatabaseManager()
        self.conn_mgr = ConnectionManager()
        self.symbol_mgr = None
        self.data_provider = None
        self.market_engine = None
        self.risk_manager = None
        self.strategy_manager = None
        self.order_executor = None
        self.rl_engine = None
        self.reporter = None
        self.telegram_bot = None
        self.calendar_filter = CalendarFilter()
        
        self.running = False
        self.current_symbol = None
        self.current_strategy = None
        self.main_loop_task = None
        self.monitoring_task = None

        self.telegram_token = telegram_token
        self.telegram_chat_id = None

        self.mt5_login = mt5_login
        self.mt5_password = mt5_password
        self.mt5_server = mt5_server
        
        self.last_price = None
        self.last_activity_log_time = None
        self.last_activity_time = None
    
    async def initialize(self):
        logger.info("Starting bot initialization...")
        
        if not self.conn_mgr.connect(self.mt5_login, self.mt5_password, self.mt5_server):
            raise Exception("Error connecting to MetaTrader 5")
        
        self.symbol_mgr = SymbolManager(self.conn_mgr)
        
        self.data_provider = MarketDataProvider(self.conn_mgr)
        
        self.market_engine = MarketEngine(self.data_provider, self.db_manager)
        
        self.risk_manager = RiskManager(self.conn_mgr, self.db_manager)
        
        self.strategy_manager = StrategyManager(
            self.data_provider, self.market_engine, self.risk_manager
        )
        
        self.order_executor = OrderExecutor(self.conn_mgr, self.db_manager)
        self.order_executor.set_data_provider(self.data_provider)
        
        self.rl_engine = RLEngine(self.db_manager)
        
        self.reporter = PerformanceReporter(self.db_manager)
        
        if self.telegram_token:
            try:
                self.telegram_bot = TelegramBot(self.telegram_token, self)
                await self.telegram_bot.start()
            except Exception as e:
                logger.error(f"Error initializing Telegram Bot: {e}")
                self.telegram_bot = None
        
        logger.info("Initialization completed successfully")
    
    async def start_trading(self, symbol: SymbolType, strategy: StrategyType):
        self.current_symbol = symbol
        self.current_strategy = strategy
        
        symbol_name = self.symbol_mgr.get_symbol_name(symbol)
        old_strategy = self.current_strategy.value if self.current_strategy else "None"
        self.strategy_manager.set_strategy(strategy, symbol_name)
        logger.info(f"[SENSITIVE] Strategy changed: Symbol={symbol_name}, OldStrategy={old_strategy}, NewStrategy={strategy.value}")
        
        if self.order_executor.has_open_position(symbol_name):
            logger.info("Open position exists. Managing position...")
            positions = self.order_executor.get_open_positions()
            for pos in positions:
                if pos.symbol == symbol_name:
                    self.order_executor.current_position = pos.ticket
                    break
        
        self.running = True
        self.main_loop_task = asyncio.create_task(self.main_loop())
        self.monitoring_task = asyncio.create_task(self.monitoring_loop())
        
        logger.info(f"Trading started: {symbol.value} - {strategy.value}")
        logger.info(f"[STARTUP] Main loop task created: {self.main_loop_task}, Monitoring task created: {self.monitoring_task}")
        logger.info(f"[STARTUP] Bot is now running. Waiting for market analysis...")
    
    async def main_loop(self):
        if self.current_strategy != StrategyType.SUPER_SCALP:
            logger.info("[MAIN_LOOP] Main trading loop started")
        loop_count = 0
        last_activity_check = time.time()
        while self.running:
            try:
                loop_count += 1
                
                if not self.conn_mgr.check_connection():
                    logger.error("[MAIN_LOOP] MT5 connection lost")
                    await asyncio.sleep(5)
                    continue
                
                if not self.calendar_filter.is_trading_time(self.current_symbol):
                    if self.current_strategy != StrategyType.SUPER_SCALP:
                        logger.info(f"[MAIN_LOOP] Not trading time for {self.current_symbol.value}")
                    if self.calendar_filter.should_switch_to_crypto(self.current_symbol):
                        if SymbolType.BTCUSD in self.symbol_mgr.get_active_symbols():
                            await self.switch_symbol(SymbolType.BTCUSD)
                    await asyncio.sleep(60)
                    continue
                
                symbol_name = self.symbol_mgr.get_symbol_name(self.current_symbol)
                if self.order_executor.has_open_position(symbol_name):
                    if self.order_executor.current_position:
                        self.order_executor.manage_position(
                            self.order_executor.current_position,
                            self.strategy_manager,
                            self.risk_manager,
                            self.market_engine
                        )
                    await asyncio.sleep(1)
                    continue
                
                if self.current_strategy == StrategyType.SUPER_SCALP:
                    if mt5 is None:
                        await asyncio.sleep(1)
                        continue
                    
                    try:
                        tick = mt5.symbol_info_tick(symbol_name)
                        if tick is None:
                            await asyncio.sleep(1)
                            continue
                        
                        symbol_info = mt5.symbol_info(symbol_name)
                        if symbol_info is None:
                            await asyncio.sleep(1)
                            continue
                        
                        current_price = tick.bid
                        price_threshold = symbol_info.point * 10
                        
                        if self.last_price is None:
                            self.last_price = current_price
                            signal = self.strategy_manager.analyze_market()
                        elif abs(current_price - self.last_price) >= price_threshold:
                            self.last_price = current_price
                            signal = self.strategy_manager.analyze_market()
                        else:
                            signal = None
                    except Exception as e:
                        logger.error(f"Error getting tick for {symbol_name}: {e}")
                        await asyncio.sleep(1)
                        continue
                else:
                    if not self.strategy_manager.should_analyze():
                        await asyncio.sleep(1)
                        continue
                    
                    signal = self.strategy_manager.analyze_market()
                
                if signal:
                    account_info = self.conn_mgr.get_account_info()
                    
                    if not self.risk_manager.check_daily_loss_limit():
                        logger.warning(f"[SENSITIVE] Daily loss limit exceeded - Trading stopped: Balance={account_info.balance:.2f}, DailyLoss={self.risk_manager.daily_loss:.2f}, DailyLossLimit={account_info.balance * 0.05:.2f}")
                        await asyncio.sleep(60)
                        continue
                    
                    drawdown_ok, volume_multiplier = self.risk_manager.check_drawdown()
                    if not drawdown_ok:
                        logger.warning(f"[SENSITIVE] Drawdown protection triggered - Trading stopped: Balance={account_info.balance:.2f}, Equity={account_info.equity:.2f}, Drawdown={self.risk_manager.max_drawdown*100:.2f}%")
                        await asyncio.sleep(60)
                        continue
                    
                    if volume_multiplier != 1.0:
                        old_lot_size = signal.lot_size
                        signal.lot_size *= volume_multiplier
                        logger.info(f"[SENSITIVE] Volume multiplier applied: OldLotSize={old_lot_size:.2f}, NewLotSize={signal.lot_size:.2f}, Multiplier={volume_multiplier:.2f}, Reason=Drawdown protection")
                    
                    logger.info(f"[ORDER_ENTRY] Pre-trade validation passed. Proceeding to execute trade.")
                    
                    ticket = self.order_executor.execute_order(signal)
                    
                    if ticket:
                        logger.info(f"[MAIN_LOOP] Trade opened successfully: Ticket {ticket}")
                        self.last_activity_time = time.time()
                        
                        params = self.rl_engine.get_parameters(signal.symbol, self.current_strategy.value)
                        
                        state = {
                            'symbol': signal.symbol,
                            'trend': 'UP' if signal.direction == 'BUY' else 'DOWN',
                            'entry_price': signal.entry_price,
                            'entry_points': signal.entry_points if hasattr(signal, 'entry_points') else [],
                            'trends': getattr(signal, 'trends', [])
                        }
                        action = {
                            'method': 'node',
                            'sl': signal.stop_loss,
                            'tp': signal.take_profit,
                            **{k: v for k, v in params.items() if k.startswith(('atr_', 'garch_', 'node_', 'min_rr', 'max_risk'))}
                        }
                        self.rl_engine.save_experience(
                            symbol=signal.symbol,
                            strategy=self.current_strategy.value,
                            timeframe=signal.timeframe.value,
                            state=state,
                            action=action,
                            reward=0.0,
                            trade_id=ticket
                        )
                else:
                    if self.current_strategy != StrategyType.SUPER_SCALP and loop_count % 10 == 0:
                        logger.info(f"[MAIN_LOOP] No valid signal generated. Waiting for next analysis cycle.")
                
                current_time = time.time()
                if current_time - last_activity_check >= 60:
                    if self.last_activity_time is None or current_time - self.last_activity_time >= 60:
                        logger.info(f"[MAIN_LOOP] No activity in the last minute. Bot is running and waiting for opportunities.")
                    last_activity_check = current_time
                
                if self.rl_engine.should_optimize(symbol_name, self.current_strategy.value):
                    logger.info("[MAIN_LOOP] Starting RL optimization...")
                    old_params = self.rl_engine.get_parameters(symbol_name, self.current_strategy.value)
                    logger.info(f"[SENSITIVE] Starting RL optimization: Symbol={symbol_name}, Strategy={self.current_strategy.value}")
                    new_params = self.rl_engine.optimize_all(symbol_name, self.current_strategy.value)
                    logger.info(f"[SENSITIVE] RL optimization completed: Symbol={symbol_name}, Strategy={self.current_strategy.value}, Updated {len(new_params)} parameters")
                    self.last_activity_time = time.time()
                
                if self.current_strategy == StrategyType.SUPER_SCALP:
                    await asyncio.sleep(1)
                elif self.current_strategy == StrategyType.SCALP:
                    await asyncio.sleep(5)
                else:
                    await asyncio.sleep(15)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def monitoring_loop(self):
        while self.running:
            try:
                positions = self.order_executor.get_open_positions()
                closed_positions = []
                
                cursor = self.db_manager.conn.cursor()
                cursor.execute("""
                    SELECT ticket FROM trades WHERE status = 'OPEN'
                """)
                open_tickets = [row[0] for row in cursor.fetchall()]
                
                for ticket in open_tickets:
                    if not any(p.ticket == ticket for p in positions):
                        closed_positions.append(ticket)
                
                if mt5:
                    for ticket in closed_positions:
                        position = mt5.history_deals_get(ticket=ticket)
                        if position:
                            deals = list(position)
                            if deals:
                                total_profit = sum(d.profit for d in deals)
                                account_info = self.conn_mgr.get_account_info()
                                logger.info(f"[SENSITIVE] Trade closed: Ticket={ticket}, TotalProfit={total_profit:.2f}, Balance={account_info.balance:.2f}, Equity={account_info.equity:.2f}")
                                if not self.db_manager.update_order(ticket, {
                                    'status': 'CLOSED',
                                    'exit_time': datetime.now(),
                                    'profit': total_profit
                                }):
                                    logger.warning(f"Failed to update trade {ticket} in database after monitoring detected closure")
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def switch_symbol(self, new_symbol: SymbolType):
        old_symbol_name = self.symbol_mgr.get_symbol_name(self.current_symbol)
        
        if self.order_executor.has_open_position(old_symbol_name):
            positions = self.order_executor.get_open_positions()
            for pos in positions:
                if pos.symbol == old_symbol_name:
                    self.order_executor.frozen_positions.append(pos.ticket)
                    logger.info(f"Position {pos.ticket} frozen")
        
        if self.symbol_mgr.switch_symbol(new_symbol):
            old_symbol = self.current_symbol.value if self.current_symbol else "None"
            self.current_symbol = new_symbol
            symbol_name = self.symbol_mgr.get_symbol_name(new_symbol)
            self.strategy_manager.set_strategy(self.current_strategy, symbol_name)
            logger.info(f"[SENSITIVE] Symbol switched: OldSymbol={old_symbol}, NewSymbol={new_symbol.value}, Strategy={self.current_strategy.value}")
    
    def is_running(self) -> bool:
        return self.running
    
    async def stop(self):
        logger.info("Stopping bot...")
        self.running = False
        
        try:
            if self.main_loop_task and not self.main_loop_task.done():
                self.main_loop_task.cancel()
                try:
                    await self.main_loop_task
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            logger.error(f"Error cancelling main loop: {e}")
        
        try:
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            logger.error(f"Error cancelling monitoring loop: {e}")
        
        try:
            if self.telegram_bot:
                await self.telegram_bot.stop()
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")
        
        try:
            self.conn_mgr.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting MT5: {e}")
        
        try:
            self.db_manager.close()
        except Exception as e:
            logger.error(f"Error closing database: {e}")
        
        logger.info("Bot stopped")
