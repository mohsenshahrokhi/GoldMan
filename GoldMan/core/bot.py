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


class GoldManBot:
    
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
        self.max_profit_tracker = {}  # Track max profit for each open position
        self.open_orders_max_profit = {}  # Track max profit for each open order
    
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
        self.order_executor.main_controller = self
        
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
    
    async def start_operating(self, symbol: SymbolType, strategy: StrategyType):
        self.current_symbol = symbol
        self.current_strategy = strategy
        
        symbol_name = self.symbol_mgr.get_symbol_name(symbol)
        old_strategy = self.current_strategy.value if self.current_strategy else "None"
        self.strategy_manager.set_strategy(strategy, symbol_name)
        logger.info(f"[SENSITIVE] Strategy changed: Symbol={symbol_name}, OldStrategy={old_strategy}, NewStrategy={strategy.value}")
        
        all_positions = self.order_executor.get_open_positions()
        if all_positions:
            logger.info(f"Found {len(all_positions)} existing open position(s). Initializing management...")
            for pos in all_positions:
                if pos.ticket not in self.order_executor.original_sl_tp:
                    self.order_executor.original_sl_tp[pos.ticket] = {
                        'sl': pos.sl,
                        'tp': pos.tp
                    }
                    logger.info(f"[SENSITIVE] Initialized SL/TP protection for existing position: Ticket={pos.ticket}, Symbol={pos.symbol}, SL={pos.sl:.5f}, TP={pos.tp:.5f}")
        
        if self.order_executor.has_open_position(symbol_name):
            logger.info("Open position exists for current symbol. Managing position...")
            for pos in all_positions:
                if pos.symbol == symbol_name:
                    self.order_executor.current_position = pos.ticket
                    break
        
        self.running = True
        self.main_loop_task = asyncio.create_task(self.main_loop())
        self.monitoring_task = asyncio.create_task(self.monitoring_loop())
        
        logger.info(f"Operation started: {symbol.value} - {strategy.value}")
        
        if self.telegram_bot:
            for chat_id in self.telegram_bot.chat_ids:
                try:
                    await self.telegram_bot.send_status_message(chat_id, is_start=True)
                except Exception as e:
                    logger.error(f"Error sending start status message: {e}")
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
                all_positions = self.order_executor.get_open_positions()
                
                if all_positions:
                    managed_any = False
                    for position in all_positions:
                        if position.symbol == symbol_name:
                            try:
                                if self.current_strategy == StrategyType.SUPER_SCALP:
                                    logger.info(f"[MAIN_LOOP] Managing position: Ticket={position.ticket}, Symbol={position.symbol}, Type={'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'}, Entry={position.price_open:.5f}, Current={position.price_current:.5f}")
                                    result = self.order_executor.manage_position(
                                        position.ticket,
                                        self.strategy_manager,
                                        self.risk_manager,
                                        self.market_engine
                                    )
                                    if result:
                                        managed_any = True
                                        logger.info(f"[MAIN_LOOP] Position {position.ticket} managed successfully")

                                        # اطلاع‌رسانی مدیریت موقعیت
                                        if self.telegram_bot:
                                            try:
                                                account_info = self.conn_mgr.get_account_info()
                                                message = f"""🔄 <b>Position Managed</b>

📊 <b>Position Details:</b>
• Ticket: {position.ticket}
• Symbol: {position.symbol}
• Direction: {'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'}
• Entry: {position.price_open:.5f}
• Current: {position.price_current:.5f}
• SL: {position.sl:.5f}
• TP: {position.tp:.5f}
• Volume: {position.volume:.2f}

💰 <b>Account:</b>
• Balance: ${account_info.balance:.2f}
• Equity: ${account_info.equity:.2f}"""
                                                await self.telegram_bot.send_notification(message)
                                                logger.info(f"[TELEGRAM] Position managed notification sent for ticket {position.ticket}")
                                            except Exception as e:
                                                logger.error(f"[TELEGRAM] Error sending position managed notification: {e}")

                                    else:
                                        logger.warning(f"[MAIN_LOOP] Position {position.ticket} management returned False")
                                elif position.ticket == self.order_executor.current_position:
                                    result = self.order_executor.manage_position(
                                        position.ticket,
                                        self.strategy_manager,
                                        self.risk_manager,
                                        self.market_engine
                                    )
                                    if result:
                                        managed_any = True
                            except Exception as e:
                                logger.error(f"Error managing position {position.ticket} in main loop: {e}", exc_info=True)
                    
                    if managed_any or all_positions:
                        await asyncio.sleep(1)
                        continue
                    else:
                        logger.warning(f"[MAIN_LOOP] No positions managed. All positions: {[(p.ticket, p.symbol) for p in all_positions]}, Current symbol: {symbol_name}, Strategy: {self.current_strategy}")
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
                            logger.debug(f"[SUPER_SCALP] Initial price set: {current_price:.5f}, threshold: {price_threshold:.5f}")
                            signal = self.strategy_manager.analyze_market()
                            if signal is None:
                                logger.debug(f"[SUPER_SCALP] Analysis completed - No signal generated")
                        elif abs(current_price - self.last_price) >= price_threshold:
                            price_change = current_price - self.last_price
                            logger.debug(f"[SUPER_SCALP] Price change detected: {price_change:.5f} (threshold: {price_threshold:.5f})")
                            self.last_price = current_price
                            signal = self.strategy_manager.analyze_market()
                            if signal is None:
                                logger.debug(f"[SUPER_SCALP] Analysis completed - No signal generated")
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
                        if self.current_strategy == StrategyType.SUPER_SCALP:
                            logger.warning(f"[SENSITIVE] Drawdown protection triggered but Super Scalp continues: Balance={account_info.balance:.2f}, Equity={account_info.equity:.2f}, Drawdown={self.risk_manager.max_drawdown*100:.2f}%")
                            volume_multiplier = 0.3
                        else:
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
                        self.open_orders_max_profit[ticket] = 0.0
                        
                        if self.current_strategy == StrategyType.SUPER_SCALP:
                            all_positions = self.order_executor.get_open_positions()
                            for pos in all_positions:
                                if pos.ticket == ticket:
                                    logger.info(f"[MAIN_LOOP] Immediately managing new position: Ticket={ticket}, Symbol={pos.symbol}")
                                    try:
                                        self.order_executor.manage_position(
                                            ticket,
                                            self.strategy_manager,
                                            self.risk_manager,
                                            self.market_engine
                                        )
                                    except Exception as e:
                                        logger.error(f"Error managing new position {ticket}: {e}", exc_info=True)
                                    break
                        
                        if self.current_strategy == StrategyType.SUPER_SCALP and signal.entry_points and signal.trends and signal.timeframes:
                            logger.info("=" * 60)
                            logger.info("[SUPER_SCALP] Trade Analysis Details:")
                            logger.info(f"  Final Entry Price: {signal.entry_price:.5f}")
                            logger.info("  Entry Points by Timeframe:")
                            for i, (tf, entry) in enumerate(zip(signal.timeframes, signal.entry_points)):
                                logger.info(f"    {tf}: {entry:.5f}")
                            logger.info("  Trends (First 3 Timeframes):")
                            for i, (tf, trend) in enumerate(zip(signal.timeframes[:3], signal.trends)):
                                logger.info(f"    {tf}: {trend}")
                            logger.info("=" * 60)
                        
                        if self.telegram_bot:
                            try:
                                account_info = self.conn_mgr.get_account_info()
                                message = f"""✅ <b>Order Opened</b>

📊 <b>Order Details:</b>
• Ticket: {ticket}
• Symbol: {signal.symbol}
• Direction: {signal.direction}
• Entry: {signal.entry_price:.5f}
• SL: {signal.stop_loss:.5f}
• TP: {signal.take_profit:.5f}
• Lot Size: {signal.lot_size:.2f}
• R/R: {signal.confidence:.2f}

💰 <b>Account:</b>
• Balance: ${account_info.balance:.2f}
• Equity: ${account_info.equity:.2f}
• Free Margin: ${getattr(account_info, 'free_margin', account_info.equity - account_info.margin):.2f}"""
                                await self.telegram_bot.send_notification(message)
                                logger.info(f"[TELEGRAM] Order opened notification sent for ticket {ticket}")
                            except Exception as e:
                                logger.error(f"[TELEGRAM] Error sending order opened notification: {e}")
                        else:
                            logger.warning(f"[TELEGRAM] Telegram bot not available, skipping order opened notification for ticket {ticket}")
                        
                        params = self.rl_engine.get_parameters(signal.symbol, self.current_strategy.value)
                        entry_weights = self.rl_engine.get_entry_weights(signal.symbol, self.current_strategy.value)
                        trend_weights = self.rl_engine.get_trend_weights(signal.symbol, self.current_strategy.value)
                        weights = self.rl_engine.get_weights(signal.symbol, self.current_strategy.value)
                        
                        state = {
                            'symbol': signal.symbol,
                            'trend': 'UP' if signal.direction == 'BUY' else 'DOWN',
                            'entry_price': signal.entry_price,
                            'entry_points': signal.entry_points if hasattr(signal, 'entry_points') else [],
                            'trends': getattr(signal, 'trends', []),
                            'timeframes': getattr(signal, 'timeframes', []),
                            'trend_strengths': getattr(signal, 'trend_strengths', []) if hasattr(signal, 'trend_strengths') else [],
                            'technical_signal': getattr(signal, 'technical_signal', 'NEUTRAL')
                        }
                        
                        sl_distance = abs(signal.entry_price - signal.stop_loss) if signal.stop_loss else 0
                        tp_distance = abs(signal.take_profit - signal.entry_price) if signal.take_profit else 0
                        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
                        
                        action = {
                            'method': weights.get('node', 0.25) > 0.5 and 'node' or (weights.get('atr', 0.25) > 0.5 and 'atr' or (weights.get('garch', 0.25) > 0.5 and 'garch' or 'fixed_rr')),
                            'sl': signal.stop_loss,
                            'tp': signal.take_profit,
                            'sl_distance': sl_distance,
                            'tp_distance': tp_distance,
                            'rr_ratio': rr_ratio,
                            'entry_0': entry_weights.get('entry_0', 0.4),
                            'entry_1': entry_weights.get('entry_1', 0.3),
                            'entry_2': entry_weights.get('entry_2', 0.2),
                            'entry_3': entry_weights.get('entry_3', 0.1),
                            'trend_0': trend_weights.get('trend_0', 0.4),
                            'trend_1': trend_weights.get('trend_1', 0.3),
                            'trend_2': trend_weights.get('trend_2', 0.2),
                            'trend_confidence': signal.confidence if hasattr(signal, 'confidence') else 0.0,
                            **{k: v for k, v in params.items() if k.startswith(('atr_', 'garch_', 'node_', 'min_rr', 'max_risk', 'max_sl_distance', 'max_tp_distance'))}
                        }
                        self.rl_engine.save_experience(
                            symbol=signal.symbol,
                            strategy=self.current_strategy.value,
                            timeframe=signal.timeframe.value,
                            state=state,
                            action=action,
                            reward=0.0,
                            order_id=ticket
                        )
                        logger.debug(f"[RL] Experience saved for trade {ticket}: Entry weights={entry_weights}, Trend weights={trend_weights}, R/R={rr_ratio:.2f}")
                else:
                    if self.current_strategy != StrategyType.SUPER_SCALP and loop_count % 10 == 0:
                        logger.info(f"[MAIN_LOOP] No valid signal generated. Waiting for next analysis cycle.")
                
                current_time = time.time()
                if current_time - last_activity_check >= 60:
                    if self.last_activity_time is None or current_time - self.last_activity_time >= 60:
                        logger.info(f"[MAIN_LOOP] No activity in the last minute. Bot is running and waiting for opportunities.")
                    last_activity_check = current_time
                
                if self.rl_engine.should_optimize(symbol_name, self.current_strategy.value):
                    logger.info(f"[RL] Starting optimization after 20 trades for {symbol_name}-{self.current_strategy.value}")
                    new_params = self.rl_engine.optimize_all(symbol_name, self.current_strategy.value)
                    logger.info(f"[RL] Optimization completed: Updated {len(new_params)} parameters in database (using default values in code)")
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
                    for position in positions:
                        ticket = position.ticket
                        current_profit = position.profit
                        if ticket not in self.max_profit_tracker:
                            self.max_profit_tracker[ticket] = current_profit
                        else:
                            if current_profit > self.max_profit_tracker[ticket]:
                                self.max_profit_tracker[ticket] = current_profit
                        
                        if ticket not in self.order_executor.original_sl_tp:
                            self.order_executor.original_sl_tp[ticket] = {
                                'sl': position.sl,
                                'tp': position.tp
                            }
                            logger.info(f"[SENSITIVE] Initialized SL/TP protection for existing position in monitoring: Ticket={ticket}, Symbol={position.symbol}, SL={position.sl:.5f}, TP={position.tp:.5f}")
                        
                        if self.current_strategy == StrategyType.SUPER_SCALP and position.symbol == self.symbol_mgr.get_symbol_name(self.current_symbol):
                            try:
                                self.order_executor.manage_position(
                                    ticket,
                                    self.strategy_manager,
                                    self.risk_manager,
                                    self.market_engine
                                )
                            except Exception as e:
                                logger.error(f"Error managing existing position {ticket} in monitoring loop: {e}", exc_info=True)
                    
                    for ticket in closed_positions:
                        position = mt5.history_deals_get(ticket=ticket)
                        if position:
                            deals = list(position)
                            if deals:
                                total_profit = sum(d.profit for d in deals)
                                account_info = self.conn_mgr.get_account_info()
                                
                                close_deal = deals[-1]
                                close_reason = close_deal.reason
                                
                                is_sl_tp_close = (close_reason == mt5.DEAL_REASON_SL or 
                                                  close_reason == mt5.DEAL_REASON_TP or 
                                                  close_reason == mt5.DEAL_REASON_SO)
                                
                                close_reason_str = "SL" if close_reason == mt5.DEAL_REASON_SL else \
                                                  "TP" if close_reason == mt5.DEAL_REASON_TP else \
                                                  "SO" if close_reason == mt5.DEAL_REASON_SO else \
                                                  "MANUAL"
                                
                                max_profit = self.max_profit_tracker.pop(ticket, total_profit)
                                
                                self.order_executor.delete_order_lines(ticket)
                                
                                logger.info(f"[SENSITIVE] Trade closed: Ticket={ticket}, TotalProfit={total_profit:.2f}, MaxProfit={max_profit:.2f}, CloseReason={close_reason_str}, Balance={account_info.balance:.2f}, Equity={account_info.equity:.2f}")
                                cursor.execute("SELECT * FROM trades WHERE ticket = ?", (ticket,))
                                order_data = cursor.fetchone()
                                
                                if not self.db_manager.update_order(ticket, {
                                    'status': 'CLOSED',
                                    'exit_time': datetime.now(),
                                    'profit': total_profit,
                                    'max_profit': max_profit
                                }):
                                    logger.warning(f"Failed to update trade {ticket} in database after monitoring detected closure")
                                
                                if not is_sl_tp_close:
                                    logger.info(f"[RL] Trade {ticket} closed manually by user (reason: {close_reason_str}). Skipping RL learning.")
                                
                                if is_sl_tp_close:
                                    self.rl_engine.update_experience_closed_by_sl_tp(ticket, 1)
                                    
                                    trailing_stop_applied = order_data.get('trailing_stop_applied', 0) == 1 if order_data else False
                                    
                                    cursor.execute("""
                                        SELECT action FROM rl_experiences 
                                        WHERE order_id = ? AND symbol = ? AND strategy = ?
                                    """, (ticket, order_data['symbol'], self.current_strategy.value))
                                    exp_data = cursor.fetchone()
                                    rr_ratio_used = 1.0
                                    if exp_data:
                                        try:
                                            import json
                                            action_data = json.loads(exp_data['action'])
                                            rr_ratio_used = action_data.get('rr_ratio', 1.0)
                                        except:
                                            pass
                                    
                                    technical_signal = 'NEUTRAL'
                                    cursor.execute("""
                                        SELECT state FROM rl_experiences
                                        WHERE order_id = ? AND symbol = ? AND strategy = ?
                                        ORDER BY timestamp DESC LIMIT 1
                                    """, (ticket, order_data['symbol'], self.current_strategy.value))
                                    exp_data = cursor.fetchone()
                                    if exp_data:
                                        try:
                                            import json
                                            state_data = json.loads(exp_data['state'])
                                            technical_signal = state_data.get('technical_signal', 'NEUTRAL')
                                        except:
                                            pass

                                    reward = self.rl_engine.calculate_reward(
                                        order_profit=total_profit,
                                        transaction_cost=abs(total_profit) * 0.001,
                                        risk_penalty=0.0,
                                        rr_ratio=rr_ratio_used,
                                        hold_time=0.0,
                                        max_profit=max_profit,
                                        close_reason=close_reason_str,
                                        trailing_stop_applied=trailing_stop_applied,
                                        technical_signal=technical_signal
                                    )
                                    
                                    cursor.execute("""
                                        UPDATE rl_experiences 
                                        SET reward = ? 
                                        WHERE order_id = ? AND symbol = ? AND strategy = ?
                                    """, (reward, ticket, order_data['symbol'], self.current_strategy.value))
                                    
                                    self.rl_engine.optimize_based_on_result(
                                        symbol=order_data['symbol'],
                                        strategy=self.current_strategy.value,
                                        order_id=ticket,
                                        close_reason=close_reason_str,
                                        trailing_stop_applied=trailing_stop_applied,
                                        max_profit=max_profit,
                                        order_profit=total_profit,
                                        technical_signal=technical_signal
                                    )
                                
                                if self.current_strategy == StrategyType.SUPER_SCALP and order_data and is_sl_tp_close:
                                    import json
                                    cursor.execute("""
                                        SELECT state, timeframe FROM rl_experiences 
                                        WHERE order_id = ? AND symbol = ? AND strategy = ?
                                        ORDER BY timestamp DESC LIMIT 1
                                    """, (ticket, order_data['symbol'], self.current_strategy.value))
                                    exp_data = cursor.fetchone()
                                    if exp_data:
                                        try:
                                            state = json.loads(exp_data['state'])
                                            timeframe = exp_data['timeframe']
                                            if 'trend_strengths' in state and state['trend_strengths']:
                                                timeframes_list = state.get('timeframes', [])
                                                if timeframes_list and len(state['trend_strengths']) == len(timeframes_list):
                                                    for i, tf_name in enumerate(timeframes_list[:3]):
                                                        if i < len(state['trend_strengths']):
                                                            actual_strength = state['trend_strengths'][i]
                                                            self.rl_engine.optimize_trend_strength(
                                                                symbol=order_data['symbol'],
                                                                strategy=self.current_strategy.value,
                                                                timeframe=tf_name,
                                                                actual_strength=actual_strength,
                                                                reward=reward
                                                            )
                                        except Exception as e:
                                            logger.error(f"Error optimizing trend strength for ticket {ticket}: {e}")
                                
                                if self.telegram_bot and order_data:
                                    try:
                                        profit_emoji = "✅" if total_profit > 0 else "❌"
                                        message = f"""{profit_emoji} <b>Order Closed</b>

📊 <b>Order Details:</b>
• Ticket: {ticket}
• Symbol: {order_data['symbol']}
• Direction: {order_data['direction']}
• Entry: {order_data['entry_price']:.5f}
• Exit: {deals[-1].price if deals else 'N/A':.5f}
• Profit/Loss: ${total_profit:.2f}

💰 <b>Account:</b>
• Balance: ${account_info.balance:.2f}
• Equity: ${account_info.equity:.2f}"""
                                        await self.telegram_bot.send_notification(message)
                                        logger.info(f"[TELEGRAM] Order closed notification sent for ticket {ticket}")
                                    except Exception as e:
                                        logger.error(f"[TELEGRAM] Error sending order closed notification: {e}")
                                elif not self.telegram_bot:
                                    logger.warning(f"[TELEGRAM] Telegram bot not available, skipping order closed notification for ticket {ticket}")
                                elif not order_data:
                                    logger.warning(f"[TELEGRAM] Order data not found for ticket {ticket}, skipping notification")
                
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
                for chat_id in self.telegram_bot.chat_ids:
                    try:
                        await self.telegram_bot.send_status_message(chat_id, is_start=False)
                    except Exception as e:
                        logger.error(f"Error sending stop status message: {e}")
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
