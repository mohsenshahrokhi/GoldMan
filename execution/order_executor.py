from typing import List, Optional, Any

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from datetime import datetime

from utils.logger import logger
from utils.data_classes import OrderSignal
from config.constants import MIN_RR_RATIO
from config.enums import TimeFrame
from connection.mt5_manager import ConnectionManager
from database.manager import DatabaseManager
from strategy.strategy_manager import StrategyManager
from risk.risk_manager import RiskManager
from analysis.market_engine import MarketEngine
from market.data_provider import MarketDataProvider


class OrderExecutor:
    
    def __init__(self, connection_manager: ConnectionManager, db_manager: DatabaseManager):
        self.conn_mgr = connection_manager
        self.db = db_manager
        self.current_position = None
        self.frozen_positions = []
        self.data_provider = None
        self.main_controller = None
        self.original_sl_tp = {}  # Store original SL/TP for each ticket
        self._last_sl_tp_warning = {}  # Debounce SL/TP warnings
    
    async def _send_telegram_notification(self, message: str):
        """ارسال اعلان به تلگرام"""
        if self.main_controller and self.main_controller.telegram_bot:
            try:
                await self.main_controller.telegram_bot.send_notification(message)
                logger.info(f"[TELEGRAM] Notification sent from OrderExecutor")
            except Exception as e:
                logger.error(f"[TELEGRAM] Error sending Telegram notification: {e}")
        else:
            if not self.main_controller:
                logger.warning("[TELEGRAM] Main controller not available, skipping notification")
            elif not self.main_controller.telegram_bot:
                logger.warning("[TELEGRAM] Telegram bot not available, skipping notification")
    
    def get_filling_mode(self, symbol_info) -> int:
        """تشخیص filling mode مناسب برای symbol"""
        if mt5 is None:
            return 0
        
        filling_mode = symbol_info.filling_mode
        
        if filling_mode & 1:
            result = mt5.ORDER_FILLING_IOC
        elif filling_mode & 2:
            result = mt5.ORDER_FILLING_RETURN
        elif filling_mode & 4:
            result = mt5.ORDER_FILLING_FOK
        else:
            logger.warning(f"Unknown filling mode {filling_mode} for {symbol_info.name}, using FOK as default")
            result = mt5.ORDER_FILLING_FOK
        
        logger.debug(f"Filling mode for {symbol_info.name}: symbol_filling={filling_mode}, order_filling={result}")
        return result
    
    def get_available_filling_modes(self, symbol_info) -> List[int]:
        """دریافت لیست filling modeهای موجود برای symbol به ترتیب اولویت"""
        if mt5 is None:
            return [mt5.ORDER_FILLING_FOK]
        
        filling_mode = symbol_info.filling_mode
        modes = []
        
        if filling_mode & 1:
            modes.append(mt5.ORDER_FILLING_IOC)
        if filling_mode & 2:
            modes.append(mt5.ORDER_FILLING_RETURN)
        if filling_mode & 4:
            modes.append(mt5.ORDER_FILLING_FOK)
        
        if not modes:
            modes = [mt5.ORDER_FILLING_FOK]
        
        return modes
    
    def set_data_provider(self, data_provider: MarketDataProvider):
        self.data_provider = data_provider
    
    def get_open_positions(self) -> List[Any]:
        if mt5 is None:
            return []
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            return list(positions)
        except Exception as e:
            logger.error(f"Error retrieving open positions: {e}")
            return []
    
    def has_open_position(self, symbol: str = None) -> bool:
        positions = self.get_open_positions()
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return len(positions) > 0
    
    def execute_order(self, signal: OrderSignal) -> Optional[int]:
        if not self.pre_order_validation(signal):
            return None
        
        if mt5 is None:
            logger.error("MetaTrader5 is not installed")
            return None
        
        try:
            symbol_info = mt5.symbol_info(signal.symbol)
            if symbol_info is None:
                logger.error(f"Symbol info for {signal.symbol} not found")
                return None
            
            order_type = mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL
            
            if signal.direction == "BUY":
                price = mt5.symbol_info_tick(signal.symbol).ask
            else:
                price = mt5.symbol_info_tick(signal.symbol).bid
            
            sl = signal.stop_loss
            tp = signal.take_profit
            
            available_modes = self.get_available_filling_modes(symbol_info)
            all_modes = [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]
            
            modes_to_try = available_modes + [m for m in all_modes if m not in available_modes]
            logger.info(f"Filling modes to try for {signal.symbol}: {modes_to_try} (symbol_filling={symbol_info.filling_mode})")
            
            spread_pips = (symbol_info.ask - symbol_info.bid) / symbol_info.point
            if symbol_info.digits == 3 or symbol_info.digits == 5:
                spread_pips = spread_pips / 10
            
            sl_distance = abs(price - sl)
            tp_distance = abs(tp - price)
            sl_pips = sl_distance / symbol_info.point
            tp_pips = tp_distance / symbol_info.point
            if symbol_info.digits == 3 or symbol_info.digits == 5:
                sl_pips = sl_pips / 10
                tp_pips = tp_pips / 10
            
            risk_amount = sl_distance * signal.lot_size * symbol_info.trade_contract_size
            reward_amount = tp_distance * signal.lot_size * symbol_info.trade_contract_size
            
            logger.info(f"[SENSITIVE] Attempting to open trade: Symbol={signal.symbol}, Direction={signal.direction}, Entry={price:.5f}, SL={sl:.5f}, TP={tp:.5f}, LotSize={signal.lot_size:.2f}, SLDistance={sl_pips:.1f}pips, TPDistance={tp_pips:.1f}pips, Spread={spread_pips:.1f}pips, RiskAmount=${risk_amount:.2f}, RewardAmount=${reward_amount:.2f}, R/R={signal.confidence:.2f}")
            
            result = None
            last_error = None
            
            for filling_mode in modes_to_try:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": signal.symbol,
                    "volume": signal.lot_size,
                    "type": order_type,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": f"GoldMan-{signal.timeframe}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling_mode,
                }
            
            result = mt5.order_send(request)
            
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"Successfully opened order with filling mode {filling_mode}")
                    break
                else:
                    last_error = result
                    if result.retcode == 10030:
                        logger.warning(f"Filling mode {filling_mode} not supported, trying next mode...")
                        continue
                    else:
                        logger.error(f"[SENSITIVE] Failed to open trade: Retcode={result.retcode}, Comment={result.comment}, Symbol={signal.symbol}, Direction={signal.direction}, LotSize={signal.lot_size:.2f}, FillingMode={filling_mode}")
                        break
            
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                if last_error:
                    logger.error(f"[SENSITIVE] Failed to open trade after trying all filling modes: Retcode={last_error.retcode}, Comment={last_error.comment}, Symbol={signal.symbol}, Direction={signal.direction}, LotSize={signal.lot_size:.2f}")
                return None
            
            ticket = result.order
            account_info = self.conn_mgr.get_account_info()
            logger.info(f"[SENSITIVE] Trade opened successfully: Ticket={ticket}, Symbol={signal.symbol}, Direction={signal.direction}, Entry={price:.5f}, SL={sl:.5f}, TP={tp:.5f}, LotSize={signal.lot_size:.2f}, SLDistance={sl_pips:.1f}pips, TPDistance={tp_pips:.1f}pips, RiskAmount=${risk_amount:.2f}, RewardAmount=${reward_amount:.2f}, R/R={signal.confidence:.2f}, Balance={account_info.balance:.2f}, Equity={account_info.equity:.2f}, FreeMargin={account_info.free_margin:.2f}, MarginLevel={account_info.margin_level:.2f}%")
            
            if not self.db.save_order({
                'ticket': ticket,
                'symbol': signal.symbol,
                'direction': signal.direction,
                'entry_price': price,
                'stop_loss': sl,
                'take_profit': tp,
                'lot_size': signal.lot_size,
                'entry_time': datetime.now(),
                'strategy': signal.timeframe.name
            }):
                logger.warning(f"Failed to save trade {ticket} to database, but trade is open")
            
            self.original_sl_tp[ticket] = {'sl': sl, 'tp': tp}
            self.current_position = ticket
            return ticket
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return None
    
    def pre_order_validation(self, signal: OrderSignal) -> bool:
        if self.has_open_position(signal.symbol):
            logger.info("Active position exists")
            return False
        
        if not self.conn_mgr.check_connection():
            logger.error("MT5 connection is not established")
            return False
        
        if mt5 is None:
            return False
        symbol_info = mt5.symbol_info(signal.symbol)
        if symbol_info is None or not symbol_info.visible:
            logger.error(f"Symbol {signal.symbol} is not active")
            return False
        
        account_info = self.conn_mgr.get_account_info()
        if account_info is None or account_info.balance < 500:
            logger.error("Insufficient balance")
            return False
        
        symbol_info = mt5.symbol_info(signal.symbol)
        required_margin = mt5.order_calc_margin(
            mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL,
            signal.symbol,
            signal.lot_size,
            signal.entry_price
        )
        
        if required_margin > account_info.free_margin:
            logger.error("Insufficient margin")
            return False
        
        if signal.confidence < MIN_RR_RATIO:
            logger.error(f"R/R ratio ({signal.confidence:.2f}) is below minimum")
            return False
        
        return True
    
    def manage_position(self, ticket: int, strategy_manager: StrategyManager, 
                       risk_manager: RiskManager, market_engine: MarketEngine) -> bool:
        if mt5 is None:
            return False
            
        try:
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                self.current_position = None
                return False
            
            position = position[0]
            current_price = position.price_current
            entry_price = position.price_open
            sl_current = position.sl
            tp_current = position.tp
            volume = position.volume
            symbol = position.symbol
            position_profit = position.profit
            position_profit = position.profit
            
            if ticket in self.original_sl_tp:
                original_sl = self.original_sl_tp[ticket]['sl']
                original_tp = self.original_sl_tp[ticket]['tp']
                
                symbol_info = None
                if self.data_provider:
                    symbol_info = self.data_provider.get_symbol_info(symbol)
                elif strategy_manager and strategy_manager.data_provider:
                    symbol_info = strategy_manager.data_provider.get_symbol_info(symbol)
                
                if symbol_info:
                    if symbol_info.digits == 3 or symbol_info.digits == 5:
                        tolerance = symbol_info.point * 100
                    else:
                        tolerance = symbol_info.point * 50
                else:
                    tolerance = 1.0
                
                sl_changed = False
                tp_changed = False
                
                if original_sl > 0:
                    sl_diff = abs(sl_current - original_sl)
                    if sl_diff > tolerance:
                        if position.type == mt5.ORDER_TYPE_BUY:
                            sl_changed = sl_current < original_sl
                        else:
                            sl_changed = sl_current > original_sl
                else:
                    sl_changed = sl_current != original_sl
                
                if original_tp > 0:
                    tp_changed = abs(tp_current - original_tp) > tolerance
                else:
                    tp_changed = tp_current != original_tp
                
                if sl_changed or tp_changed:
                    import time
                    current_time = time.time()
                    last_warning_time = self._last_sl_tp_warning.get(ticket, 0)
                    should_warn = current_time - last_warning_time >= 60
                    
                    if should_warn:
                        logger.warning(f"[SENSITIVE] SL/TP changed by user detected: Ticket={ticket}, OriginalSL={original_sl:.5f}, CurrentSL={sl_current:.5f}, OriginalTP={original_tp:.5f}, CurrentTP={tp_current:.5f}, Tolerance={tolerance:.5f}. Restoring original values.")
                        self._last_sl_tp_warning[ticket] = current_time
                    
                    if should_warn:
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": symbol,
                            "position": ticket,
                            "sl": original_sl,
                            "tp": original_tp,
                        }
                        result = mt5.order_send(request)
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"[SENSITIVE] SL/TP restored successfully: Ticket={ticket}, SL={original_sl:.5f}, TP={original_tp:.5f}")
                            self._last_sl_tp_warning.pop(ticket, None)
                        elif result.retcode == 10025:
                            logger.debug(f"[SENSITIVE] SL/TP already at original values: Ticket={ticket}, SL={original_sl:.5f}, TP={original_tp:.5f}")
                            self._last_sl_tp_warning.pop(ticket, None)
                        else:
                            logger.error(f"[SENSITIVE] Failed to restore SL/TP: Ticket={ticket}, Retcode={result.retcode}, Comment={result.comment}")
            
            account_info = self.conn_mgr.get_account_info()
            balance_value = account_info.balance if account_info else 0.0
            
            price_change_pct = ((current_price - entry_price) / entry_price) * 100 if position.type == mt5.ORDER_TYPE_BUY else ((entry_price - current_price) / entry_price) * 100
            
            if position.type == mt5.ORDER_TYPE_BUY:
                if current_price >= entry_price and tp_current > entry_price:
                    distance_to_tp = current_price - entry_price
                    total_tp_distance = tp_current - entry_price
                    profit_pct = (distance_to_tp / total_tp_distance) * 100 if total_tp_distance > 0 else 0
                    profit_calc_method = "tp_distance"
                    logger.debug(f"[PROFIT_CALC] BUY in profit: Entry={entry_price:.5f}, Current={current_price:.5f}, TP={tp_current:.5f}, DistanceToTP={distance_to_tp:.5f}, TotalTPDistance={total_tp_distance:.5f}, Profit%={profit_pct:.2f}%")
                elif current_price < entry_price and sl_current > 0 and sl_current < entry_price:
                    distance_to_sl = entry_price - current_price
                    total_sl_distance = entry_price - sl_current
                    profit_pct = ((current_price - entry_price) / total_sl_distance) * 100 if total_sl_distance > 0 else 0
                    profit_calc_method = "sl_distance"
                    logger.debug(f"[PROFIT_CALC] BUY in loss: Entry={entry_price:.5f}, Current={current_price:.5f}, SL={sl_current:.5f}, DistanceToSL={distance_to_sl:.5f}, TotalSLDistance={total_sl_distance:.5f}, Profit%={profit_pct:.2f}%")
                else:
                    profit_pct = price_change_pct
                    profit_calc_method = "price_change"
                    logger.debug(f"[PROFIT_CALC] BUY fallback: Entry={entry_price:.5f}, Current={current_price:.5f}, TP={tp_current:.5f}, SL={sl_current:.5f}, PriceChange%={price_change_pct:.2f}%, Profit%={profit_pct:.2f}%")
            else:
                if current_price <= entry_price and tp_current > 0 and tp_current < entry_price:
                    distance_to_tp = entry_price - current_price
                    total_tp_distance = entry_price - tp_current
                    profit_pct = (distance_to_tp / total_tp_distance) * 100 if total_tp_distance > 0 else 0
                    profit_calc_method = "tp_distance"
                    logger.debug(f"[PROFIT_CALC] SELL in profit: Entry={entry_price:.5f}, Current={current_price:.5f}, TP={tp_current:.5f}, DistanceToTP={distance_to_tp:.5f}, TotalTPDistance={total_tp_distance:.5f}, Profit%={profit_pct:.2f}%")
                elif current_price > entry_price and sl_current > entry_price:
                    distance_to_sl = current_price - entry_price
                    total_sl_distance = sl_current - entry_price
                    profit_pct = ((entry_price - current_price) / total_sl_distance) * 100 if total_sl_distance > 0 else 0
                    profit_calc_method = "sl_distance"
                    logger.debug(f"[PROFIT_CALC] SELL in loss: Entry={entry_price:.5f}, Current={current_price:.5f}, SL={sl_current:.5f}, DistanceToSL={distance_to_sl:.5f}, TotalSLDistance={total_sl_distance:.5f}, Profit%={profit_pct:.2f}%")
            else:
                    profit_pct = price_change_pct
                    profit_calc_method = "price_change"
                    logger.debug(f"[PROFIT_CALC] SELL fallback: Entry={entry_price:.5f}, Current={current_price:.5f}, TP={tp_current:.5f}, SL={sl_current:.5f}, PriceChange%={price_change_pct:.2f}%, Profit%={profit_pct:.2f}%")
            
            profit_vs_balance_pct = (position_profit / balance_value) * 100 if balance_value > 0 else 0.0
            
            logger.info(f"[MANAGE_POSITION] Ticket={ticket}, Symbol={symbol}, Direction={'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'}")
            logger.info(f"  Entry={entry_price:.5f}, Current={current_price:.5f}, PriceChange%={price_change_pct:.2f}%")
            logger.info(f"  Profit=${position_profit:.2f}, Balance=${balance_value:.2f}")
            logger.info(f"  Profit%={profit_pct:.2f}% (to TP/SL, method: {profit_calc_method}), Profit% vs Balance={profit_vs_balance_pct:.2f}%")
            logger.info(f"  SL={sl_current:.5f}, TP={tp_current:.5f}, Volume={volume:.2f}")
            logger.info(f"  Trailing Stop Conditions: >=10%={profit_pct >= 10}, >=15%={profit_pct >= 15}, >=25%={profit_pct >= 25}, >=50%={profit_pct >= 50}, >=75%={profit_pct >= 75}, >=80%={profit_pct >= 80}")
            logger.info(f"  Partial Exit Conditions: >=5%={profit_pct >= 5}, >=10%={profit_pct >= 10}, >=50%={profit_pct >= 50}, >=75%={profit_pct >= 75}")
            
            should_close_and_reopen = self.check_close_and_reopen(position, profit_pct, strategy_manager, risk_manager, market_engine)
            if should_close_and_reopen:
                logger.info(f"[SENSITIVE] Closing position at 80% profit for reopening: Ticket={ticket}, Symbol={symbol}, Profit={profit_pct:.2f}%")
                self.close_position(ticket)
                return False
            
            logger.info(f"[MANAGE_POSITION] Calling apply_trailing_stop and apply_partial_exit for ticket {ticket}, profit_pct={profit_pct:.2f}%, strategy={strategy_manager.current_strategy}")
            
            self.apply_trailing_stop(position, profit_pct, risk_manager, market_engine, strategy_manager.current_strategy)
            self.apply_partial_exit(position, profit_pct, risk_manager, strategy_manager.current_strategy)
            
            from config.enums import StrategyType
            if strategy_manager.current_strategy != StrategyType.SUPER_SCALP:
                position_after_trailing = mt5.positions_get(ticket=ticket)
                if position_after_trailing and len(position_after_trailing) > 0:
                    sl_after_trailing = position_after_trailing[0].sl
                    weakness, weakness_price = strategy_manager.check_trend_weakness()
                    if weakness and weakness_price:
                        if position.type == mt5.ORDER_TYPE_BUY:
                            if weakness_price > sl_after_trailing:
                                logger.info(f"[SENSITIVE] Trend weakness detected (BUY): Ticket={ticket}, Symbol={symbol}, CurrentSL={sl_after_trailing:.5f}, WeaknessPrice={weakness_price:.5f}, Updating SL")
                                self.update_stop_loss(ticket, weakness_price)
                            else:
                                logger.debug(f"[TREND_WEAKNESS] Weakness price {weakness_price:.5f} not better than current SL {sl_after_trailing:.5f} for BUY")
                        else:
                            if sl_after_trailing == 0 or weakness_price < sl_after_trailing:
                                logger.info(f"[SENSITIVE] Trend weakness detected (SELL): Ticket={ticket}, Symbol={symbol}, CurrentSL={sl_after_trailing:.5f}, WeaknessPrice={weakness_price:.5f}, Updating SL")
                                self.update_stop_loss(ticket, weakness_price)
                            else:
                                logger.debug(f"[TREND_WEAKNESS] Weakness price {weakness_price:.5f} not better than current SL {sl_after_trailing:.5f} for SELL")
            
            logger.info(f"[MANAGE_POSITION] Completed management for ticket {ticket}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error managing position {ticket}: {e}")
            return False
    
    def apply_trailing_stop(self, position: Any, profit_pct: float, 
                           risk_manager: RiskManager, market_engine: MarketEngine,
                           strategy: Any = None):
        if mt5 is None or self.data_provider is None:
            return
            
        symbol = position.symbol
        entry_price = position.price_open
        current_sl = position.sl
        direction = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return
        
        spread = symbol_info.spread * symbol_info.point
        commission = getattr(symbol_info, 'commission', 0.0)
        if hasattr(symbol_info, 'commission_mode') and symbol_info.commission_mode == mt5.COMMISSION_MONEY:
            commission = commission / symbol_info.trade_contract_size if symbol_info.trade_contract_size > 0 else 0.0
        safety_margin = 5 * symbol_info.point
        adjustment = spread + safety_margin + commission
        
        if strategy:
            from config.enums import StrategyType
            if strategy == StrategyType.DAY_TRADING:
                node_timeframe = TimeFrame.M15
            elif strategy == StrategyType.SCALP:
                node_timeframe = TimeFrame.M5
            elif strategy == StrategyType.SUPER_SCALP:
                node_timeframe = TimeFrame.M1
            else:
                node_timeframe = TimeFrame.M15
        else:
            node_timeframe = TimeFrame.M15
        
        df = self.data_provider.get_ohlc_data(symbol, node_timeframe, 200)
        if df is None:
            return
        
        new_sl = None
        from config.enums import StrategyType
        
        if strategy == StrategyType.SUPER_SCALP:
            logger.info(f"[TRAILING_STOP] Super Scalp - Checking conditions: Ticket={position.ticket}, Profit%={profit_pct:.2f}%, Direction={direction}")
            
            if profit_pct >= 10 and profit_pct < 15:
                logger.info(f"[TRAILING_STOP] Condition met: 10% <= Profit% < 15% (Current: {profit_pct:.2f}%)")
                if direction == "BUY":
                    node = market_engine.find_nearest_node(df, entry_price, "below")
                    if node:
                        new_sl = node - adjustment
                        logger.info(f"[TRAILING_STOP] Found node for BUY: {node:.5f}, NewSL={new_sl:.5f}")
                    else:
                        logger.warning(f"[TRAILING_STOP] No node found for BUY at entry_price={entry_price:.5f}")
                else:
                    node = market_engine.find_nearest_node(df, entry_price, "above")
                    if node:
                        new_sl = node + adjustment
                        logger.info(f"[TRAILING_STOP] Found node for SELL: {node:.5f}, NewSL={new_sl:.5f}")
                    else:
                        logger.warning(f"[TRAILING_STOP] No node found for SELL at entry_price={entry_price:.5f}")
            
            elif profit_pct >= 15 and profit_pct < 25:
                logger.info(f"[TRAILING_STOP] Condition met: 15% <= Profit% < 25% (Current: {profit_pct:.2f}%)")
                if direction == "BUY":
                    new_sl = entry_price + adjustment
                else:
                    new_sl = entry_price - adjustment
                logger.info(f"[TRAILING_STOP] Break Even +: NewSL={new_sl:.5f}, Entry={entry_price:.5f}, Adjustment={adjustment:.5f}")
            
            elif profit_pct >= 25 and profit_pct < 50:
                logger.info(f"[TRAILING_STOP] Condition met: 25% <= Profit% < 50% (Current: {profit_pct:.2f}%)")
                current_price = position.price_current
                if direction == "BUY":
                    new_sl = current_price - (1.5 * adjustment)
                    if current_sl > 0:
                        if new_sl <= current_sl:
                            new_sl = current_sl + (symbol_info.point * 5)
                        elif new_sl < current_sl + (symbol_info.point * 5):
                            new_sl = current_sl + (symbol_info.point * 5)
                else:
                    new_sl = current_price + (1.5 * adjustment)
                    if current_sl > 0:
                        if new_sl >= current_sl:
                            new_sl = current_sl - (symbol_info.point * 5)
                        elif new_sl > current_sl - (symbol_info.point * 5):
                            new_sl = current_sl - (symbol_info.point * 5)
                logger.info(f"[TRAILING_STOP] Medium trailing: NewSL={new_sl:.5f}, Current={current_price:.5f}, CurrentSL={current_sl:.5f}, Adjustment={1.5 * adjustment:.5f}")
            
            elif profit_pct >= 50 and profit_pct < 75:
                logger.info(f"[TRAILING_STOP] Condition met: 50% <= Profit% < 75% (Current: {profit_pct:.2f}%)")
                current_price = position.price_current
                if direction == "BUY":
                    new_sl = current_price - adjustment
                    if current_sl > 0 and new_sl <= current_sl:
                        new_sl = current_sl + (symbol_info.point * 5)
                else:
                    new_sl = current_price + adjustment
                    if current_sl > 0 and new_sl >= current_sl:
                        new_sl = current_sl - (symbol_info.point * 5)
                logger.info(f"[TRAILING_STOP] Close trailing: NewSL={new_sl:.5f}, Current={current_price:.5f}, CurrentSL={current_sl:.5f}, Adjustment={adjustment:.5f}")
            
            elif profit_pct >= 75 and profit_pct < 80:
                logger.info(f"[TRAILING_STOP] Condition met: 75% <= Profit% < 80% (Current: {profit_pct:.2f}%)")
                current_price = position.price_current
                if direction == "BUY":
                    new_sl = current_price - (0.5 * adjustment)
                    if current_sl > 0 and new_sl <= current_sl:
                        new_sl = current_sl + (symbol_info.point * 5)
                else:
                    new_sl = current_price + (0.5 * adjustment)
                    if current_sl > 0 and new_sl >= current_sl:
                        new_sl = current_sl - (symbol_info.point * 5)
                logger.info(f"[TRAILING_STOP] Aggressive trailing: NewSL={new_sl:.5f}, Current={current_price:.5f}, CurrentSL={current_sl:.5f}, Adjustment={0.5 * adjustment:.5f}")
            
            elif profit_pct >= 80:
                logger.info(f"[TRAILING_STOP] Condition met: Profit% >= 80% (Current: {profit_pct:.2f}%)")
                current_price = position.price_current
                if direction == "BUY":
                    new_sl = current_price - adjustment
                    if current_sl > 0 and new_sl <= current_sl:
                        new_sl = current_sl + (symbol_info.point * 5)
                else:
                    new_sl = current_price + adjustment
                    if current_sl > 0 and new_sl >= current_sl:
                        new_sl = current_sl - (symbol_info.point * 5)
                logger.info(f"[TRAILING_STOP] Maximum trailing: NewSL={new_sl:.5f}, Current={current_price:.5f}, CurrentSL={current_sl:.5f}, Adjustment={adjustment:.5f}")
            else:
                logger.info(f"[TRAILING_STOP] No condition met: Profit%={profit_pct:.2f}% (Required: >=10% for first stage)")
        else:
        if profit_pct >= 10 and profit_pct < 15:
            if direction == "BUY":
                    node = market_engine.find_nearest_node(df, entry_price, "below")
                if node:
                    new_sl = node - adjustment
            else:
                    node = market_engine.find_nearest_node(df, entry_price, "above")
                if node:
                    new_sl = node + adjustment
        
        elif profit_pct >= 15 and profit_pct < 50:
            if direction == "BUY":
                new_sl = entry_price + adjustment
            else:
                new_sl = entry_price - adjustment
        
        elif profit_pct >= 50 and profit_pct < 75:
            target_price = entry_price * (1 + 0.5 * (1 if direction == "BUY" else -1))
            if direction == "BUY":
                    node = market_engine.find_nearest_node(df, target_price, "below")
                if node:
                    new_sl = node - adjustment
            else:
                    node = market_engine.find_nearest_node(df, target_price, "above")
                if node:
                    new_sl = node + adjustment
        
        elif profit_pct >= 75 and profit_pct < 80:
            target_price = entry_price * (1 + 0.75 * (1 if direction == "BUY" else -1))
            if direction == "BUY":
                    node = market_engine.find_nearest_node(df, target_price, "below")
                if node:
                    new_sl = node - adjustment
            else:
                    node = market_engine.find_nearest_node(df, target_price, "above")
                if node:
                    new_sl = node + adjustment
        
        elif profit_pct >= 80:
            current_price = position.price_current
            if direction == "BUY":
                new_sl = current_price - adjustment
            else:
                new_sl = current_price + adjustment
        
        if new_sl is not None:
            should_update = False
            tolerance = symbol_info.point * 3
            
            if direction == "BUY":
                if current_sl == 0:
                    should_update = True
                elif new_sl > current_sl + tolerance:
                    should_update = True
                else:
                    logger.debug(f"[TRAILING_STOP] BUY: NewSL ({new_sl:.5f}) not significantly better than CurrentSL ({current_sl:.5f}), difference: {new_sl - current_sl:.5f}, tolerance: {tolerance:.5f}")
            else:
                if current_sl == 0:
                    should_update = True
                elif new_sl < current_sl - tolerance:
                    should_update = True
                else:
                    logger.debug(f"[TRAILING_STOP] SELL: NewSL ({new_sl:.5f}) not significantly better than CurrentSL ({current_sl:.5f}), difference: {current_sl - new_sl:.5f}, tolerance: {tolerance:.5f}")
            
            if should_update:
                logger.info(f"[SENSITIVE] Trailing stop triggered ({direction}): Ticket={position.ticket}, Symbol={symbol}, Profit={profit_pct:.2f}%, OldSL={current_sl:.5f}, NewSL={new_sl:.5f}")
                update_success = self.update_stop_loss(position.ticket, new_sl)
                
                if strategy == StrategyType.SUPER_SCALP:
                    stage = None
                    if profit_pct >= 10 and profit_pct < 15:
                        stage = "10"
                    elif profit_pct >= 15 and profit_pct < 25:
                        stage = "15"
                    elif profit_pct >= 25 and profit_pct < 50:
                        stage = "25"
                    elif profit_pct >= 50 and profit_pct < 75:
                        stage = "50"
                    elif profit_pct >= 75 and profit_pct < 80:
                        stage = "75"
                    elif profit_pct >= 80:
                        stage = "80"
                    
                    if not stage:
                        logger.debug(f"[TRAILING_STOP] No stage determined for profit_pct={profit_pct:.2f}%")
            else:
                logger.debug(f"[TRAILING_STOP] No update needed: Ticket={position.ticket}, Direction={direction}, CurrentSL={current_sl:.5f}, NewSL={new_sl:.5f}, Profit%={profit_pct:.2f}%")
        else:
            if strategy == StrategyType.SUPER_SCALP:
                logger.info(f"[TRAILING_STOP] Super Scalp: No trailing stop calculated. Ticket={position.ticket}")
                logger.info(f"  Profit%={profit_pct:.2f}%, Direction={direction}, Entry={entry_price:.5f}, Current={position.price_current:.5f}")
                logger.info(f"  Conditions check: >=10%={profit_pct >= 10} ({profit_pct:.2f} >= 10), >=15%={profit_pct >= 15} ({profit_pct:.2f} >= 15), >=25%={profit_pct >= 25} ({profit_pct:.2f} >= 25)")
                logger.info(f"  All conditions: >=10%={profit_pct >= 10}, >=15%={profit_pct >= 15}, >=25%={profit_pct >= 25}, >=50%={profit_pct >= 50}, >=75%={profit_pct >= 75}, >=80%={profit_pct >= 80}")
    
    def apply_partial_exit(self, position: Any, profit_pct: float, risk_manager: RiskManager, strategy: Any = None):
        from config.enums import StrategyType
        
        logger.info(f"[PARTIAL_EXIT] Checking conditions: Ticket={position.ticket}, Strategy={strategy}, Profit%={profit_pct:.2f}%, Volume={position.volume:.2f}")
        
        if strategy == StrategyType.SUPER_SCALP:
            logger.info(f"[PARTIAL_EXIT] Super Scalp - Checking conditions: Ticket={position.ticket}, Profit%={profit_pct:.2f}%, Volume={position.volume:.2f}")
            
            if profit_pct >= 5 and profit_pct < 7:
                logger.info(f"[PARTIAL_EXIT] Condition met: 5% <= Profit% < 7% (Current: {profit_pct:.2f}%)")
                if position.volume > 0.02:
                    close_volume = position.volume * 0.3
                    logger.info(f"[SENSITIVE] Partial exit executing: Closing 30% of volume={close_volume:.2f} (Current volume: {position.volume:.2f})")
                    self.close_partial_position(position.ticket, close_volume)
                else:
                    logger.warning(f"[PARTIAL_EXIT] Volume too small: {position.volume:.2f} <= 0.02, skipping partial exit")
            
            elif profit_pct >= 10 and profit_pct < 15:
                logger.info(f"[PARTIAL_EXIT] Condition met: 10% <= Profit% < 15% (Current: {profit_pct:.2f}%)")
                if position.volume > 0.01:
                    close_volume = position.volume * 0.2
                    logger.info(f"[SENSITIVE] Partial exit executing: Closing 20% of remaining volume={close_volume:.2f} (Current volume: {position.volume:.2f})")
                    self.close_partial_position(position.ticket, close_volume)
                else:
                    logger.warning(f"[PARTIAL_EXIT] Volume too small: {position.volume:.2f} <= 0.01, skipping partial exit")
            
            elif profit_pct >= 15 and profit_pct < 50:
                logger.info(f"[PARTIAL_EXIT] Condition met: 15% <= Profit% < 50% (Current: {profit_pct:.2f}%)")
                if position.volume > 0.01:
                    close_volume = position.volume * 0.5
                    logger.info(f"[SENSITIVE] Partial exit executing: Closing 50% of remaining volume={close_volume:.2f} (Current volume: {position.volume:.2f})")
                    self.close_partial_position(position.ticket, close_volume)
                else:
                    logger.warning(f"[PARTIAL_EXIT] Volume too small: {position.volume:.2f} <= 0.01, skipping partial exit")
            
            elif profit_pct >= 50 and profit_pct < 75:
                logger.info(f"[PARTIAL_EXIT] Condition met: 50% <= Profit% < 75% (Current: {profit_pct:.2f}%)")
                if position.volume > 0.01:
                    close_volume = position.volume * 0.3
                    logger.info(f"[SENSITIVE] Partial exit executing: Closing 30% of remaining volume={close_volume:.2f} (Current volume: {position.volume:.2f})")
                    self.close_partial_position(position.ticket, close_volume)
                else:
                    logger.warning(f"[PARTIAL_EXIT] Volume too small: {position.volume:.2f} <= 0.01, skipping partial exit")
            
            elif profit_pct >= 75:
                logger.info(f"[PARTIAL_EXIT] Condition met: Profit% >= 75% (Current: {profit_pct:.2f}%)")
                if position.volume > 0.01:
                    close_volume = position.volume * 0.2
                    logger.info(f"[SENSITIVE] Partial exit executing: Closing 20% of remaining volume={close_volume:.2f} (Current volume: {position.volume:.2f})")
                    self.close_partial_position(position.ticket, close_volume)
                else:
                    logger.warning(f"[PARTIAL_EXIT] Volume too small: {position.volume:.2f} <= 0.01, skipping partial exit")
            else:
                logger.info(f"[PARTIAL_EXIT] No condition met: Profit%={profit_pct:.2f}% (Required: >=5% for first stage, Current volume: {position.volume:.2f})")
        else:
        if profit_pct >= 50 and profit_pct < 75:
            if position.volume > 0.02:
                close_volume = position.volume * 0.5
                logger.info(f"[SENSITIVE] Partial exit condition met (50% profit): Ticket={position.ticket}, Symbol={position.symbol}, Profit={profit_pct:.2f}%, Closing 50% of volume={close_volume:.2f}")
                self.close_partial_position(position.ticket, close_volume)
        
        elif profit_pct >= 75:
            if position.volume > 0.01:
                close_volume = position.volume * 0.3
                logger.info(f"[SENSITIVE] Partial exit condition met (75% profit): Ticket={position.ticket}, Symbol={position.symbol}, Profit={profit_pct:.2f}%, Closing 30% of volume={close_volume:.2f}")
                self.close_partial_position(position.ticket, close_volume)
    
    def update_stop_loss(self, ticket: int, new_sl: float) -> bool:
        if mt5 is None:
            return False
        try:
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                return False
            
            position = position[0]
            
            if ticket in self.original_sl_tp:
                original_tp = self.original_sl_tp[ticket]['tp']
            else:
                original_tp = position.tp
                self.original_sl_tp[ticket] = {'sl': position.sl, 'tp': position.tp}
            
            if abs(position.tp - original_tp) > 0.00001 if original_tp > 0 else position.tp != original_tp:
                logger.warning(f"[SENSITIVE] TP changed by user detected: Ticket={ticket}, OriginalTP={original_tp:.5f}, CurrentTP={position.tp:.5f}. Restoring original TP.")
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": ticket,
                "sl": new_sl,
                "tp": original_tp,
            }
            
            old_sl = position.sl
            old_tp = position.tp
            
            logger.info(f"[SENSITIVE] Attempting to update SL: Ticket={ticket}, Symbol={position.symbol}, OldSL={old_sl:.5f}, NewSL={new_sl:.5f}, OldTP={old_tp:.5f}")
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[SENSITIVE] SL updated successfully: Ticket={ticket}, Symbol={position.symbol}, OldSL={old_sl:.5f}, NewSL={new_sl:.5f}, TP={old_tp:.5f}")
                
                is_trailing_stop = False
                if position.type == mt5.ORDER_TYPE_BUY:
                    is_trailing_stop = new_sl > old_sl
                elif position.type == mt5.ORDER_TYPE_SELL:
                    is_trailing_stop = (old_sl == 0 or new_sl < old_sl)
                
                if is_trailing_stop:
                    self.db.update_order(ticket, {'trailing_stop_applied': 1})
                    if ticket in self.original_sl_tp:
                        self.original_sl_tp[ticket]['sl'] = new_sl
                        logger.debug(f"[SENSITIVE] Updated original_sl_tp for ticket {ticket}: SL={new_sl:.5f}")
                
                if self.main_controller and self.main_controller.telegram_bot:
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(self._send_telegram_notification(
                                f"""📈 <b>Trailing Stop Updated</b>

📊 <b>Order Details:</b>
• Ticket: {ticket}
• Symbol: {position.symbol}
• Old SL: {old_sl:.5f}
• New SL: {new_sl:.5f}
• TP: {old_tp:.5f}"""
                            ))
                        else:
                            loop.run_until_complete(self._send_telegram_notification(
                                f"""📈 <b>Trailing Stop Updated</b>

📊 <b>Order Details:</b>
• Ticket: {ticket}
• Symbol: {position.symbol}
• Old SL: {old_sl:.5f}
• New SL: {new_sl:.5f}
• TP: {old_tp:.5f}"""
                            ))
                    except Exception as e:
                        logger.error(f"Error sending trailing stop notification: {e}")
                
                return True
            else:
                logger.warning(f"[SENSITIVE] Failed to update SL: Ticket={ticket}, Retcode={result.retcode}, Comment={result.comment}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating SL: {e}")
            return False
    
    def close_partial_position(self, ticket: int, volume: float) -> bool:
        if mt5 is None:
            return False
        try:
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                return False
            
            position = position[0]
            volume = round(volume, 2)
            
            if volume >= position.volume:
                return False
            
            order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info is None:
                logger.error(f"Symbol info for {position.symbol} not found")
                return False
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": volume,
                "type": order_type,
                "position": ticket,
                "deviation": 20,
                "magic": 234000,
                "comment": "GoldMan-Partial",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self.get_filling_mode(symbol_info),
            }
            
            old_volume = position.volume
            remaining_volume = old_volume - volume
            
            logger.info(f"[SENSITIVE] Attempting partial exit: Ticket={ticket}, Symbol={position.symbol}, OldVolume={old_volume:.2f}, CloseVolume={volume:.2f}, RemainingVolume={remaining_volume:.2f}")
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                account_info = self.conn_mgr.get_account_info()
                logger.info(f"[SENSITIVE] Partial exit successful: Ticket={ticket}, Symbol={position.symbol}, ClosedVolume={volume:.2f}, RemainingVolume={remaining_volume:.2f}, Balance={account_info.balance:.2f}, Equity={account_info.equity:.2f}")
                
                if self.main_controller and self.main_controller.telegram_bot:
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(self._send_telegram_notification(
                                f"""📉 <b>Partial Exit Executed</b>

📊 <b>Order Details:</b>
• Ticket: {ticket}
• Symbol: {position.symbol}
• Closed Volume: {volume:.2f}
• Remaining Volume: {remaining_volume:.2f}

💰 <b>Account:</b>
• Balance: ${account_info.balance:.2f}
• Equity: ${account_info.equity:.2f}"""
                            ))
                        else:
                            loop.run_until_complete(self._send_telegram_notification(
                                f"""📉 <b>Partial Exit Executed</b>

📊 <b>Order Details:</b>
• Ticket: {ticket}
• Symbol: {position.symbol}
• Closed Volume: {volume:.2f}
• Remaining Volume: {remaining_volume:.2f}

💰 <b>Account:</b>
• Balance: ${account_info.balance:.2f}
• Equity: ${account_info.equity:.2f}"""
                            ))
                    except Exception as e:
                        logger.error(f"Error sending partial exit notification: {e}")
                
                return True
            else:
                logger.warning(f"[SENSITIVE] Failed partial exit: Ticket={ticket}, Retcode={result.retcode}, Comment={result.comment}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing partial position: {e}")
            return False
    
    def check_close_and_reopen(self, position: Any, profit_pct: float, 
                               strategy_manager: StrategyManager, 
                               risk_manager: RiskManager, 
                               market_engine: MarketEngine) -> bool:
        if profit_pct < 80:
            return False
        
        if mt5 is None or self.data_provider is None:
            return False
        
        symbol = position.symbol
        direction = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
        entry_price = position.price_open
        current_price = position.price_current
        tp = position.tp
        sl = position.sl
        
        if sl == 0:
            return False
        
        current_rr = abs(tp - entry_price) / abs(entry_price - sl) if abs(entry_price - sl) > 0 else 0
        
        if current_rr < 2.0:
            logger.info(f"[CLOSE_REOPEN] Position at 80% profit but R/R ({current_rr:.2f}) < 2.0, not closing for reopen")
            return False
        
        timeframes = strategy_manager.strategy_timeframes.get(strategy_manager.current_strategy, [])
        if not timeframes:
            return False
        
        trend_confirmations = []
        strong_confirmations = 0
        
        for tf in timeframes[:3]:
            df = self.data_provider.get_ohlc_data(symbol, tf, 100)
            if df is None or df.empty:
                continue
            
            trend = market_engine.detect_trend(df)
            
            if direction == "BUY" and trend == "UP":
                strong_confirmations += 1
                trend_confirmations.append((tf.value, trend))
            elif direction == "SELL" and trend == "DOWN":
                strong_confirmations += 1
                trend_confirmations.append((tf.value, trend))
            else:
                trend_confirmations.append((tf.value, trend))
        
        if strong_confirmations < 2:
            logger.info(f"[CLOSE_REOPEN] Position at 80% profit but insufficient trend confirmation: StrongConfirmations={strong_confirmations}, TrendConfirmations={trend_confirmations}")
            return False
        
        new_signal = strategy_manager.analyze_market()
        if new_signal is None:
            logger.info(f"[CLOSE_REOPEN] Position at 80% profit but no new signal generated")
            return False
        
        if new_signal.direction != direction:
            logger.info(f"[CLOSE_REOPEN] Position at 80% profit but new signal direction ({new_signal.direction}) differs from current ({direction})")
            return False
        
        if new_signal.confidence < 2.0:
            logger.info(f"[CLOSE_REOPEN] Position at 80% profit but new signal R/R ({new_signal.confidence:.2f}) < 2.0")
            return False
        
        logger.info(f"[CLOSE_REOPEN] Conditions met for close and reopen: Profit={profit_pct:.2f}%, CurrentR/R={current_rr:.2f}, NewSignalR/R={new_signal.confidence:.2f}, StrongConfirmations={strong_confirmations}, TrendConfirmations={trend_confirmations}")
        return True
    
    def close_position(self, ticket: int) -> bool:
        if mt5 is None:
            return False
        try:
            position = mt5.positions_get(ticket=ticket)
            if position is None or len(position) == 0:
                logger.warning(f"[CLOSE_POSITION] Position {ticket} not found")
                return False
            
            position = position[0]
            symbol = position.symbol
            volume = position.volume
            direction = position.type
            
            order_type = mt5.ORDER_TYPE_SELL if direction == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            if direction == mt5.ORDER_TYPE_BUY:
                price = mt5.symbol_info_tick(symbol).bid
            else:
                price = mt5.symbol_info_tick(symbol).ask
            
            logger.info(f"[SENSITIVE] Attempting to close position: Ticket={ticket}, Symbol={symbol}, Volume={volume:.2f}, Direction={direction}, Price={price:.5f}")
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "position": ticket,
                "deviation": 20,
                "magic": 234000,
                "comment": "GoldMan-CloseReopen",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self.get_filling_mode(mt5.symbol_info(symbol)),
            }
            
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                account_info = self.conn_mgr.get_account_info()
                if account_info:
                    logger.info(f"[SENSITIVE] Position closed successfully: Ticket={ticket}, Symbol={symbol}, Volume={volume:.2f}, Balance={account_info.balance:.2f}, Equity={account_info.equity:.2f}")
                else:
                    logger.info(f"[SENSITIVE] Position closed successfully: Ticket={ticket}, Symbol={symbol}, Volume={volume:.2f}")
                
                cursor = self.db.conn.cursor()
                cursor.execute("SELECT * FROM trades WHERE ticket = ?", (ticket,))
                order_data = cursor.fetchone()
                
                if not self.db.update_order(ticket, {
                    'status': 'CLOSED',
                    'exit_time': datetime.now(),
                    'exit_price': price
                }):
                    logger.warning(f"Failed to update trade {ticket} in database after closing")
                
                if self.main_controller and self.main_controller.telegram_bot and order_data:
                    import asyncio
                    try:
                        profit = order_data['profit'] if 'profit' in order_data else 0.0
                        profit_emoji = "✅" if profit > 0 else "❌"
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(self._send_telegram_notification(
                                f"""{profit_emoji} <b>Order Closed</b>

📊 <b>Order Details:</b>
• Ticket: {ticket}
• Symbol: {symbol}
• Direction: {order_data['direction']}
• Entry: {order_data['entry_price']:.5f}
• Exit: {price:.5f}
• Profit/Loss: ${profit:.2f}

💰 <b>Account:</b>
• Balance: ${account_info.balance:.2f}
• Equity: ${account_info.equity:.2f}"""
                            ))
                        else:
                            loop.run_until_complete(self._send_telegram_notification(
                                f"""{profit_emoji} <b>Order Closed</b>

📊 <b>Order Details:</b>
• Ticket: {ticket}
• Symbol: {symbol}
• Direction: {order_data['direction']}
• Entry: {order_data['entry_price']:.5f}
• Exit: {price:.5f}
• Profit/Loss: ${profit:.2f}

💰 <b>Account:</b>
• Balance: ${account_info.balance:.2f}
• Equity: ${account_info.equity:.2f}"""
                            ))
                    except Exception as e:
                        logger.error(f"Error sending close position notification: {e}")
                
                self.current_position = None
                return True
            else:
                logger.warning(f"[SENSITIVE] Failed to close position: Ticket={ticket}, Retcode={result.retcode}, Comment={result.comment}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
