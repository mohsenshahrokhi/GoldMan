from typing import Tuple, Dict, Optional, TYPE_CHECKING

try:
    import MetaTrader5 as mt5
    import numpy as np
    import pandas as pd
except ImportError:
    mt5 = None
    np = None
    pd = None

from datetime import datetime

if TYPE_CHECKING:
    from analysis.market_engine import MarketEngine

from utils.logger import logger
from config.constants import (
    MAX_RISK_PER_TRADE, MIN_LOT_SIZE, MAX_LOT_SIZE,
    MIN_RR_RATIO, DAILY_LOSS_LIMIT,
    DRAWDOWN_PROTECTION_1, DRAWDOWN_PROTECTION_2
)
from config.enums import TimeFrame


class RiskManager:
    
    def __init__(self, connection_manager, db_manager):
        self.conn_mgr = connection_manager
        self.db = db_manager
        self.daily_loss = 0.0
        self.last_reset_date = datetime.now().date()
        self.max_drawdown = 0.0
        self.initial_balance = 0.0
    
    def reset_daily_loss(self):
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_loss = 0.0
            self.last_reset_date = today
    
    def calculate_lot_size(self, stop_loss_pips: float, symbol: str) -> float:
        account_info = self.conn_mgr.get_account_info()
        if account_info is None:
            return MIN_LOT_SIZE
        
        max_risk = account_info.balance * MAX_RISK_PER_TRADE
        
        if mt5 is None:
            return MIN_LOT_SIZE
            
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return MIN_LOT_SIZE
        
        pip_value = symbol_info.trade_tick_value
        if symbol_info.digits == 3 or symbol_info.digits == 5:
            pip_value = pip_value * 10
        
        if stop_loss_pips == 0:
            return MIN_LOT_SIZE
        
        lot_size = max_risk / (stop_loss_pips * pip_value)
        
        lot_size = max(MIN_LOT_SIZE, min(MAX_LOT_SIZE, lot_size))
        
        lot_size = round(lot_size, 2)
        
        return lot_size
    
    def calculate_sl_tp_node_based(self, entry_price: float, direction: str, 
                                   symbol: str, df: pd.DataFrame, 
                                   market_engine: 'MarketEngine',
                                   safety_margin_points: float = 5.0,
                                   spread_factor: float = 1.0) -> Tuple[float, float]:
        if mt5 is None:
            return 0.0, 0.0
            
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0.0, 0.0
        
        spread = symbol_info.spread * symbol_info.point * spread_factor
        commission = getattr(symbol_info, 'commission', 0.0)
        if hasattr(symbol_info, 'commission_mode') and symbol_info.commission_mode == mt5.COMMISSION_MONEY:
            commission = commission / symbol_info.trade_contract_size if symbol_info.trade_contract_size > 0 else 0.0
        safety_margin = safety_margin_points * symbol_info.point
        
        adjustment = spread + safety_margin + commission
        
        if direction == "BUY":
            sl_node = market_engine.find_nearest_node(df, entry_price, "below")
            if sl_node is None:
                sl_node = entry_price * 0.995
            
            sl = sl_node - adjustment
            
            tp_node = market_engine.find_nearest_node(df, entry_price, "above")
            if tp_node is None:
                tp_node = entry_price * 1.005
            
            tp = tp_node - adjustment
        
        else:
            sl_node = market_engine.find_nearest_node(df, entry_price, "above")
            if sl_node is None:
                sl_node = entry_price * 1.005
            
            sl = sl_node + adjustment
            
            tp_node = market_engine.find_nearest_node(df, entry_price, "below")
            if tp_node is None:
                tp_node = entry_price * 0.995
            
            tp = tp_node + adjustment
        
        if sl <= 0 or tp <= 0:
            logger.warning(f"[SL_TP] Invalid SL/TP calculated: SL={sl}, TP={tp}, Entry={entry_price}, Direction={direction}")
            return 0.0, 0.0
        
        if direction == "BUY" and (sl >= entry_price or tp <= entry_price):
            logger.warning(f"[SL_TP] Invalid SL/TP for BUY: SL={sl}, TP={tp}, Entry={entry_price}")
            return 0.0, 0.0
        
        if direction == "SELL" and (sl <= entry_price or tp >= entry_price):
            logger.warning(f"[SL_TP] Invalid SL/TP for SELL: SL={sl}, TP={tp}, Entry={entry_price}")
            return 0.0, 0.0
        
        return sl, tp
    
    def calculate_sl_tp_atr_based(self, entry_price: float, direction: str, 
                                   df: pd.DataFrame, 
                                   atr_multiplier_sl: float = 2.0,
                                   atr_multiplier_tp: float = 2.0) -> Tuple[float, float]:
        if np is None or len(df) < 14:
            return 0.0, 0.0
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr_list = []
        for i in range(1, len(df)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            tr_list.append(tr)
        
        atr = np.mean(tr_list[-14:])
        
        if direction == "BUY":
            sl = entry_price - atr_multiplier_sl * atr
            tp = entry_price + atr_multiplier_tp * atr
        else:
            sl = entry_price + atr_multiplier_sl * atr
            tp = entry_price - atr_multiplier_tp * atr
        
        return sl, tp
    
    def calculate_sl_tp_garch_based(self, entry_price: float, direction: str, 
                                     df: pd.DataFrame,
                                     alpha_0: float = 0.0001,
                                     alpha_1: float = 0.1,
                                     beta_1: float = 0.8,
                                     k: float = 2.0) -> Tuple[float, float]:
        if np is None or len(df) < 20:
            return 0.0, 0.0
        
        returns = df['close'].pct_change().dropna().values
        
        sigma_squared = np.var(returns)
        for i in range(1, len(returns)):
            epsilon_sq = returns[i-1] ** 2
            sigma_squared = alpha_0 + alpha_1 * epsilon_sq + beta_1 * sigma_squared
        
        sigma = np.sqrt(sigma_squared)
        
        if direction == "BUY":
            sl = entry_price - k * sigma * entry_price
            tp = entry_price + k * sigma * entry_price
        else:
            sl = entry_price + k * sigma * entry_price
            tp = entry_price - k * sigma * entry_price
        
        return sl, tp
    
    def calculate_sl_tp_fixed_rr(self, entry_price: float, stop_loss: float, 
                                  rr_ratio: float = 2.0) -> float:
        sl_distance = abs(entry_price - stop_loss)
        tp_distance = rr_ratio * sl_distance
        
        if entry_price > stop_loss:
            tp = entry_price + tp_distance
        else:
            tp = entry_price - tp_distance
        
        return tp
    
    def calculate_final_sl_tp(self, entry_price: float, direction: str, symbol: str,
                             df: pd.DataFrame, market_engine: 'MarketEngine',
                             weights: Dict[str, float] = None,
                             parameters: Dict[str, float] = None,
                             strategy: str = None) -> Tuple[float, float]:
        if weights is None or not weights:
            weights = {
                'node': 0.25,
                'atr': 0.25,
                'garch': 0.25,
                'fixed_rr': 0.25
            }
        
        for key in ['node', 'atr', 'garch', 'fixed_rr']:
            if key not in weights:
                weights[key] = 0.25
        
        if parameters is None:
            parameters = {}
        
        if strategy == "SUPER_SCALP":
            atr_multiplier_sl = parameters.get('atr_multiplier_sl', 0.5)
            atr_multiplier_tp = parameters.get('atr_multiplier_tp', 1.0)
            garch_alpha_0 = parameters.get('garch_alpha_0', 0.0001)
            garch_alpha_1 = parameters.get('garch_alpha_1', 0.1)
            garch_beta_1 = parameters.get('garch_beta_1', 0.8)
            garch_k = parameters.get('garch_k', 1.0)
            node_safety_margin = parameters.get('node_safety_margin', 2.0)
            node_spread_factor = parameters.get('node_spread_factor', 1.0)
        elif strategy == "SCALP":
            atr_multiplier_sl = parameters.get('atr_multiplier_sl', 1.0)
            atr_multiplier_tp = parameters.get('atr_multiplier_tp', 2.0)
            garch_alpha_0 = parameters.get('garch_alpha_0', 0.0001)
            garch_alpha_1 = parameters.get('garch_alpha_1', 0.1)
            garch_beta_1 = parameters.get('garch_beta_1', 0.8)
            garch_k = parameters.get('garch_k', 1.5)
            node_safety_margin = parameters.get('node_safety_margin', 3.0)
            node_spread_factor = parameters.get('node_spread_factor', 1.0)
        else:
            atr_multiplier_sl = parameters.get('atr_multiplier_sl', 2.0)
            atr_multiplier_tp = parameters.get('atr_multiplier_tp', 2.0)
            garch_alpha_0 = parameters.get('garch_alpha_0', 0.0001)
            garch_alpha_1 = parameters.get('garch_alpha_1', 0.1)
            garch_beta_1 = parameters.get('garch_beta_1', 0.8)
            garch_k = parameters.get('garch_k', 2.0)
            node_safety_margin = parameters.get('node_safety_margin', 5.0)
            node_spread_factor = parameters.get('node_spread_factor', 1.0)
        
        sl_node, tp_node = self.calculate_sl_tp_node_based(
            entry_price, direction, symbol, df, market_engine,
            safety_margin_points=node_safety_margin,
            spread_factor=node_spread_factor
        )
        sl_atr, tp_atr = self.calculate_sl_tp_atr_based(
            entry_price, direction, df,
            atr_multiplier_sl=atr_multiplier_sl,
            atr_multiplier_tp=atr_multiplier_tp
        )
        sl_garch, tp_garch = self.calculate_sl_tp_garch_based(
            entry_price, direction, df,
            alpha_0=garch_alpha_0,
            alpha_1=garch_alpha_1,
            beta_1=garch_beta_1,
            k=garch_k
        )
        
        valid_sl_values = []
        valid_tp_values = []
        
        if strategy == "SUPER_SCALP":
            max_sl_distance = entry_price * 0.01
            max_tp_distance = entry_price * 0.02
        elif strategy == "SCALP":
            max_sl_distance = entry_price * 0.02
            max_tp_distance = entry_price * 0.04
        else:
            max_sl_distance = entry_price * 0.05
            max_tp_distance = entry_price * 0.10
        
        if sl_node != 0.0:
            if (direction == "BUY" and sl_node < entry_price) or (direction == "SELL" and sl_node > entry_price):
                if abs(sl_node - entry_price) <= max_sl_distance and sl_node > 0:
                    valid_sl_values.append(('node', sl_node, weights['node']))
        
        if sl_atr != 0.0:
            if (direction == "BUY" and sl_atr < entry_price) or (direction == "SELL" and sl_atr > entry_price):
                if abs(sl_atr - entry_price) <= max_sl_distance and sl_atr > 0:
                    valid_sl_values.append(('atr', sl_atr, weights['atr']))
        
        if sl_garch != 0.0:
            if (direction == "BUY" and sl_garch < entry_price) or (direction == "SELL" and sl_garch > entry_price):
                if abs(sl_garch - entry_price) <= max_sl_distance and sl_garch > 0:
                    valid_sl_values.append(('garch', sl_garch, weights['garch']))
        
        if tp_node != 0.0:
            if (direction == "BUY" and tp_node > entry_price) or (direction == "SELL" and tp_node < entry_price):
                if abs(tp_node - entry_price) <= max_tp_distance and tp_node > 0:
                    valid_tp_values.append(('node', tp_node, weights['node']))
        
        if tp_atr != 0.0:
            if (direction == "BUY" and tp_atr > entry_price) or (direction == "SELL" and tp_atr < entry_price):
                if abs(tp_atr - entry_price) <= max_tp_distance and tp_atr > 0:
                    valid_tp_values.append(('atr', tp_atr, weights['atr']))
        
        if tp_garch != 0.0:
            if (direction == "BUY" and tp_garch > entry_price) or (direction == "SELL" and tp_garch < entry_price):
                if abs(tp_garch - entry_price) <= max_tp_distance and tp_garch > 0:
                    valid_tp_values.append(('garch', tp_garch, weights['garch']))
        
        if not valid_sl_values or not valid_tp_values:
            if sl_atr != 0.0 and tp_atr != 0.0:
                return sl_atr, tp_atr
            elif sl_node != 0.0 and tp_node != 0.0:
                return sl_node, tp_node
            else:
                return 0.0, 0.0
        
        total_sl_weight = sum(w for _, _, w in valid_sl_values)
        total_tp_weight = sum(w for _, _, w in valid_tp_values)
        
        sl_final = sum(sl * w for _, sl, w in valid_sl_values) / total_sl_weight if total_sl_weight > 0 else entry_price * 0.99
        tp_weighted = sum(tp * w for _, tp, w in valid_tp_values) / total_tp_weight if total_tp_weight > 0 else entry_price * 1.01
        
        min_rr_ratio = parameters.get('min_rr_ratio', MIN_RR_RATIO)
        tp_fixed = self.calculate_sl_tp_fixed_rr(entry_price, sl_final, min_rr_ratio)
        
        if tp_fixed != 0.0 and ((direction == "BUY" and tp_fixed > entry_price) or (direction == "SELL" and tp_fixed < entry_price)):
            if abs(tp_fixed - entry_price) < entry_price * 0.1:
                total_tp_weight_final = total_tp_weight + weights.get('fixed_rr', 0.25)
                tp_final = (tp_weighted * total_tp_weight + tp_fixed * weights.get('fixed_rr', 0.25)) / total_tp_weight_final
            else:
                tp_final = tp_weighted
        else:
            tp_final = tp_weighted
        
        current_rr = abs(tp_final - entry_price) / abs(entry_price - sl_final) if abs(entry_price - sl_final) > 0 else 0
        
        if current_rr < min_rr_ratio:
            tp_fixed_forced = self.calculate_sl_tp_fixed_rr(entry_price, sl_final, min_rr_ratio)
            if tp_fixed_forced != 0.0:
                if (direction == "BUY" and tp_fixed_forced > entry_price) or (direction == "SELL" and tp_fixed_forced < entry_price):
                    max_tp_for_forced = entry_price * 0.03 if strategy == "SUPER_SCALP" else (entry_price * 0.05 if strategy == "SCALP" else entry_price * 0.15)
                    if abs(tp_fixed_forced - entry_price) < max_tp_for_forced:
                        tp_final = tp_fixed_forced
                        logger.debug(f"[SL_TP] Forced TP adjustment to meet min R/R: OldTP={tp_weighted:.5f}, NewTP={tp_final:.5f}, R/R={min_rr_ratio:.2f}")
        
        if strategy == "SUPER_SCALP":
            if direction == "BUY":
                if sl_final < entry_price - max_sl_distance:
                    sl_final = entry_price - max_sl_distance
                if tp_final > entry_price + max_tp_distance:
                    tp_final = entry_price + max_tp_distance
            else:
                if sl_final > entry_price + max_sl_distance:
                    sl_final = entry_price + max_sl_distance
                if tp_final < entry_price - max_tp_distance:
                    tp_final = entry_price - max_tp_distance
        elif strategy == "SCALP":
            if direction == "BUY":
                if sl_final < entry_price - max_sl_distance:
                    sl_final = entry_price - max_sl_distance
                if tp_final > entry_price + max_tp_distance:
                    tp_final = entry_price + max_tp_distance
            else:
                if sl_final > entry_price + max_sl_distance:
                    sl_final = entry_price + max_sl_distance
                if tp_final < entry_price - max_tp_distance:
                    tp_final = entry_price - max_tp_distance
        
        if direction == "BUY" and (sl_final >= entry_price or tp_final <= entry_price):
            return 0.0, 0.0
        
        if direction == "SELL" and (sl_final <= entry_price or tp_final >= entry_price):
            return 0.0, 0.0
        
        return sl_final, tp_final
    
    def check_daily_loss_limit(self) -> bool:
        self.reset_daily_loss()
        account_info = self.conn_mgr.get_account_info()
        if account_info is None:
            return False
        
        if account_info.profit < 0:
            self.daily_loss = abs(account_info.profit)
        
        if self.daily_loss / account_info.balance > DAILY_LOSS_LIMIT:
            logger.warning(f"[SENSITIVE] Daily loss limit exceeded: Balance={account_info.balance:.2f}, DailyLoss={self.daily_loss:.2f}, LossPercentage={(self.daily_loss/account_info.balance)*100:.2f}%, Limit={DAILY_LOSS_LIMIT*100:.2f}%")
            return False
        
        return True
    
    def check_drawdown(self) -> Tuple[bool, float]:
        account_info = self.conn_mgr.get_account_info()
        if account_info is None:
            return True, 0.0
        
        if self.initial_balance == 0:
            self.initial_balance = account_info.balance
        
        current_drawdown = (self.initial_balance - account_info.equity) / self.initial_balance
        
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        if current_drawdown > DRAWDOWN_PROTECTION_2:
            logger.warning(f"[SENSITIVE] Drawdown exceeds 5% - Full stop: InitialBalance={self.initial_balance:.2f}, CurrentEquity={account_info.equity:.2f}, Drawdown={current_drawdown*100:.2f}%, MaxDrawdown={self.max_drawdown*100:.2f}%")
            return False, 0.5
        
        if current_drawdown > DRAWDOWN_PROTECTION_1:
            logger.warning(f"[SENSITIVE] Drawdown exceeds 3% - Volume reduction: InitialBalance={self.initial_balance:.2f}, CurrentEquity={account_info.equity:.2f}, Drawdown={current_drawdown*100:.2f}%, MaxDrawdown={self.max_drawdown*100:.2f}%")
            return True, 0.5
        
        return True, 1.0
