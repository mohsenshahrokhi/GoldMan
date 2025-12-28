from typing import Optional, Tuple
import time

try:
    import numpy as np
except ImportError:
    np = None

from utils.logger import logger
from utils.data_classes import OrderSignal
from config.enums import StrategyType, TimeFrame
from config.constants import MIN_RR_RATIO
from market.data_provider import MarketDataProvider
from analysis.market_engine import MarketEngine
from risk.risk_manager import RiskManager


class StrategyManager:
    
    def __init__(self, market_data_provider: MarketDataProvider, 
                 market_engine: MarketEngine, risk_manager: RiskManager):
        self.data_provider = market_data_provider
        self.market_engine = market_engine
        self.risk_manager = risk_manager
        self.current_strategy = None
        self.current_symbol = None
        self._last_sideways_log_time = {}
        self._last_trend_log_time = {}
        self._last_analysis_time = {}
        self._last_trend_log_time = {}
        self._last_analysis_time = {}
        
        self.strategy_timeframes = {
            StrategyType.DAY_TRADING: [
                TimeFrame.H4,
                TimeFrame.H1,
                TimeFrame.M15,
                TimeFrame.M5
            ],
            StrategyType.SCALP: [
                TimeFrame.H1,
                TimeFrame.M15,
                TimeFrame.M5,
                TimeFrame.M1
            ],
            StrategyType.SUPER_SCALP: [
                TimeFrame.M5,
                TimeFrame.M3,
                TimeFrame.M1,
                TimeFrame.M1
            ]
        }
        
        self.min_analysis_timeframe = {
            StrategyType.DAY_TRADING: TimeFrame.M15,
            StrategyType.SCALP: TimeFrame.M5,
            StrategyType.SUPER_SCALP: TimeFrame.M3
        }
        
        self.weakness_timeframes = {
            StrategyType.DAY_TRADING: (TimeFrame.H1, TimeFrame.M15),
            StrategyType.SCALP: (TimeFrame.M15, TimeFrame.M5),
            StrategyType.SUPER_SCALP: (TimeFrame.M5, TimeFrame.M3)
        }
    
    def set_strategy(self, strategy: StrategyType, symbol: str):
        self.current_strategy = strategy
        self.current_symbol = symbol
        logger.info(f"Strategy {strategy.value} set for symbol {symbol}")
    
    def analyze_market(self) -> Optional[OrderSignal]:
        if self.current_strategy is None or self.current_symbol is None:
            return None
        
        timeframes = self.strategy_timeframes[self.current_strategy]
        entry_points = []
        trends = []
        trend_strengths = []
        trend_details = {}
        
        sl_tp_timeframe = None
        if self.current_strategy == StrategyType.DAY_TRADING:
            sl_tp_timeframe = TimeFrame.M15
        elif self.current_strategy == StrategyType.SCALP:
            sl_tp_timeframe = TimeFrame.M5
        else:
            sl_tp_timeframe = TimeFrame.M1
        
        sl_tp_timeframe_index = None
        for idx, tf in enumerate(timeframes):
            if tf == sl_tp_timeframe:
                sl_tp_timeframe_index = idx
                break
        
        if sl_tp_timeframe_index is None:
            logger.error(f"[MARKET_ANALYSIS] SL/TP timeframe {sl_tp_timeframe.value} not found in strategy timeframes")
            return None
        
        for idx, tf in enumerate(timeframes):
            df = self.data_provider.get_ohlc_data(self.current_symbol, tf, 500)
            if df is None or df.empty:
                if idx <= sl_tp_timeframe_index:
                    return None
                continue
            
            current_time = time.time()
            timeframe_key = f"{tf.value}_{self.current_symbol}"
            
            should_analyze = True
            if self.current_strategy == StrategyType.SUPER_SCALP:
                if timeframe_key in self._last_analysis_time:
                    time_since_last = current_time - self._last_analysis_time[timeframe_key]
                    if time_since_last < 1:
                        should_analyze = False
                
                if should_analyze:
                    self._last_analysis_time[timeframe_key] = current_time
            
            if not should_analyze:
                continue
            
            if self.current_strategy == StrategyType.SUPER_SCALP:
                trend, trend_strength = self.market_engine.detect_trend_with_strength(df)
            else:
                trend = self.market_engine.detect_trend(df)
                trend_strength = 0.0
            
            if self.current_strategy == StrategyType.SUPER_SCALP:
                should_log_trend = True
                log_key = f"{timeframe_key}_trend"
                if log_key in self._last_trend_log_time:
                    time_since_last_log = current_time - self._last_trend_log_time[log_key]
                    if time_since_last_log < 300:
                        should_log_trend = False
                
                if should_log_trend:
                    logger.info(f"[ANALYSIS] {tf.value}: {trend} (before entry point calculation)")
                    self._last_trend_log_time[log_key] = current_time
            
            if trend == "SIDEWAYS":
                if idx < sl_tp_timeframe_index:
                    if self.current_strategy == StrategyType.SUPER_SCALP:
                        if idx == 0:
                            timeframe_key = f"{tf.value}_{self.current_symbol}"
                            current_time = time.time()
                            if timeframe_key not in self._last_sideways_log_time or current_time - self._last_sideways_log_time[timeframe_key] >= 30:
                                from ml.rl_engine import RLEngine
                                rl_engine_temp = RLEngine(self.risk_manager.db)
                                learned_strength = rl_engine_temp.get_trend_strength(self.current_symbol, self.current_strategy.value, tf.value)
                                if learned_strength > 0:
                                    logger.info(f"[TREND] {tf.value}: {trend} ({learned_strength:.1f}%) - Analysis stopped (SIDEWAYS detected before SL/TP timeframe)")
                                else:
                                    logger.info(f"[TREND] {tf.value}: {trend} ({trend_strength:.1f}%) - Analysis stopped (SIDEWAYS detected before SL/TP timeframe)")
                                self._last_sideways_log_time[timeframe_key] = current_time
                            return None
                    else:
                        return None
            
            if self.current_strategy == StrategyType.SUPER_SCALP:
                trend_strengths.append(trend_strength)
                if idx < 3:
                    from ml.rl_engine import RLEngine
                    rl_engine_temp = RLEngine(self.risk_manager.db)
                    learned_strength = rl_engine_temp.get_trend_strength(self.current_symbol, self.current_strategy.value, tf.value)
                    if learned_strength > 0:
                        logger.info(f"[TREND] {tf.value}: {trend} ({learned_strength:.1f}%)")
                    else:
                        logger.info(f"[TREND] {tf.value}: {trend} ({trend_strength:.1f}%)")
            else:
                trend_strengths.append(0.0)
            
            trends.append(trend)
            
            entry_point = self.market_engine.find_entry_point(df, trend, tf)
            
            if entry_point is None:
                if idx < sl_tp_timeframe_index:
                    if self.current_strategy == StrategyType.SUPER_SCALP:
                        logger.debug(f"[ENTRY] No entry point found at {tf.value} - Analysis stopped")
                    return None
            else:
                entry_points.append(entry_point)
            
            
            if self.current_strategy != StrategyType.SUPER_SCALP or idx < len(timeframes) - 1:
                current_price = df['close'].iloc[-1] if not df.empty else 0
                trend_details[tf.value] = {
                    'trend': trend,
                    'entry_point': entry_point,
                    'current_price': current_price
                }
        
        if len(entry_points) < 4:
            if self.current_strategy != StrategyType.SUPER_SCALP:
                logger.info(f"[MARKET_ANALYSIS] Not enough entry points found ({len(entry_points)}/4). Skipping trade.")
            return None
        
        if len(trends) < 3:
            if self.current_strategy != StrategyType.SUPER_SCALP:
                logger.info(f"[MARKET_ANALYSIS] Not enough trends found ({len(trends)}/3). Skipping trade.")
            return None
        
        from ml.rl_engine import RLEngine
        rl_engine = RLEngine(self.risk_manager.db)
        
        entry_weights = rl_engine.get_entry_weights(self.current_symbol, self.current_strategy.value)
        trend_weights = rl_engine.get_trend_weights(self.current_symbol, self.current_strategy.value)
        
        base_weights = [
            entry_weights.get('entry_0', 0.25),
            entry_weights.get('entry_1', 0.25),
            entry_weights.get('entry_2', 0.25),
            entry_weights.get('entry_3', 0.25)
        ]
        
        adjusted_weights = []
        for i in range(4):
            base_weight = base_weights[i]
            if self.current_strategy == StrategyType.SUPER_SCALP and i < len(trend_strengths):
                trend_strength = trend_strengths[i]
                learned_strength = rl_engine.get_trend_strength(self.current_symbol, self.current_strategy.value, timeframes[i].value)
                if learned_strength > 0:
                    strength_factor = learned_strength / 100.0
                else:
                    strength_factor = trend_strength / 100.0 if trend_strength > 0 else 0.5
                
                adjusted_weight = base_weight * strength_factor
            else:
                adjusted_weight = base_weight
            
            adjusted_weights.append(adjusted_weight)
        
        total_weight = sum(adjusted_weights)
        if total_weight > 0:
            adjusted_weights = [w / total_weight for w in adjusted_weights]
        else:
            adjusted_weights = base_weights
        
        if np is None:
            final_entry = (
                adjusted_weights[0] * entry_points[0] +
                adjusted_weights[1] * entry_points[1] +
                adjusted_weights[2] * entry_points[2] +
                adjusted_weights[3] * entry_points[3]
            )
        else:
            weighted_entries = np.array([
                adjusted_weights[0] * entry_points[0],
                adjusted_weights[1] * entry_points[1],
                adjusted_weights[2] * entry_points[2],
                adjusted_weights[3] * entry_points[3]
            ])
            final_entry = np.sum(weighted_entries)
        
        trend_scores = {
            'UP': 0.0,
            'DOWN': 0.0,
            'SIDEWAYS': 0.0
        }
        
        for i in range(min(3, len(trends))):
            trend_key = f'trend_{i}'
            weight = trend_weights.get(trend_key, 0.33)
            trend = trends[i]
            if trend in trend_scores:
                trend_scores[trend] += weight
        
        dominant_trend = max(trend_scores, key=trend_scores.get)
        direction = "BUY" if dominant_trend == "UP" else "SELL"
        
        trend_confidence = trend_scores[dominant_trend] / sum(trend_scores.values()) if sum(trend_scores.values()) > 0 else 0.5
        
        if self.current_strategy == StrategyType.SUPER_SCALP:
            if self.current_strategy == StrategyType.SUPER_SCALP:
                logger.info(f"[TREND] {dominant_trend} | [ENTRY] {direction} @ {final_entry:.2f}")
        else:
            trend_distribution = {t: trends.count(t) for t in set(trends)}
            logger.info(f"[MARKET_ANALYSIS] Market Regime: Trend={dominant_trend}, Distribution={trend_distribution}, Direction={direction}, Entry={final_entry:.5f}")
        
        df_sl_tp = self.data_provider.get_ohlc_data(self.current_symbol, sl_tp_timeframe, 500)
        if df_sl_tp is None:
            return None
        
        weights = rl_engine.get_weights(self.current_symbol, self.current_strategy.value)
        parameters = rl_engine.get_parameters(self.current_symbol, self.current_strategy.value)
        
        sl, tp = self.risk_manager.calculate_final_sl_tp(
            final_entry, direction, self.current_symbol, df_sl_tp, 
            self.market_engine, weights, parameters, self.current_strategy.value
        )
        
        if sl == 0.0 or tp == 0.0:
            if self.current_strategy == StrategyType.SUPER_SCALP:
                logger.debug("[ENTRY] Invalid SL/TP calculated")
            return None
        
        symbol_info = self.data_provider.get_symbol_info(self.current_symbol)
        if symbol_info is None:
            return None
        
        sl_distance_pips = abs(final_entry - sl) / symbol_info.point
        if symbol_info.digits == 3 or symbol_info.digits == 5:
            sl_distance_pips = sl_distance_pips / 10
        
        tp_distance_pips = abs(tp - final_entry) / symbol_info.point
        if symbol_info.digits == 3 or symbol_info.digits == 5:
            tp_distance_pips = tp_distance_pips / 10
        
        lot_size = self.risk_manager.calculate_lot_size(sl_distance_pips, self.current_symbol)
        
        profit_potential = abs(tp - final_entry)
        loss_potential = abs(sl - final_entry)
        rr_ratio = profit_potential / loss_potential if loss_potential > 0 else 0
        
        min_rr_ratio = parameters.get('min_rr_ratio', MIN_RR_RATIO)
        
        if self.current_strategy == StrategyType.SUPER_SCALP:
            if self.current_strategy == StrategyType.SUPER_SCALP:
                logger.info(f"[R/R] {rr_ratio:.2f} (Min: {min_rr_ratio:.2f})")
        
        if rr_ratio < min_rr_ratio:
            if self.current_strategy == StrategyType.SUPER_SCALP:
                if self.current_strategy != StrategyType.SUPER_SCALP:
                    logger.info(f"[ENTRY] Rejected - R/R={rr_ratio:.2f} < {min_rr_ratio:.2f}")
            else:
                logger.info(f"[MARKET_ANALYSIS] R/R ratio ({rr_ratio:.2f}) is below minimum ({min_rr_ratio:.2f}). Trade rejected.")
            return None
        
        if self.current_strategy == StrategyType.SUPER_SCALP:
            logger.info(f"[ENTRY] {direction} Entry={final_entry:.2f} SL={sl:.2f} TP={tp:.2f} R/R={rr_ratio:.2f}")
        else:
            logger.info(f"[MARKET_ANALYSIS] Trade Signal: {direction} Entry={final_entry:.5f}, SL={sl:.5f}, TP={tp:.5f}, R/R={rr_ratio:.2f}, Lot={lot_size:.2f}")
        
        signal = OrderSignal(
            symbol=self.current_symbol,
            direction=direction,
            entry_price=final_entry,
            stop_loss=sl,
            take_profit=tp,
            lot_size=lot_size,
            timeframe=sl_tp_timeframe,
            confidence=rr_ratio,
            entry_points=entry_points if self.current_strategy == StrategyType.SUPER_SCALP else None,
            trends=trends[:3] if self.current_strategy == StrategyType.SUPER_SCALP and len(trends) >= 3 else None,
            timeframes=[tf.value for tf in timeframes] if self.current_strategy == StrategyType.SUPER_SCALP else None
        )
        
        return signal
    
    def should_analyze(self) -> bool:
        if self.current_strategy is None:
            return False
        
        min_tf = self.min_analysis_timeframe[self.current_strategy]
        
        if self.current_strategy == StrategyType.SUPER_SCALP:
            return True
        else:
            is_new = self.data_provider.is_new_candle(self.current_symbol, min_tf)
            if is_new:
                logger.info(f"[ANALYSIS] New candle detected on {min_tf.value} for {self.current_symbol}. Analysis will proceed.")
            return is_new
    
    def check_trend_weakness(self) -> Tuple[bool, Optional[float]]:
        if self.current_strategy is None or self.current_symbol is None:
            return False, None
        
        tf_weakness, tf_confirmation = self.weakness_timeframes[self.current_strategy]
        
        df_weakness = self.data_provider.get_ohlc_data(self.current_symbol, tf_weakness, 100)
        if df_weakness is None:
            return False, None
        
        weakness_detected = self.market_engine.detect_trend_weakness(df_weakness, tf_weakness)
        
        if weakness_detected:
            df_confirmation = self.data_provider.get_ohlc_data(self.current_symbol, tf_confirmation, 100)
            if df_confirmation is None:
                return False, None
            
            trend_change = self.market_engine.detect_trend(df_confirmation) != self.market_engine.detect_trend(df_weakness)
            
            if trend_change:
                weakness_price = df_weakness['close'].iloc[-1]
                return True, weakness_price
        
        return False, None
