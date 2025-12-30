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
            timeframe_key_with_idx = f"{tf.value}_{idx}_{self.current_symbol}"
            
            should_analyze = True
            if self.current_strategy == StrategyType.SUPER_SCALP:
                if timeframe_key_with_idx in self._last_analysis_time:
                    time_since_last = current_time - self._last_analysis_time[timeframe_key_with_idx]
                    if time_since_last < 1:
                        should_analyze = False
                
                if should_analyze:
                    self._last_analysis_time[timeframe_key_with_idx] = current_time
            
            if not should_analyze:
                if self.current_strategy == StrategyType.SUPER_SCALP:
                    logger.debug(f"[ENTRY] Skipping analysis for {tf.value} due to debounce (idx={idx})")
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
                    trend_emoji = "📈" if trend == "UP" else "📉" if trend == "DOWN" else "🔄"
                    logger.info(f"📊 [ANALYSIS] {tf.value}: {trend_emoji} {trend} (before entry point calculation)")
                    self._last_trend_log_time[log_key] = current_time
            
            if trend == "SIDEWAYS":
                if idx < sl_tp_timeframe_index:
                    if self.current_strategy == StrategyType.SUPER_SCALP:
                        from ml.rl_engine import RLEngine
                        rl_engine_temp = RLEngine(self.risk_manager.db)
                        trend_weights_temp = rl_engine_temp.get_trend_weights(self.current_symbol, self.current_strategy.value)
                        
                        current_trend_weight = trend_weights_temp.get(f'trend_{idx}', 0.5 if idx == 0 else 0.25)
                        
                        other_weights = []
                        for other_idx in range(3):
                            if other_idx != idx:
                                other_weights.append(trend_weights_temp.get(f'trend_{other_idx}', 0.5 if other_idx == 0 else 0.25))
                        
                        sum_other_weights = sum(other_weights)
                        
                        timeframe_key = f"{tf.value}_{self.current_symbol}"
                        current_time = time.time()
                        
                        if sum_other_weights > current_trend_weight:
                            if timeframe_key not in self._last_sideways_log_time or current_time - self._last_sideways_log_time[timeframe_key] >= 30:
                                logger.info(f"[TREND] {tf.value}: {trend} (weight: {current_trend_weight:.2f}) - Low weight, continuing (other trends weight: {sum_other_weights:.2f})")
                                self._last_sideways_log_time[timeframe_key] = current_time
                        else:
                            if timeframe_key not in self._last_sideways_log_time or current_time - self._last_sideways_log_time[timeframe_key] >= 30:
                                logger.info(f"[TREND] {tf.value}: {trend} (weight: {current_trend_weight:.2f}) - Analysis stopped (SIDEWAYS detected before SL/TP timeframe)")
                                self._last_sideways_log_time[timeframe_key] = current_time
                            return None
                    else:
                        return None
            
            if self.current_strategy == StrategyType.SUPER_SCALP:
                if idx < 3:
                    trend_strengths.append(trend_strength)
                    from ml.rl_engine import RLEngine
                    rl_engine_temp = RLEngine(self.risk_manager.db)
                    trend_weights_temp = rl_engine_temp.get_trend_weights(self.current_symbol, self.current_strategy.value)
                    trend_key = f'trend_{idx}'
                    trend_weight = trend_weights_temp.get(trend_key, 0.4 if idx == 0 else (0.3 if idx == 1 else 0.2))
                    trend_emoji = "📈" if trend == "UP" else "📉" if trend == "DOWN" else "🔄"
                    logger.info(f"📊 [TREND] {tf.value}: {trend_emoji} {trend} (weight: {trend_weight:.2f})")
                    trends.append(trend)
                else:
                    trend_strengths.append(0.0)
                    trends.append(None)
            else:
                trend_strengths.append(0.0)
                trends.append(trend)
            
            entry_point = self.market_engine.find_entry_point(df, trend, tf)

            if hasattr(self.market_engine, 'identify_nodes_improved'):
                nodes = self.market_engine.identify_nodes_improved(df, lookback=20, min_amplitude=0.002)
            else:
                nodes = self.market_engine.identify_nodes(df)
            
            if self.current_strategy == StrategyType.SUPER_SCALP:
                trend_emoji = "📈" if trend == "UP" else "📉" if trend == "DOWN" else "🔄"
                logger.info(f"🎯 [ENTRY_POINT] {tf.value}: {trend_emoji} trend={trend}, entry_point={entry_point}, idx={idx}, sl_tp_idx={sl_tp_timeframe_index}")

            if entry_point is None:
                if idx < sl_tp_timeframe_index:
                    if self.current_strategy == StrategyType.SUPER_SCALP:
                        logger.info(f"❌ [ENTRY] No entry point found at {tf.value} (idx={idx}) - Analysis stopped")
                    return None
                else:
                    if self.current_strategy == StrategyType.SUPER_SCALP:
                        logger.info(f"⚠️ [ENTRY] No entry point found at {tf.value} (idx={idx}) - Skipping this timeframe")
            else:
                entry_points.append(entry_point)
                if self.current_strategy == StrategyType.SUPER_SCALP:
                    logger.info(f"✅ [ENTRY_POINT] Added: {tf.value} -> {entry_point:.5f} (total: {len(entry_points)}/4)")
            
            
            if self.current_strategy != StrategyType.SUPER_SCALP or idx < len(timeframes) - 1:
                current_price = df['close'].iloc[-1] if not df.empty else 0
                trend_details[tf.value] = {
                    'trend': trend,
                    'entry_point': entry_point,
                    'current_price': current_price
                }
        
        if len(entry_points) < 4:
            if self.current_strategy == StrategyType.SUPER_SCALP:
                logger.info(f"[ENTRY] Rejected - Not enough entry points found ({len(entry_points)}/4)")
            else:
                logger.info(f"[MARKET_ANALYSIS] Not enough entry points found ({len(entry_points)}/4). Skipping trade.")
            return None
        
        if len(trends) < 3:
            if self.current_strategy == StrategyType.SUPER_SCALP:
                logger.info(f"[ENTRY] Rejected - Not enough trends found ({len(trends)}/3)")
            else:
                logger.info(f"[MARKET_ANALYSIS] Not enough trends found ({len(trends)}/3). Skipping trade.")
            return None
        
        from ml.rl_engine import RLEngine
        rl_engine = RLEngine(self.risk_manager.db)

        entry_weights = rl_engine.get_entry_weights(self.current_symbol, self.current_strategy.value)
        trend_weights = rl_engine.get_trend_weights(self.current_symbol, self.current_strategy.value)

        technical_analysis = self.market_engine.analyze_technical_indicators(df)
        logger.info(f"⚡ [TECHNICAL] RSI={technical_analysis['rsi_value']:.1f} ({technical_analysis['rsi_signal']}), "
                   f"MACD={technical_analysis['macd_histogram']:.4f} ({technical_analysis['macd_signal']}), "
                   f"Combined: {technical_analysis['combined_signal']}")

        technical_signal = technical_analysis['combined_signal']
        
        if self.current_strategy == StrategyType.SUPER_SCALP:
            trend_weights_list = []
            for i in range(3):
                trend_key = f'trend_{i}'
                trend_weight = trend_weights.get(trend_key, 0.4 if i == 0 else (0.3 if i == 1 else 0.2))
                
                if i < len(trend_strengths):
                    trend_strength = trend_strengths[i]
                    learned_strength = rl_engine.get_trend_strength(self.current_symbol, self.current_strategy.value, timeframes[i].value)
                    if learned_strength > 0:
                        strength_factor = learned_strength / 100.0
                    else:
                        strength_factor = trend_strength / 100.0 if trend_strength > 0 else 0.5
                    
                    final_weight = trend_weight * strength_factor
                else:
                    final_weight = trend_weight
                
                trend_weights_list.append(final_weight)
            
            total_weights = sum(trend_weights_list)
            if total_weights > 0:
                trend_weights_list = [w / total_weights for w in trend_weights_list]
        else:
            trend_weights_list = []
            for i in range(3):
                trend_key = f'trend_{i}'
                trend_weight = trend_weights.get(trend_key, 0.5 if i == 0 else 0.25)
                trend_weights_list.append(trend_weight)
        
        timeframe_weights = []
        for i in range(4):
            if i < 3:
                timeframe_weights.append(trend_weights_list[i])
            else:
                timeframe_weights.append(trend_weights_list[2])
        
        weight_0 = trend_weights_list[0]
        weight_1 = trend_weights_list[1]
        weight_2 = trend_weights_list[2]
        
        if self.current_strategy == StrategyType.SUPER_SCALP:
            selected_entries = []
            selected_weights = []
            
            sum_1_2 = weight_1 + weight_2
            sum_0_1 = weight_0 + weight_1
            sum_0_2 = weight_0 + weight_2
            
            if weight_0 > sum_1_2:
                selected_entries = [entry_points[0]]
                selected_weights = [weight_0]
                logger.info(f"🎯 [ENTRY] Single timeframe selected (weight: {weight_0:.2f} > {sum_1_2:.2f}): {timeframes[0].value} @ {entry_points[0]:.5f}")
            elif weight_1 > (weight_0 + weight_2):
                selected_entries = [entry_points[1]]
                selected_weights = [weight_1]
                logger.info(f"🎯 [ENTRY] Single timeframe selected (weight: {weight_1:.2f} > {weight_0 + weight_2:.2f}): {timeframes[1].value} @ {entry_points[1]:.5f}")
            elif weight_2 > (weight_0 + weight_1):
                selected_entries = [entry_points[2], entry_points[3]]
                selected_weights = [weight_2, weight_2]
                logger.info(f"🎯 [ENTRY] Single timeframe selected (weight: {weight_2:.2f} > {weight_0 + weight_1:.2f}): {timeframes[2].value} @ {entry_points[2]:.5f}")
            elif sum_1_2 > weight_0:
                selected_entries = [entry_points[1], entry_points[2], entry_points[3]]
                selected_weights = [weight_1, weight_2, weight_2]
                logger.info(f"🎯 [ENTRY] Two timeframes selected (combined weight: {sum_1_2:.2f} > {weight_0:.2f}): {timeframes[1].value} + {timeframes[2].value}")
            elif sum_0_1 > weight_2:
                selected_entries = [entry_points[0], entry_points[1]]
                selected_weights = [weight_0, weight_1]
                logger.info(f"🎯 [ENTRY] Two timeframes selected (combined weight: {sum_0_1:.2f} > {weight_2:.2f}): {timeframes[0].value} + {timeframes[1].value}")
            elif sum_0_2 > weight_1:
                selected_entries = [entry_points[0], entry_points[2], entry_points[3]]
                selected_weights = [weight_0, weight_2, weight_2]
                logger.info(f"🎯 [ENTRY] Two timeframes selected (combined weight: {sum_0_2:.2f} > {weight_1:.2f}): {timeframes[0].value} + {timeframes[2].value}")
            else:
                selected_entries = entry_points
                selected_weights = timeframe_weights
                logger.info(f"🎯 [ENTRY] All timeframes selected (equal weights): Average of all entry points")
            
            if len(selected_entries) == 1:
                final_entry = selected_entries[0]
            else:
                if selected_weights and sum(selected_weights) > 0:
                    total_weight = sum(selected_weights)
                    normalized_weights = [w / total_weight for w in selected_weights]
                    if np is None:
                        final_entry = sum(normalized_weights[i] * selected_entries[i] for i in range(len(selected_entries)))
                    else:
                        weighted_entries = np.array([normalized_weights[i] * selected_entries[i] for i in range(len(selected_entries))])
                        final_entry = np.sum(weighted_entries)
                else:
                    final_entry = sum(entry_points) / len(entry_points) if entry_points else 0.0
        else:
            sum_1_2 = weight_1 + weight_2
            sum_0_1 = weight_0 + weight_1
            sum_0_2 = weight_0 + weight_2
            
            selected_entries = []
            selected_weights = []
            
            if weight_0 > sum_1_2:
                selected_entries = [entry_points[0]]
                selected_weights = [weight_0]
            elif weight_1 > (weight_0 + weight_2):
                selected_entries = [entry_points[1]]
                selected_weights = [weight_1]
            elif weight_2 > (weight_0 + weight_1):
                selected_entries = [entry_points[2], entry_points[3]]
                selected_weights = [weight_2, weight_2]
            elif sum_1_2 > weight_0:
                selected_entries = [entry_points[1], entry_points[2], entry_points[3]]
                selected_weights = [weight_1, weight_2, weight_2]
            elif sum_0_1 > weight_2:
                selected_entries = [entry_points[0], entry_points[1]]
                selected_weights = [weight_0, weight_1]
            elif sum_0_2 > weight_1:
                selected_entries = [entry_points[0], entry_points[2], entry_points[3]]
                selected_weights = [weight_0, weight_2, weight_2]
            else:
                selected_entries = entry_points
                selected_weights = timeframe_weights
            
            if len(selected_entries) == 1:
                final_entry = selected_entries[0]
            else:
                if selected_weights and sum(selected_weights) > 0:
                    total_weight = sum(selected_weights)
                    normalized_weights = [w / total_weight for w in selected_weights]
                    if np is None:
                        final_entry = sum(normalized_weights[i] * selected_entries[i] for i in range(len(selected_entries)))
                    else:
                        weighted_entries = np.array([normalized_weights[i] * selected_entries[i] for i in range(len(selected_entries))])
                        final_entry = np.sum(weighted_entries)
                else:
                    final_entry = sum(entry_points) / len(entry_points) if entry_points else 0.0
        
        trend_scores = {
            'UP': 0.0,
            'DOWN': 0.0,
            'SIDEWAYS': 0.0
        }
        
        for i in range(min(3, len(trends))):
            if trends[i] is not None:
                trend_key = f'trend_{i}'
                weight = trend_weights.get(trend_key, 0.4 if i == 0 else (0.3 if i == 1 else 0.2))
                trend = trends[i]
                if trend in trend_scores:
                    trend_scores[trend] += weight
                if self.current_strategy == StrategyType.SUPER_SCALP:
                    trend_emoji = "📈" if trend == "UP" else "📉" if trend == "DOWN" else "🔄"
                    logger.info(f"📈 [TREND_SCORE] {timeframes[i].value}: {trend_emoji} {trend} (weight: {weight:.2f}) -> {trend_scores[trend]:.2f}")
        
        dominant_trend = max(trend_scores, key=trend_scores.get)
        
        if self.current_strategy == StrategyType.SUPER_SCALP:
            dominant_emoji = "📈" if dominant_trend == "UP" else "📉" if dominant_trend == "DOWN" else "🔄"
            logger.info(f"📊 [TREND_SCORE] Final scores: {trend_scores}, Dominant: {dominant_emoji} {dominant_trend}")

        if dominant_trend == "SIDEWAYS":
            if self.current_strategy == StrategyType.SUPER_SCALP:
                if trend_scores["DOWN"] > 0.15 or trend_scores["UP"] > 0.15:
                    logger.info(f"⚠️ [ENTRY] Super Scalp: 🔄 SIDEWAYS dominant but DOWN/UP have weight > 0.15, continuing...")
                else:
                    logger.info(f"❌ [ENTRY] Rejected - Dominant trend is 🔄 SIDEWAYS")
                    return None
            else:
                logger.info(f"[ENTRY] Rejected - Dominant trend is SIDEWAYS")
                return None
        
        if self.current_strategy == StrategyType.SUPER_SCALP:
            trend_0 = trends[0] if len(trends) > 0 and trends[0] is not None else None
            trend_1 = trends[1] if len(trends) > 1 and trends[1] is not None else None
            trend_2 = trends[2] if len(trends) > 2 and trends[2] is not None else None
            
            weight_0 = trend_weights_list[0]
            weight_1 = trend_weights_list[1]
            weight_2 = trend_weights_list[2]
            
            compatible_pairs = []
            if trend_0 and trend_1 and trend_0 == trend_1 and trend_0 != "SIDEWAYS":
                combined_weight = weight_0 + weight_1
                if combined_weight >= 0.6:
                    compatible_pairs.append((0, 1, combined_weight, trend_0))
            if trend_0 and trend_2 and trend_0 == trend_2 and trend_0 != "SIDEWAYS":
                combined_weight = weight_0 + weight_2
                if combined_weight >= 0.6:
                    compatible_pairs.append((0, 2, combined_weight, trend_0))
            if trend_1 and trend_2 and trend_1 == trend_2 and trend_1 != "SIDEWAYS":
                combined_weight = weight_1 + weight_2
                if combined_weight >= 0.6:
                    compatible_pairs.append((1, 2, combined_weight, trend_1))
            
            if not compatible_pairs:
                logger.info(f"❌ [ENTRY] Rejected - No compatible trend pairs with combined weight >= 80%")
                trend0_emoji = "📈" if trend_0 == "UP" else "📉" if trend_0 == "DOWN" else "🔄"
                trend1_emoji = "📈" if trend_1 == "UP" else "📉" if trend_1 == "DOWN" else "🔄"
                trend2_emoji = "📈" if trend_2 == "UP" else "📉" if trend_2 == "DOWN" else "🔄"
                logger.info(f"  📊 Trend 0 (M5): {trend0_emoji} {trend_0} (weight: {weight_0:.2f})")
                logger.info(f"  📊 Trend 1 (M3): {trend1_emoji} {trend_1} (weight: {weight_1:.2f})")
                logger.info(f"  📊 Trend 2 (M1): {trend2_emoji} {trend_2} (weight: {weight_2:.2f})")
                logger.info(f"  ⚖️ Combined weights: M5+M3={weight_0+weight_1:.2f}, M5+M1={weight_0+weight_2:.2f}, M3+M1={weight_1+weight_2:.2f}")
                return None
            
            best_pair = max(compatible_pairs, key=lambda x: x[2])
            selected_tf_indices = [best_pair[0], best_pair[1], 3]
            selected_entry_points = [entry_points[i] for i in selected_tf_indices]
            
            final_entry = sum(selected_entry_points) / len(selected_entry_points)
            logger.info(f"[ENTRY] Selected timeframes: {timeframes[best_pair[0]].value} + {timeframes[best_pair[1]].value} + {timeframes[3].value} (combined weight: {best_pair[2]:.2f}, trend: {best_pair[3]})")
            logger.info(f"[ENTRY] Entry points: {[f'{ep:.5f}' for ep in selected_entry_points]}, Final entry: {final_entry:.5f}")
            
            direction = "BUY" if best_pair[3] == "UP" else "SELL"

        direction = "BUY" if dominant_trend == "UP" else "SELL"

        trend_confidence = trend_scores[dominant_trend] / sum(trend_scores.values()) if sum(trend_scores.values()) > 0 else 0.5

        if self.current_strategy == StrategyType.SUPER_SCALP:
            if self.current_strategy == StrategyType.SUPER_SCALP:
                direction_emoji = "🟢" if direction == "BUY" else "🔴"
                trend_emoji = "📈" if dominant_trend == "UP" else "📉" if dominant_trend == "DOWN" else "🔄"
                logger.info(f"📊 [TREND] {trend_emoji} {dominant_trend} (confidence: {trend_confidence:.2f}) | 🎯 [ENTRY] {direction_emoji} {direction} @ {final_entry:.2f}")
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
                logger.info("[ENTRY] Rejected - Invalid SL/TP calculated (SL=0.0 or TP=0.0)")
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
                rr_emoji = "✅" if rr_ratio >= min_rr_ratio else "❌"
                logger.info(f"⚖️ [R/R] {rr_emoji} {rr_ratio:.2f} (Min: {min_rr_ratio:.2f})")

        if rr_ratio < min_rr_ratio:
            if self.current_strategy == StrategyType.SUPER_SCALP:
                logger.info(f"❌ [ENTRY] Rejected - R/R={rr_ratio:.2f} < {min_rr_ratio:.2f}")
            else:
                logger.info(f"❌ [MARKET_ANALYSIS] R/R ratio ({rr_ratio:.2f}) is below minimum ({min_rr_ratio:.2f}). Trade rejected.")
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
            timeframes=[tf.value for tf in timeframes] if self.current_strategy == StrategyType.SUPER_SCALP else None,
            technical_signal=technical_signal
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
