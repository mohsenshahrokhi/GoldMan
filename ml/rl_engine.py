from typing import Dict, Optional, List, Tuple
from collections import deque
import json
import math

from utils.logger import logger
from database.manager import DatabaseManager


class RLEngine:
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.initial_learning_rate = 0.01
        self.learning_rate = 0.01
        self.learning_rate_decay = 0.995
        self.min_learning_rate = 0.001
        self.gamma = 0.95
        self.trade_count_threshold = 20
        self.experience_replay_buffer_size = 1000
        self.experience_replay_batch_size = 32
        self.experience_buffers = {}
        self.optimization_count = {}
        
        self.default_weights = {
            'node': 0.25,
            'atr': 0.25,
            'garch': 0.25,
            'fixed_rr': 0.25
        }
        
        self.default_entry_weights = {
            'entry_0': 0.25,
            'entry_1': 0.25,
            'entry_2': 0.25,
            'entry_3': 0.25
        }
        
        self.default_trend_weights = {
            'trend_0': 0.33,
            'trend_1': 0.33,
            'trend_2': 0.34
        }
        
        self.default_parameters = {
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 2.0,
            'garch_alpha_0': 0.0001,
            'garch_alpha_1': 0.1,
            'garch_beta_1': 0.8,
            'garch_k': 2.0,
            'node_safety_margin': 5.0,
            'node_spread_factor': 1.0,
            'min_rr_ratio': 1.5,
            'max_risk_per_trade': 0.005,
            'nds_lookback': 20.0,
            'nds_trend_period': 20.0,
            'trailing_stop_threshold_1': 10.0,
            'trailing_stop_threshold_2': 15.0,
            'trailing_stop_threshold_3': 50.0,
            'trailing_stop_threshold_4': 75.0,
            'trailing_stop_threshold_5': 80.0,
            'partial_exit_threshold_1': 50.0,
            'partial_exit_threshold_2': 75.0,
            'partial_exit_percentage_1': 0.5,
            'partial_exit_percentage_2': 0.3
        }
        
        self.parameter_constraints = {
            'atr_multiplier_sl': (1.0, 4.0),
            'atr_multiplier_tp': (1.0, 4.0),
            'garch_alpha_0': (0.00001, 0.001),
            'garch_alpha_1': (0.05, 0.2),
            'garch_beta_1': (0.7, 0.9),
            'garch_k': (1.5, 3.0),
            'node_safety_margin': (3.0, 10.0),
            'node_spread_factor': (0.5, 2.0),
            'min_rr_ratio': (1.2, 2.5),
            'max_risk_per_trade': (0.001, 0.05),
            'nds_lookback': (10.0, 50.0),
            'nds_trend_period': (10.0, 30.0),
            'trailing_stop_threshold_1': (5.0, 15.0),
            'trailing_stop_threshold_2': (10.0, 20.0),
            'trailing_stop_threshold_3': (40.0, 60.0),
            'trailing_stop_threshold_4': (70.0, 80.0),
            'trailing_stop_threshold_5': (75.0, 85.0),
            'partial_exit_threshold_1': (40.0, 60.0),
            'partial_exit_threshold_2': (70.0, 80.0),
            'partial_exit_percentage_1': (0.3, 0.7),
            'partial_exit_percentage_2': (0.2, 0.5)
        }
    
    def calculate_reward(self, trade_profit: float, transaction_cost: float, 
                        risk_penalty: float, missed_opportunity: float = 0.0,
                        rr_ratio: float = 1.0, hold_time: float = 0.0,
                        max_profit: float = 0.0, drawdown: float = 0.0) -> float:
        base_reward = trade_profit - transaction_cost - risk_penalty - missed_opportunity
        
        rr_bonus = 0.0
        if rr_ratio >= 2.0:
            rr_bonus = trade_profit * 0.1
        elif rr_ratio >= 1.5:
            rr_bonus = trade_profit * 0.05
        
        time_penalty = 0.0
        if hold_time > 24 * 3600:
            time_penalty = trade_profit * 0.02
        
        drawdown_penalty = drawdown * 0.1 if drawdown > 0 else 0.0
        
        max_profit_bonus = 0.0
        if max_profit > 0 and trade_profit > 0:
            profit_ratio = trade_profit / max_profit if max_profit > 0 else 0
            if profit_ratio >= 0.8:
                max_profit_bonus = trade_profit * 0.05
        
        total_reward = base_reward + rr_bonus - time_penalty - drawdown_penalty + max_profit_bonus
        
        return total_reward
    
    def get_weights(self, symbol: str, strategy: str) -> Dict[str, float]:
        weights = self.db.get_rl_weights(symbol, strategy)
        if not weights:
            weights = self.default_weights.copy()
            self.save_weights(symbol, strategy, weights)
        return weights
    
    def get_parameters(self, symbol: str, strategy: str) -> Dict[str, float]:
        params = self.db.get_rl_weights(symbol, strategy)
        result = self.default_parameters.copy()
        
        for param_name in self.default_parameters.keys():
            if param_name in params:
                result[param_name] = params[param_name]
        
        return result
    
    def save_parameters(self, symbol: str, strategy: str, parameters: Dict[str, float]):
        self.db.save_rl_weights(symbol, strategy, parameters)
    
    def save_weights(self, symbol: str, strategy: str, weights: Dict[str, float]):
        self.db.save_rl_weights(symbol, strategy, weights)
    
    def get_entry_weights(self, symbol: str, strategy: str) -> Dict[str, float]:
        weights = self.db.get_rl_entry_weights(symbol, strategy)
        if not weights:
            weights = self.default_entry_weights.copy()
            self.save_entry_weights(symbol, strategy, weights)
        
        result = self.default_entry_weights.copy()
        for weight_name in self.default_entry_weights.keys():
            if weight_name in weights:
                result[weight_name] = weights[weight_name]
        
        total = sum(result.values())
        if total > 0:
            for key in result:
                result[key] /= total
        else:
            result = self.default_entry_weights.copy()
        
        return result
    
    def get_trend_weights(self, symbol: str, strategy: str) -> Dict[str, float]:
        weights = self.db.get_rl_trend_weights(symbol, strategy)
        if not weights:
            weights = self.default_trend_weights.copy()
            self.save_trend_weights(symbol, strategy, weights)
        
        result = self.default_trend_weights.copy()
        for weight_name in self.default_trend_weights.keys():
            if weight_name in weights:
                result[weight_name] = weights[weight_name]
        
        total = sum(result.values())
        if total > 0:
            for key in result:
                result[key] /= total
        else:
            result = self.default_trend_weights.copy()
        
        return result
    
    def save_entry_weights(self, symbol: str, strategy: str, weights: Dict[str, float]):
        self.db.save_rl_entry_weights(symbol, strategy, weights)
    
    def save_trend_weights(self, symbol: str, strategy: str, weights: Dict[str, float]):
        self.db.save_rl_trend_weights(symbol, strategy, weights)
    
    def _get_experience_buffer_key(self, symbol: str, strategy: str) -> str:
        return f"{symbol}_{strategy}"
    
    def _get_experience_buffer(self, symbol: str, strategy: str) -> deque:
        key = self._get_experience_buffer_key(symbol, strategy)
        if key not in self.experience_buffers:
            self.experience_buffers[key] = deque(maxlen=self.experience_replay_buffer_size)
        return self.experience_buffers[key]
    
    def _update_learning_rate(self, symbol: str, strategy: str):
        key = self._get_experience_buffer_key(symbol, strategy)
        if key not in self.optimization_count:
            self.optimization_count[key] = 0
        
        self.optimization_count[key] += 1
        self.learning_rate = max(
            self.min_learning_rate,
            self.initial_learning_rate * (self.learning_rate_decay ** self.optimization_count[key])
        )
    
    def optimize_weights(self, symbol: str, strategy: str) -> Dict[str, float]:
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT state, action, reward, trade_id
            FROM rl_experiences
            WHERE symbol = ? AND strategy = ?
            ORDER BY timestamp DESC
            LIMIT 100
        """, (symbol, strategy))
        
        experiences = cursor.fetchall()
        if len(experiences) < 10:
            return self.get_weights(symbol, strategy)
        
        method_performance = {
            'node': 0.0,
            'atr': 0.0,
            'garch': 0.0,
            'fixed_rr': 0.0
        }
        method_count = {
            'node': 0,
            'atr': 0,
            'garch': 0,
            'fixed_rr': 0
        }
        
        for exp in experiences:
            try:
                action = json.loads(exp['action'])
                reward = exp['reward']
                method = action.get('method', 'node')
                
                if method in method_performance:
                    method_performance[method] += reward
                    method_count[method] += 1
            except:
                continue
        
        avg_performance = {}
        total_perf = 0.0
        for method in method_performance:
            if method_count[method] > 0:
                avg_perf = method_performance[method] / method_count[method]
                avg_performance[method] = max(0.0, avg_perf)
                total_perf += avg_performance[method]
            else:
                avg_performance[method] = 0.0
        
        if total_perf > 0:
            new_weights = {}
            for method in avg_performance:
                if total_perf > 0:
                    new_weights[method] = avg_performance[method] / total_perf
                else:
                    new_weights[method] = 0.25
            
            old_weights = self.get_weights(symbol, strategy)
            final_weights = {}
            for method in old_weights:
                final_weights[method] = (
                    (1 - self.learning_rate) * old_weights[method] +
                    self.learning_rate * new_weights.get(method, 0.25)
                )
            
            total = sum(final_weights.values())
            if total > 0:
                for method in final_weights:
                    final_weights[method] /= total
        else:
            final_weights = self.get_weights(symbol, strategy)
        
        self.save_weights(symbol, strategy, final_weights)
        logger.info(f"RL weights optimized for {symbol}-{strategy}: {final_weights}")
        
        return final_weights
    
    def optimize_parameters(self, symbol: str, strategy: str) -> Dict[str, float]:
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT state, action, reward, trade_id
            FROM rl_experiences
            WHERE symbol = ? AND strategy = ?
            ORDER BY timestamp DESC
            LIMIT 200
        """, (symbol, strategy))
        
        experiences = cursor.fetchall()
        if len(experiences) < 20:
            return self.get_parameters(symbol, strategy)
        
        parameter_performance = {}
        parameter_count = {}
        
        for param_name in self.default_parameters.keys():
            parameter_performance[param_name] = []
            parameter_count[param_name] = 0
        
        for exp in experiences:
            try:
                action = json.loads(exp['action'])
                reward = exp['reward']
                
                for param_name in self.default_parameters.keys():
                    if param_name in action:
                        param_value = action[param_name]
                        parameter_performance[param_name].append((param_value, reward))
                        parameter_count[param_name] += 1
            except:
                continue
        
        current_params = self.get_parameters(symbol, strategy)
        optimized_params = {}
        
        for param_name, (min_val, max_val) in self.parameter_constraints.items():
            if parameter_count[param_name] < 5:
                optimized_params[param_name] = current_params[param_name]
                continue
            
            param_data = parameter_performance[param_name]
            if not param_data:
                optimized_params[param_name] = current_params[param_name]
                continue
            
            best_value = current_params[param_name]
            best_reward = 0.0
            
            for param_value, reward in param_data:
                if reward > best_reward:
                    best_reward = reward
                    best_value = param_value
            
            new_value = (
                (1 - self.learning_rate) * current_params[param_name] +
                self.learning_rate * best_value
            )
            
            new_value = max(min_val, min(max_val, new_value))
            optimized_params[param_name] = new_value
        
        self.save_parameters(symbol, strategy, optimized_params)
        logger.info(f"RL parameters optimized for {symbol}-{strategy}")
        
        return optimized_params
    
    def optimize_all(self, symbol: str, strategy: str) -> Dict[str, float]:
        weights = self.optimize_weights(symbol, strategy)
        parameters = self.optimize_parameters(symbol, strategy)
        entry_weights = self.optimize_entry_weights(symbol, strategy)
        trend_weights = self.optimize_trend_weights(symbol, strategy)
        
        all_params = {**weights, **parameters, **entry_weights, **trend_weights}
        return all_params
    
    def optimize_entry_weights(self, symbol: str, strategy: str) -> Dict[str, float]:
        self._update_learning_rate(symbol, strategy)
        
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT state, action, reward, trade_id, timestamp
            FROM rl_experiences
            WHERE symbol = ? AND strategy = ?
            ORDER BY timestamp DESC
            LIMIT 500
        """, (symbol, strategy))
        
        experiences = cursor.fetchall()
        if len(experiences) < 20:
            return self.get_entry_weights(symbol, strategy)
        
        buffer = self._get_experience_buffer(symbol, strategy)
        for exp in experiences:
            try:
                state = json.loads(exp['state'])
                action = json.loads(exp['action'])
                reward = exp['reward']
                buffer.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'trade_id': exp['trade_id']
                })
            except:
                continue
        
        if len(buffer) < 20:
            return self.get_entry_weights(symbol, strategy)
        
        entry_contribution = {
            'entry_0': {'rewards': [], 'weights': []},
            'entry_1': {'rewards': [], 'weights': []},
            'entry_2': {'rewards': [], 'weights': []},
            'entry_3': {'rewards': [], 'weights': []}
        }
        
        batch_size = min(self.experience_replay_batch_size, len(buffer))
        batch = list(buffer)[:batch_size]
        
        for exp in batch:
            try:
                state = exp['state']
                action = exp['action']
                reward = exp['reward']
                
                entry_points = state.get('entry_points', [])
                entry_weights_used = {
                    'entry_0': action.get('entry_0', 0.25),
                    'entry_1': action.get('entry_1', 0.25),
                    'entry_2': action.get('entry_2', 0.25),
                    'entry_3': action.get('entry_3', 0.25)
                }
                
                if len(entry_points) >= 4:
                    for i in range(4):
                        entry_key = f'entry_{i}'
                        if entry_key in entry_contribution:
                            entry_contribution[entry_key]['rewards'].append(reward)
                            entry_contribution[entry_key]['weights'].append(entry_weights_used[entry_key])
            except:
                continue
        
        avg_performance = {}
        weighted_performance = {}
        total_perf = 0.0
        
        for entry_key in entry_contribution:
            rewards = entry_contribution[entry_key]['rewards']
            weights = entry_contribution[entry_key]['weights']
            
            if len(rewards) > 0:
                avg_reward = sum(rewards) / len(rewards)
                
                if len(weights) > 0:
                    avg_weight = sum(weights) / len(weights)
                    weighted_avg = sum(r * w for r, w in zip(rewards, weights)) / sum(weights) if sum(weights) > 0 else avg_reward
                else:
                    weighted_avg = avg_reward
                
                avg_performance[entry_key] = max(0.0, avg_reward)
                weighted_performance[entry_key] = max(0.0, weighted_avg)
                total_perf += weighted_performance[entry_key]
            else:
                avg_performance[entry_key] = 0.0
                weighted_performance[entry_key] = 0.0
        
        if total_perf > 0:
            new_weights = {}
            for entry_key in weighted_performance:
                new_weights[entry_key] = weighted_performance[entry_key] / total_perf
        else:
            new_weights = self.default_entry_weights.copy()
        
        old_weights = self.get_entry_weights(symbol, strategy)
        final_weights = {}
        for entry_key in old_weights:
            final_weights[entry_key] = (
                (1 - self.learning_rate) * old_weights[entry_key] +
                self.learning_rate * new_weights.get(entry_key, 0.25)
            )
        
        total = sum(final_weights.values())
        if total > 0:
            for entry_key in final_weights:
                final_weights[entry_key] /= total
        else:
            final_weights = self.default_entry_weights.copy()
        
        self.save_entry_weights(symbol, strategy, final_weights)
        logger.info(f"RL entry weights optimized for {symbol}-{strategy}: {final_weights}, LR={self.learning_rate:.6f}")
        
        return final_weights
    
    def optimize_trend_weights(self, symbol: str, strategy: str) -> Dict[str, float]:
        self._update_learning_rate(symbol, strategy)
        
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT state, action, reward, trade_id, timestamp
            FROM rl_experiences
            WHERE symbol = ? AND strategy = ?
            ORDER BY timestamp DESC
            LIMIT 500
        """, (symbol, strategy))
        
        experiences = cursor.fetchall()
        if len(experiences) < 20:
            return self.get_trend_weights(symbol, strategy)
        
        buffer = self._get_experience_buffer(symbol, strategy)
        for exp in experiences:
            try:
                state = json.loads(exp['state'])
                action = json.loads(exp['action'])
                reward = exp['reward']
                buffer.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'trade_id': exp['trade_id']
                })
            except:
                continue
        
        if len(buffer) < 20:
            return self.get_trend_weights(symbol, strategy)
        
        trend_contribution = {
            'trend_0': {'rewards': [], 'weights': [], 'trends': []},
            'trend_1': {'rewards': [], 'weights': [], 'trends': []},
            'trend_2': {'rewards': [], 'weights': [], 'trends': []}
        }
        
        batch_size = min(self.experience_replay_batch_size, len(buffer))
        batch = list(buffer)[:batch_size]
        
        for exp in batch:
            try:
                state = exp['state']
                action = exp['action']
                reward = exp['reward']
                
                trends = state.get('trends', [])
                trend_weights_used = {
                    'trend_0': action.get('trend_0', 0.33),
                    'trend_1': action.get('trend_1', 0.33),
                    'trend_2': action.get('trend_2', 0.34)
                }
                trend_confidence = state.get('trend_confidence', 0.5)
                
                if len(trends) >= 3:
                    for i in range(3):
                        trend_key = f'trend_{i}'
                        if trend_key in trend_contribution:
                            trend_contribution[trend_key]['rewards'].append(reward)
                            trend_contribution[trend_key]['weights'].append(trend_weights_used[trend_key])
                            trend_contribution[trend_key]['trends'].append(trends[i] if i < len(trends) else 'SIDEWAYS')
            except:
                continue
        
        avg_performance = {}
        weighted_performance = {}
        trend_consistency = {}
        total_perf = 0.0
        
        for trend_key in trend_contribution:
            rewards = trend_contribution[trend_key]['rewards']
            weights = trend_contribution[trend_key]['weights']
            trends_list = trend_contribution[trend_key]['trends']
            
            if len(rewards) > 0:
                avg_reward = sum(rewards) / len(rewards)
                
                if len(weights) > 0:
                    avg_weight = sum(weights) / len(weights)
                    weighted_avg = sum(r * w for r, w in zip(rewards, weights)) / sum(weights) if sum(weights) > 0 else avg_reward
                else:
                    weighted_avg = avg_reward
                
                consistency_bonus = 0.0
                if len(trends_list) > 1:
                    dominant_trend = max(set(trends_list), key=trends_list.count)
                    consistency = trends_list.count(dominant_trend) / len(trends_list)
                    if consistency >= 0.7:
                        consistency_bonus = avg_reward * 0.1
                
                avg_performance[trend_key] = max(0.0, avg_reward)
                weighted_performance[trend_key] = max(0.0, weighted_avg + consistency_bonus)
                trend_consistency[trend_key] = consistency if len(trends_list) > 1 else 0.5
                total_perf += weighted_performance[trend_key]
            else:
                avg_performance[trend_key] = 0.0
                weighted_performance[trend_key] = 0.0
                trend_consistency[trend_key] = 0.5
        
        if total_perf > 0:
            new_weights = {}
            for trend_key in weighted_performance:
                new_weights[trend_key] = weighted_performance[trend_key] / total_perf
        else:
            new_weights = self.default_trend_weights.copy()
        
        old_weights = self.get_trend_weights(symbol, strategy)
        final_weights = {}
        for trend_key in old_weights:
            final_weights[trend_key] = (
                (1 - self.learning_rate) * old_weights[trend_key] +
                self.learning_rate * new_weights.get(trend_key, 0.33)
            )
        
        total = sum(final_weights.values())
        if total > 0:
            for trend_key in final_weights:
                final_weights[trend_key] /= total
        else:
            final_weights = self.default_trend_weights.copy()
        
        self.save_trend_weights(symbol, strategy, final_weights)
        logger.info(f"RL trend weights optimized for {symbol}-{strategy}: {final_weights}, LR={self.learning_rate:.6f}")
        
        return final_weights
    
    def should_optimize(self, symbol: str, strategy: str) -> bool:
        trade_count = self.db.get_trade_count(symbol, strategy)
        return trade_count >= self.trade_count_threshold and trade_count % self.trade_count_threshold == 0
    
    def save_experience(self, symbol: str, strategy: str, timeframe: str,
                       state: Dict, action: Dict, reward: float, 
                       next_state: Dict = None, trade_id: int = None):
        experience = {
            'symbol': symbol,
            'strategy': strategy,
            'timeframe': timeframe,
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state or {},
            'trade_id': trade_id
        }
        self.db.save_experience(experience)
