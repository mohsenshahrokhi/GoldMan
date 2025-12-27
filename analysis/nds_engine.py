from typing import List, Tuple, Optional

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError:
    np = None
    pd = None
    stats = None

from utils.logger import logger
from config.enums import TimeFrame


class NDSEngine:
    
    def __init__(self, market_data_provider, db_manager):
        self.data_provider = market_data_provider
        self.db = db_manager
        self.alpha = 0.86
    
    def identify_nodes(self, df: pd.DataFrame, lookback: int = 20) -> List[Tuple[int, float]]:
        if np is None or pd is None:
            logger.error("numpy or pandas is not installed")
            return []
            
        nodes = []
        
        if len(df) < lookback * 2:
            return nodes
        
        prices = df['close'].values
        
        first_derivative = np.diff(prices)
        second_derivative = np.diff(first_derivative)
        
        for i in range(1, len(first_derivative) - 1):
            if (first_derivative[i-1] * first_derivative[i+1] < 0 and 
                abs(second_derivative[i]) > 0.0001):
                node_index = i + 1
                node_price = prices[node_index]
                nodes.append((node_index, node_price))
        
        return nodes
    
    def detect_trend(self, df: pd.DataFrame, period: int = 20) -> str:
        if stats is None or np is None:
            return "SIDEWAYS"
            
        if len(df) < period:
            return "SIDEWAYS"
        
        prices = df['close'].tail(period).values
        
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        
        if abs(r_value) < 0.5:
            return "SIDEWAYS"
        elif slope > 0:
            return "UP"
        else:
            return "DOWN"
    
    def calculate_rally_correction(self, df: pd.DataFrame) -> Tuple[float, float]:
        if np is None:
            return 0.0, 0.0
            
        if len(df) < 2:
            return 0.0, 0.0
        
        prices = df['close'].values
        recent_high = np.max(prices[-20:])
        recent_low = np.min(prices[-20:])
        
        rally = recent_high - recent_low
        correction = self.alpha * rally
        net_rally = rally - correction
        
        return rally, correction
    
    def find_entry_point(self, df: pd.DataFrame, trend: str, timeframe: TimeFrame) -> Optional[float]:
        if df is None or df.empty:
            return None
        
        nodes = self.identify_nodes(df)
        if not nodes:
            return float(df['close'].iloc[-1])
        
        current_price = df['close'].iloc[-1]
        
        if trend == "UP":
            valid_nodes = [n[1] for n in nodes if n[1] < current_price]
            if valid_nodes:
                return max(valid_nodes)
        elif trend == "DOWN":
            valid_nodes = [n[1] for n in nodes if n[1] > current_price]
            if valid_nodes:
                return min(valid_nodes)
        
        return current_price
    
    def find_nearest_node(self, df: pd.DataFrame, price: float, direction: str = "above") -> Optional[float]:
        nodes = self.identify_nodes(df)
        if not nodes:
            return None
        
        node_prices = [n[1] for n in nodes]
        
        if direction == "above":
            valid_nodes = [p for p in node_prices if p > price]
            return min(valid_nodes) if valid_nodes else None
        else:
            valid_nodes = [p for p in node_prices if p < price]
            return max(valid_nodes) if valid_nodes else None
    
    def detect_trend_weakness(self, df: pd.DataFrame, timeframe: TimeFrame) -> bool:
        if np is None:
            return False
            
        if len(df) < 10:
            return False
        
        prices = df['close'].tail(10).values
        momentum = np.diff(prices)
        
        if len(momentum) >= 3:
            recent_momentum = momentum[-3:]
            if np.all(recent_momentum < 0) and momentum[-1] < momentum[-2]:
                return True
        
        return False
