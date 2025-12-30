from typing import List, Tuple, Optional, Dict, Union

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


class MarketEngine:
    
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

        smoothed_prices = df['close'].rolling(window=3, center=True).mean().fillna(df['close'])
        prices = smoothed_prices.values

        first_derivative = np.diff(prices)
        second_derivative = np.diff(first_derivative)

        for i in range(1, len(first_derivative) - 1):
            if (first_derivative[i-1] * first_derivative[i+1] < 0 and
                abs(second_derivative[i]) > 0.0001):
                node_index = i + 1
                node_price = prices[node_index]
                nodes.append((node_index, node_price))

        return nodes

    def analyze_volume_confirmation(self, df: pd.DataFrame, node_index: int,
                                  lookback: int = 20) -> Tuple[bool, float]:
        """
        تحلیل حجم در گره‌های قیمتی

        پارامترها:
        - df: DataFrame قیمت و حجم
        - node_index: ایندکس گره
        - lookback: دوره زمانی برای محاسبه میانگین حجم

        خروجی:
        - (تأیید_حجم, امتیاز_حجم): امتیاز بین 0 تا 1
        """
        if 'volume' not in df.columns or node_index >= len(df):
            return False, 0.0

        volumes = df['volume'].values

        start_idx = max(0, node_index - lookback)
        end_idx = min(len(volumes), node_index + lookback + 1)

        avg_volume = np.mean(volumes[start_idx:end_idx])

        node_volume = volumes[node_index]

        volume_score = min(node_volume / (avg_volume + 0.001), 2.0) / 2.0

        volume_confirmed = node_volume >= avg_volume * 0.7

        return volume_confirmed, volume_score

    def identify_nodes_improved(self, df: pd.DataFrame, lookback: int = 20,
                               min_amplitude: float = 0.001,
                               volume_confirmation: bool = True,
                               volume_weight: float = 0.3) -> List[Tuple[int, float, float]]:
        """
        نسخه پیشرفته تشخیص گره‌ها با تحلیل حجم و امتیازدهی

        پارامترها:
        - min_amplitude: حداقل اختلاف قیمت برای معتبر بودن گره (به عنوان درصد)
        - volume_confirmation: استفاده از حجم برای تأیید گره
        - volume_weight: وزن حجم در امتیاز نهایی گره (0-1)

        خروجی:
        - لیست تاپل‌ها: (index, price, confidence_score)
        """
        if np is None or pd is None:
            logger.error("numpy or pandas is not installed")
            return []

        nodes = []

        if len(df) < lookback * 2:
            return nodes

        prices = df['close'].values
        price_range = np.max(prices[-lookback:]) - np.min(prices[-lookback:])

        smoothed_prices = df['close'].ewm(span=5, adjust=False).mean().values

        first_derivative = np.diff(smoothed_prices)
        second_derivative = np.diff(first_derivative)

        for i in range(2, len(first_derivative) - 2):
            direction_change = first_derivative[i-1] * first_derivative[i+1] < 0
            significant_curve = abs(second_derivative[i]) > 0.0001

            if direction_change and significant_curve:
                node_index = i + 1
                node_price = smoothed_prices[node_index]

                amplitude_threshold = price_range * min_amplitude
                left_min = np.min(smoothed_prices[max(0, node_index-5):node_index])
                right_max = np.max(smoothed_prices[node_index:min(len(smoothed_prices), node_index+5)])
                left_max = np.max(smoothed_prices[max(0, node_index-5):node_index])
                right_min = np.min(smoothed_prices[node_index:min(len(smoothed_prices), node_index+5)])

                amplitude = max(abs(node_price - left_min), abs(node_price - right_max),
                              abs(node_price - left_max), abs(node_price - right_min))

                if amplitude >= amplitude_threshold:
                    price_score = min(amplitude / (price_range + 0.001), 1.0)

                    volume_confirmed, volume_score = self.analyze_volume_confirmation(df, node_index, lookback)

                    confidence_score = (1 - volume_weight) * price_score + volume_weight * volume_score

                    if not volume_confirmation or volume_confirmed:
                        nodes.append((node_index, node_price, confidence_score))

        nodes.sort(key=lambda x: x[2], reverse=True)

        return nodes
    
    def detect_trend(self, df: pd.DataFrame, period: int = 20) -> str:
        if stats is None or np is None:
            return "SIDEWAYS"

        if len(df) < period:
            return "SIDEWAYS"

        prices = df['close'].tail(period).values

        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)

        momentum_score = self.analyze_momentum(df, period)

        combined_score = r_value * 0.7 + momentum_score * 0.3

        if abs(combined_score) < 0.4:
            return "SIDEWAYS"
        elif combined_score > 0:
            return "UP"
        else:
            return "DOWN"
    
    def detect_trend_with_strength(self, df: pd.DataFrame, period: int = 20) -> Tuple[str, float]:
        if stats is None or np is None:
            return "SIDEWAYS", 0.0

        if len(df) < period:
            return "SIDEWAYS", 0.0

        prices = df['close'].tail(period).values

        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)

        momentum_score = self.analyze_momentum(df, period)

        base_strength = abs(r_value) * 100
        momentum_boost = abs(momentum_score) * 20  # تقویت حرکت
        trend_strength = min(base_strength + momentum_boost, 100.0)

        if abs(r_value) < 0.5:
            return "SIDEWAYS", trend_strength
        elif slope > 0:
            return "UP", trend_strength
        else:
            return "DOWN", trend_strength
    
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

        nodes = self.identify_nodes_improved(df, lookback=20, min_amplitude=0.002)
        if not nodes:
            return float(df['close'].iloc[-1])

        current_price = df['close'].iloc[-1]

        if trend == "UP":
            valid_nodes = [(n[1], n[2]) for n in nodes if n[1] < current_price]
            if valid_nodes:
                best_node = max(valid_nodes, key=lambda x: x[1])
                return best_node[0]
        elif trend == "DOWN":
            valid_nodes = [(n[1], n[2]) for n in nodes if n[1] > current_price]
            if valid_nodes:
                best_node = max(valid_nodes, key=lambda x: x[1])
                return best_node[0]

        return current_price

    def get_volume_analysis(self, df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
        """
        تحلیل کلی حجم بازار

        خروجی:
        - volume_trend: روند حجم (-1 تا 1)
        - volume_volatility: نوسان حجم
        - abnormal_volume: حجم غیرعادی اخیر
        """
        if 'volume' not in df.columns or len(df) < lookback:
            return {'volume_trend': 0.0, 'volume_volatility': 0.0, 'abnormal_volume': False}

        volumes = df['volume'].tail(lookback).values

        x = np.arange(len(volumes))
        slope, _, r_value, _, _ = stats.linregress(x, volumes) if stats else (0, 0, 0, 0, 0)
        volume_trend = r_value if not np.isnan(r_value) else 0.0

        volume_volatility = np.std(volumes) / (np.mean(volumes) + 0.001)

        recent_volume = volumes[-1] if len(volumes) > 0 else 0
        avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else recent_volume
        abnormal_volume = recent_volume > avg_volume * 1.5

        return {
            'volume_trend': volume_trend,
            'volume_volatility': volume_volatility,
            'abnormal_volume': abnormal_volume
        }

    def analyze_cycle_patterns(self, df: pd.DataFrame, max_cycle_level: int = 4) -> Dict[int, List[Tuple[int, int, float]]]:
        """
        Cycle analysis based on mathematical formulas

        فرمول چرخه‌های تو در تو:
        Cₙ,ₙ₊ₖ = Σ(i=n to n+k) Cᵢ
        Cₖ(t) = Σ(j=1 to Nₖ) (1/Sₖ) Cₖ₊₁(t/Sₖ)

        خروجی: دیکشنری از چرخه‌ها در سطوح مختلف
        هر چرخه: (شروع، پایان، قدرت_چرخه)
        """
        if len(df) < 50:
            return {}

        prices = df['close'].values
        cycles = {}

        base_cycles = self.identify_nested_cycles_base(prices)

        for level in range(1, max_cycle_level + 1):
            scale_factor = 2 ** (level - 1)  # Sₖ = 2^(k-1)
            level_cycles = []

            if level == 1:
                level_cycles = base_cycles
            else:
                prev_level_cycles = cycles[level - 1]
                combined_cycles = self.combine_cycles_at_level(prev_level_cycles, scale_factor)
                level_cycles = combined_cycles

            cycles[level] = level_cycles

        return cycles

    def identify_nested_cycles_base(self, prices: np.ndarray, min_cycle_length: int = 5) -> List[Tuple[int, int, float]]:
        """
        شناسایی چرخه‌های پایه (کوچکترین چرخه‌ها)
        """
        cycles = []
        i = 0

        while i < len(prices) - min_cycle_length * 2:
            peaks = []
            troughs = []

            window_size = min_cycle_length * 4
            window_end = min(i + window_size, len(prices))

            window_prices = prices[i:window_end]

            for j in range(1, len(window_prices) - 1):
                if (window_prices[j] > window_prices[j-1] and
                    window_prices[j] > window_prices[j+1]):
                    peaks.append(i + j)
                elif (window_prices[j] < window_prices[j-1] and
                      window_prices[j] < window_prices[j+1]):
                    troughs.append(i + j)

            if len(peaks) >= 2 and len(troughs) >= 1:
                cycle_start = peaks[0]
                cycle_end = peaks[-1]

                if cycle_end - cycle_start >= min_cycle_length:
                    cycle_prices = prices[cycle_start:cycle_end+1]
                    amplitude = np.max(cycle_prices) - np.min(cycle_prices)
                    avg_price = np.mean(cycle_prices)
                    cycle_strength = amplitude / avg_price if avg_price > 0 else 0

                    cycles.append((cycle_start, cycle_end, cycle_strength))

            i += min_cycle_length

        return cycles

    def combine_cycles_at_level(self, prev_cycles: List[Tuple[int, int, float]],
                               scale_factor: int) -> List[Tuple[int, int, float]]:
        """
        ترکیب چرخه‌های سطح پایین‌تر برای تشکیل چرخه‌های سطح بالاتر
        طبق فرمول: Cₖ(t) = Σ(j=1 to Nₖ) (1/Sₖ) Cₖ₊₁(t/Sₖ)
        """
        if len(prev_cycles) < 2:
            return prev_cycles

        combined_cycles = []

        i = 0
        while i < len(prev_cycles) - 1:
            cycle1 = prev_cycles[i]
            cycle2 = prev_cycles[i+1]

            if cycle2[0] - cycle1[1] <= scale_factor:
                combined_start = cycle1[0]
                combined_end = cycle2[1]
                combined_strength = (cycle1[2] + cycle2[2]) / scale_factor

                combined_cycles.append((combined_start, combined_end, combined_strength))
                i += 2  # دو چرخه رو ترکیب کردیم
            else:
                combined_cycles.append(cycle1)
                i += 1

        if i < len(prev_cycles):
            combined_cycles.append(prev_cycles[i])

        return combined_cycles

    def get_dominant_cycle(self, df: pd.DataFrame) -> Tuple[int, float]:
        """
        یافتن چرخه غالب بازار

        خروجی: (سطح_چرخه، قدرت_چرخه)
        """
        cycles = self.analyze_cycle_patterns(df)

        if not cycles:
            return 0, 0.0

        max_strength = 0.0
        dominant_level = 0

        for level, level_cycles in cycles.items():
            if level_cycles:
                level_max_strength = max(cycle[2] for cycle in level_cycles)
                if level_max_strength > max_strength:
                    max_strength = level_max_strength
                    dominant_level = level

        return dominant_level, max_strength

    def analyze_fractal_dimension(self, df: pd.DataFrame, q_orders: List[float] = None) -> Dict[str, float]:
        """
        Fractal dimension analysis based on mathematical formulas

        فرمول τ(q):
        τ(q) = [ln(μ(q))] / [ln(ε)]

        خروجی:
        - fractal_dimension: بعد فراکتالی
        - hurst_exponent: نمایی هرست
        - multifractal_spectrum: طیف چندفراکتالی
        """
        if q_orders is None:
            q_orders = [-5, -3, -1, 0, 1, 3, 5]

        if len(df) < 100:
            return {'fractal_dimension': 1.5, 'hurst_exponent': 0.5, 'multifractal_spectrum': 0.0}

        prices = df['close'].values
        returns = np.diff(np.log(prices))  # بازدهی لگاریتمی

        tau_q = []
        epsilon_values = []

        for q in q_orders:
            scales = [2**i for i in range(2, 8)]  # ε = 4, 8, 16, 32, 64, 128
            mu_q = []

            for scale in scales:
                if len(returns) < scale:
                    continue

                partitions = []
                for i in range(0, len(returns) - scale + 1, scale // 2):
                    segment = returns[i:i + scale]
                    if len(segment) == scale:
                        partitions.append(segment)

                if partitions:
                    local_measures = []
                    for partition in partitions:
                        if q == 0:
                            measure = len([x for x in partition if abs(x) > 0]) / scale
                        else:
                            measure = np.mean([abs(x)**q for x in partition])
                        local_measures.append(measure)

                    mu_q_scale = np.mean(local_measures)
                    mu_q.append(mu_q_scale)

            if mu_q:
                log_mu = [np.log(m) for m in mu_q if m > 0]
                log_epsilon = [np.log(s) for s in scales[:len(log_mu)]]

                if len(log_mu) > 1:
                    slope, _, _, _, _ = stats.linregress(log_epsilon, log_mu) if stats else (0, 0, 0, 0, 0)
                    tau_q.append(slope)
                    epsilon_values.append(np.mean(log_epsilon))

        fractal_dimension = 1.5  # مقدار پیش‌فرض
        hurst_exponent = 0.5     # مقدار پیش‌فرض

        if len(tau_q) > 1:
            if 0 in q_orders:
                q0_index = q_orders.index(0)
                if q0_index < len(tau_q):
                    fractal_dimension = tau_q[q0_index]

            if 2 in q_orders:
                q2_index = q_orders.index(2)
                if q2_index < len(tau_q):
                    hurst_exponent = 1 + tau_q[q2_index] / 2

        multifractal_spectrum = 0.0
        if tau_q:
            multifractal_spectrum = max(tau_q) - min(tau_q)

        return {
            'fractal_dimension': fractal_dimension,
            'hurst_exponent': hurst_exponent,
            'multifractal_spectrum': multifractal_spectrum
        }

    def geometric_sequence_analysis(self, df: pd.DataFrame, max_sequence_length: int = 10) -> List[Tuple[str, float, int]]:
        """
        Geometric sequence analysis based on mathematical formulas

        فرمول دنباله هندسی:
        f(Z) = N₁ → S₁ → N₂ → S₂ → N₃ → S₃

        خروجی: لیست توالی‌ها به صورت (نوع, ارزش, موقعیت)
        N: Node (گره), S: Slope (شیب)
        """
        if len(df) < 20:
            return []

        nodes = self.identify_nodes_improved(df, min_amplitude=0.001, volume_confirmation=False)
        if len(nodes) < 3:
            return []

        sequence = []
        prices = df['close'].values

        current_pos = nodes[0][0]  # موقعیت اولین گره
        sequence.append(('N', nodes[0][1], current_pos))  # N1

        for i in range(1, min(len(nodes), max_sequence_length)):
            next_node_pos = nodes[i][0]
            next_node_price = nodes[i][1]

            if next_node_pos > current_pos:
                segment_prices = prices[current_pos:next_node_pos+1]
                if len(segment_prices) > 1:
                    x = np.arange(len(segment_prices))
                    slope, _, _, _, _ = stats.linregress(x, segment_prices) if stats else (0, 0, 0, 0, 0)

                    slope_normalized = slope / (np.mean(segment_prices) + 0.001)
                    sequence.append(('S', slope_normalized, current_pos))  # شیب

            sequence.append(('N', next_node_price, next_node_pos))
            current_pos = next_node_pos

        return sequence

    def analyze_geometric_patterns(self, df: pd.DataFrame) -> Dict[str, Union[int, float, List]]:
        """
        تحلیل الگوهای هندسی بازار

        خروجی:
        - sequence_length: طول توالی
        - dominant_pattern: الگوی غالب (UPTREND, DOWNTREND, SIDEWAYS)
        - pattern_strength: قدرت الگو
        - fibonacci_levels: سطوح فیبوناچی شناسایی شده
        """
        sequence = self.geometric_sequence_analysis(df)

        if len(sequence) < 3:
            return {
                'sequence_length': len(sequence),
                'dominant_pattern': 'UNKNOWN',
                'pattern_strength': 0.0,
                'fibonacci_levels': []
            }

        node_prices = [item[1] for item in sequence if item[0] == 'N']
        slope_values = [item[1] for item in sequence if item[0] == 'S']

        if len(node_prices) >= 3:
            node_trend = np.polyfit(range(len(node_prices)), node_prices, 1)[0]

            avg_slope = np.mean(slope_values) if slope_values else 0

            if node_trend > 0.001 and avg_slope > 0.001:
                dominant_pattern = 'UPTREND'
            elif node_trend < -0.001 and avg_slope < -0.001:
                dominant_pattern = 'DOWNTREND'
            else:
                dominant_pattern = 'SIDEWAYS'
        else:
            dominant_pattern = 'UNKNOWN'

        slope_magnitude = np.mean([abs(s) for s in slope_values]) if slope_values else 0
        trend_consistency = 1.0  # coherence of the sequence

        if len(slope_values) > 1:
            slope_signs = [1 if s > 0 else -1 for s in slope_values]
            consistency = sum(1 for i in range(1, len(slope_signs)) if slope_signs[i] == slope_signs[i-1])
            trend_consistency = consistency / (len(slope_signs) - 1) if len(slope_signs) > 1 else 1.0

        pattern_strength = (slope_magnitude * 0.6) + (trend_consistency * 0.4)

        fibonacci_levels = self.identify_fibonacci_levels(node_prices)

        return {
            'sequence_length': len(sequence),
            'dominant_pattern': dominant_pattern,
            'pattern_strength': pattern_strength,
            'fibonacci_levels': fibonacci_levels
        }

    def identify_fibonacci_levels(self, prices: List[float]) -> List[Tuple[float, str]]:
        """
        شناسایی سطوح فیبوناچی در توالی قیمت‌ها

        خروجی: لیست (سطح_فیبوناچی, نوع_سطح)
        """
        if len(prices) < 2:
            return []

        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        fib_ratios = []

        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price

        if price_range == 0:
            return []

        for level in fib_levels:
            fib_price = max_price - (price_range * level)
            fib_ratios.append((fib_price, f'FIB_CORRECTION_{level:.3f}'))

        for level in fib_levels:
            fib_price = min_price + (price_range * (1 + level))
            fib_ratios.append((fib_price, f'FIB_EXTENSION_{level:.3f}'))

        return fib_ratios

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """
        محاسبه اندیکاتور RSI (Relative Strength Index)

        پارامترها:
        - period: دوره محاسبه (معمولاً ۱۴)

        خروجی: آرایه مقادیر RSI
        """
        if len(df) < period + 1:
            return np.array([])

        prices = df['close'].values
        deltas = np.diff(prices)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gains = np.zeros_like(gains)
        avg_losses = np.zeros_like(losses)

        avg_gains[period-1] = np.mean(gains[:period])
        avg_losses[period-1] = np.mean(losses[:period])

        for i in range(period, len(gains)):
            avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i]) / period
            avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i]) / period

        rs = avg_gains / (avg_losses + 1e-10)  # جلوگیری از تقسیم بر صفر
        rsi = 100 - (100 / (1 + rs))

        rsi_full = np.full(len(prices), np.nan)
        rsi_full[period:] = rsi[period-1:]

        return rsi_full

    def calculate_macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26,
                       signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        محاسبه اندیکاتور MACD (Moving Average Convergence Divergence)

        پارامترها:
        - fast_period: دوره EMA سریع (معمولاً ۱۲)
        - slow_period: دوره EMA کند (معمولاً ۲۶)
        - signal_period: دوره سیگنال (معمولاً ۹)

        خروجی: (MACD line, Signal line, Histogram)
        """
        if len(df) < slow_period + signal_period:
            return np.array([]), np.array([]), np.array([])

        prices = df['close'].values

        fast_ema = self.calculate_ema(prices, fast_period)
        slow_ema = self.calculate_ema(prices, slow_period)

        macd_line = fast_ema - slow_ema

        signal_line = self.calculate_ema(macd_line, signal_period)

        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        محاسبه میانگین متحرک نمایی (EMA)
        """
        if len(data) < period:
            return np.full(len(data), np.nan)

        ema = np.full(len(data), np.nan)
        ema[period-1] = np.mean(data[:period])

        multiplier = 2 / (period + 1)
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]

        return ema

    def analyze_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Union[float, str]]:
        """
        تحلیل جامع اندیکاتورهای تکنیکال برای تأیید سیگنال‌ها

        خروجی:
        - rsi_signal: سیگنال RSI (OVERBOUGHT, OVERSOLD, NEUTRAL)
        - macd_signal: سیگنال MACD (BULLISH, BEARISH, NEUTRAL)
        - combined_signal: سیگنال ترکیبی
        - rsi_value: مقدار فعلی RSI
        - macd_histogram: مقدار هیستوگرام MACD
        """
        if len(df) < 30:  # حداقل داده برای محاسبات
            return {
                'rsi_signal': 'NEUTRAL',
                'macd_signal': 'NEUTRAL',
                'combined_signal': 'NEUTRAL',
                'rsi_value': 50.0,
                'macd_histogram': 0.0
            }

        rsi_values = self.calculate_rsi(df, period=14)
        current_rsi = rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50.0

        if current_rsi >= 70:
            rsi_signal = 'OVERBOUGHT'  # اشباع خرید - سیگنال نزولی
        elif current_rsi <= 30:
            rsi_signal = 'OVERSOLD'    # اشباع فروش - سیگنال صعودی
        else:
            rsi_signal = 'NEUTRAL'

        macd_line, signal_line, histogram = self.calculate_macd(df)
        current_histogram = histogram[-1] if len(histogram) > 0 and not np.isnan(histogram[-1]) else 0.0

        prev_histogram = histogram[-2] if len(histogram) > 1 and not np.isnan(histogram[-2]) else 0.0

        if current_histogram > 0 and current_histogram > prev_histogram:
            macd_signal = 'BULLISH'   # سیگنال صعودی
        elif current_histogram < 0 and current_histogram < prev_histogram:
            macd_signal = 'BEARISH'   # سیگنال نزولی
        else:
            macd_signal = 'NEUTRAL'

        combined_signal = self.combine_technical_signals(rsi_signal, macd_signal)

        return {
            'rsi_signal': rsi_signal,
            'macd_signal': macd_signal,
            'combined_signal': combined_signal,
            'rsi_value': current_rsi,
            'macd_histogram': current_histogram
        }

    def combine_technical_signals(self, rsi_signal: str, macd_signal: str) -> str:
        """
        ترکیب سیگنال‌های RSI و MACD برای تصمیم‌گیری نهایی

        قوانین ترکیب:
        - اگر هر دو صعودی باشند: STRONG_BULLISH
        - اگر هر دو نزولی باشند: STRONG_BEARISH
        - اگر یکی صعودی و دیگری نزولی: NEUTRAL
        - سایر حالات: WEAK سیگنال
        """
        bullish_signals = ['OVERSOLD', 'BULLISH']
        bearish_signals = ['OVERBOUGHT', 'BEARISH']

        rsi_bullish = rsi_signal in bullish_signals
        rsi_bearish = rsi_signal in bearish_signals
        macd_bullish = macd_signal == 'BULLISH'
        macd_bearish = macd_signal == 'BEARISH'

        if rsi_bullish and macd_bullish:
            return 'STRONG_BULLISH'
        elif rsi_bearish and macd_bearish:
            return 'STRONG_BEARISH'
        elif rsi_bullish and macd_bearish:
            return 'NEUTRAL'  # تضاد سیگنال
        elif rsi_bearish and macd_bullish:
            return 'NEUTRAL'  # تضاد سیگنال
        elif rsi_bullish or macd_bullish:
            return 'WEAK_BULLISH'
        elif rsi_bearish or macd_bearish:
            return 'WEAK_BEARISH'
        else:
            return 'NEUTRAL'

    def quantum_price_modeling(self, df: pd.DataFrame) -> Dict[str, Union[float, List[complex]]]:
        """
        Quantum price modeling based on mathematical formulas

        فرمول نمایش قیمت در فضای هیلبرت:
        ψ(t) = P(t) e^(iφ(t))

        سیستم جمع صفر بازار:
        Σ(شرکت‌کنندگان i) Δψᵢ(tᵦ - tₐ) = 0

        خروجی:
        - quantum_wave_function: تابع موج کوانتومی قیمت
        - market_coherence: coherence بازار
        - zero_sum_violation: اندازه نقض سیستم جمع صفر
        - phase_space_analysis: تحلیل فضای فاز
        """
        if len(df) < 10:
            return {
                'quantum_wave_function': [],
                'market_coherence': 0.0,
                'zero_sum_violation': 0.0,
                'phase_space_analysis': {}
            }

        prices = df['close'].values
        returns = np.diff(np.log(prices))  # بازدهی لگاریتمی

        quantum_wave_function = []

        for i, price in enumerate(prices):
            if i > 0:
                price_magnitude = price
                price_phase = np.angle(returns[i-1] + 1j * 0.01)  # فاز از بازدهی مختلط
                wave_function = price_magnitude * np.exp(1j * price_phase)
                quantum_wave_function.append(wave_function)

        if len(quantum_wave_function) > 1:
            coherence_sum = 0
            count = 0

            for i in range(len(quantum_wave_function) - 1):
                psi1 = quantum_wave_function[i]
                psi2 = quantum_wave_function[i + 1]

                inner_product = psi1 * np.conj(psi2)
                magnitude = abs(inner_product)
                coherence_sum += magnitude
                count += 1

            market_coherence = coherence_sum / count if count > 0 else 0.0
        else:
            market_coherence = 0.0

        zero_sum_violation = 0.0

        if len(quantum_wave_function) > 2:
            wave_changes = np.diff(quantum_wave_function)

            total_change = np.sum(wave_changes)
            zero_sum_violation = abs(total_change)

            market_size = np.mean([abs(psi) for psi in quantum_wave_function])
            zero_sum_violation = zero_sum_violation / (market_size + 0.001)

        phase_space_analysis = {}

        if quantum_wave_function:
            phases = [np.angle(psi) for psi in quantum_wave_function]
            magnitudes = [abs(psi) for psi in quantum_wave_function]

            phase_space_analysis = {
                'phase_distribution': phases,
                'magnitude_distribution': magnitudes,
                'phase_volatility': np.std(phases) if len(phases) > 1 else 0,
                'magnitude_volatility': np.std(magnitudes) if len(magnitudes) > 1 else 0,
                'quantum_entropy': self.calculate_quantum_entropy(quantum_wave_function)
            }

        return {
            'quantum_wave_function': quantum_wave_function,
            'market_coherence': market_coherence,
            'zero_sum_violation': zero_sum_violation,
            'phase_space_analysis': phase_space_analysis
        }

    def calculate_quantum_entropy(self, wave_functions: List[complex]) -> float:
        """
        محاسبه آنتروپی کوانتومی بازار
        """
        if not wave_functions:
            return 0.0

        magnitudes = np.array([abs(psi) for psi in wave_functions])
        total_magnitude = np.sum(magnitudes)

        if total_magnitude == 0:
            return 0.0

        probabilities = magnitudes / total_magnitude

        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy

    def detect_quantum_market_regime(self, df: pd.DataFrame) -> str:
        """
        تشخیص رژیم بازار با استفاده از مدل کوانتومی

        خروجی:
        - 'COHERENT': بازار coherent (منظم)
        - 'DECOHERENT': بازار decoherent (نامنظم)
        - 'CHAOTIC': بازار chaotic (هرج و مرج)
        """
        quantum_analysis = self.quantum_price_modeling(df)

        coherence = quantum_analysis['market_coherence']
        zero_sum_violation = quantum_analysis['zero_sum_violation']
        phase_volatility = quantum_analysis['phase_space_analysis'].get('phase_volatility', 0)

        if coherence > 0.7 and zero_sum_violation < 0.1:
            return 'COHERENT'  # بازار منظم و کارآمد
        elif phase_volatility > 1.0 or zero_sum_violation > 0.3:
            return 'CHAOTIC'   # بازار هرج و مرج
        else:
            return 'DECOHERENT'  # بازار نامنظم

    def analyze_momentum(self, df: pd.DataFrame, period: int = 20) -> float:
        """
        تحلیل حرکت قیمت برای بهبود تشخیص روند

        خروجی: امتیاز حرکت بین -1 تا 1
        """
        if len(df) < period + 5:
            return 0.0

        prices = df['close'].values[-period:]

        roc = (prices[-1] - prices[0]) / prices[0]

        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = np.mean(gains[-14:]) if gains else 0
        avg_loss = np.mean(losses[-14:]) if losses else 0

        if avg_loss == 0:
            rsi_score = 1.0
        else:
            rs = avg_gain / avg_loss
            rsi_score = rs / (1 + rs)  # نرمال‌سازی بین 0-1
            rsi_score = rsi_score * 2 - 1  # تبدیل به -1 تا 1

        short_ema = self.calculate_ema_simple(prices, 12)
        long_ema = self.calculate_ema_simple(prices, 26)

        if len(short_ema) > 0 and len(long_ema) > 0:
            macd_diff = short_ema[-1] - long_ema[-1]
            macd_score = np.tanh(macd_diff / (prices[-1] * 0.01))  # نرمال‌سازی
        else:
            macd_score = 0.0

        momentum_score = roc * 0.5 + rsi_score * 0.3 + macd_score * 0.2

        return max(-1.0, min(1.0, momentum_score))

    def calculate_ema_simple(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        محاسبه ساده EMA (برای استفاده داخلی)
        """
        if len(data) < period:
            return np.array([])

        ema = np.zeros(len(data))
        ema[period-1] = np.mean(data[:period])

        multiplier = 2 / (period + 1)
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]

        return ema

    def detect_trend_advanced(self, df: pd.DataFrame, period: int = 20) -> Tuple[str, float, Dict[str, float]]:
        """
        تشخیص پیشرفته روند با استفاده از چند فاکتور

        خروجی:
        - روند: UP, DOWN, SIDEWAYS
        - اطمینان: 0-100
        - جزئیات: دیکشنری از فاکتورهای مختلف
        """
        if len(df) < period:
            return "SIDEWAYS", 0.0, {}

        factors = {}

        prices = df['close'].tail(period).values
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        factors['regression_strength'] = abs(r_value)
        factors['regression_slope'] = slope

        momentum_score = self.analyze_momentum(df, period)
        factors['momentum_score'] = momentum_score

        if 'volume' in df.columns:
            volume_analysis = self.get_volume_analysis(df, period)
            factors['volume_trend'] = volume_analysis['volume_trend']
            factors['volume_volatility'] = volume_analysis['volume_volatility']
        else:
            factors['volume_trend'] = 0.0
            factors['volume_volatility'] = 0.0

        technical_analysis = self.analyze_technical_indicators(df)
        factors['rsi_value'] = technical_analysis['rsi_value'] / 100.0  # نرمال‌سازی
        factors['macd_histogram'] = technical_analysis['macd_histogram']

        regression_weight = 0.4
        momentum_weight = 0.3
        volume_weight = 0.2
        technical_weight = 0.1

        regression_score = factors['regression_strength'] * (1 if slope > 0 else -1)
        momentum_score_norm = factors['momentum_score']
        volume_score = factors['volume_trend'] * 0.5 + (1 - factors['volume_volatility']) * 0.5
        technical_score = (factors['rsi_value'] - 0.5) * 2 + factors['macd_histogram'] * 10

        combined_score = (
            regression_score * regression_weight +
            momentum_score_norm * momentum_weight +
            volume_score * volume_weight +
            technical_score * technical_weight
        )

        confidence = min(abs(combined_score) * 100, 100.0)

        if abs(combined_score) < 0.3:
            trend = "SIDEWAYS"
        elif combined_score > 0:
            trend = "UP"
        else:
            trend = "DOWN"

        factors['combined_score'] = combined_score
        factors['confidence'] = confidence

        return trend, confidence, factors

    def analyze_multi_timeframe_consistency(self, symbol: str, timeframes: List[str] = None) -> Dict[str, Union[str, float, Dict]]:
        """
        تحلیل سازگاری روندهای چند تایم‌فریم

        پارامترها:
        - symbol: نماد معاملاتی
        - timeframes: لیست تایم‌فریم‌ها برای تحلیل

        خروجی:
        - overall_trend: روند کلی
        - consistency_score: امتیاز سازگاری (0-100)
        - timeframe_analysis: تحلیل هر تایم‌فریم
        - conflicts: تعارضات شناسایی شده
        """
        if timeframes is None:
            timeframes = ['M5', 'M15', 'H1', 'H4']

        if not self.data_provider:
            return {
                'overall_trend': 'UNKNOWN',
                'consistency_score': 0.0,
                'timeframe_analysis': {},
                'conflicts': []
            }

        timeframe_analysis = {}
        trends = []

        for tf in timeframes:
            try:
                df = self.data_provider.get_ohlc_data(symbol, tf, 100)
                if df is not None and len(df) >= 20:
                    trend, confidence, factors = self.detect_trend_advanced(df, period=20)
                    timeframe_analysis[tf] = {
                        'trend': trend,
                        'confidence': confidence,
                        'factors': factors
                    }
                    trends.append(trend)
                else:
                    timeframe_analysis[tf] = {
                        'trend': 'UNKNOWN',
                        'confidence': 0.0,
                        'factors': {}
                    }
            except Exception as e:
                logger.warning(f"Error analyzing timeframe {tf}: {e}")
                timeframe_analysis[tf] = {
                    'trend': 'UNKNOWN',
                    'confidence': 0.0,
                    'factors': {}
                }

        if trends:
            trend_counts = {}
            for trend in trends:
                if trend != 'UNKNOWN':
                    trend_counts[trend] = trend_counts.get(trend, 0) + 1

            if trend_counts:
                overall_trend = max(trend_counts, key=trend_counts.get)

                total_frames = len([t for t in trends if t != 'UNKNOWN'])
                if total_frames > 0:
                    consistency_score = (trend_counts.get(overall_trend, 0) / total_frames) * 100
                else:
                    consistency_score = 0.0
            else:
                overall_trend = 'UNKNOWN'
                consistency_score = 0.0
        else:
            overall_trend = 'UNKNOWN'
            consistency_score = 0.0

        conflicts = []
        if len(timeframes) >= 2:
            for i in range(len(timeframes) - 1):
                tf1 = timeframes[i]
                tf2 = timeframes[i + 1]

                trend1 = timeframe_analysis[tf1]['trend']
                trend2 = timeframe_analysis[tf2]['trend']

                if (trend1 != 'UNKNOWN' and trend2 != 'UNKNOWN' and
                    trend1 != trend2 and trend1 != 'SIDEWAYS' and trend2 != 'SIDEWAYS'):

                    if i == 0:  # M5 vs M15 - M5 نباید خلاف M15 باشه
                        conflicts.append({
                            'type': 'short_vs_long',
                            'timeframes': [tf1, tf2],
                            'trends': [trend1, trend2],
                            'severity': 'HIGH'
                        })

        return {
            'overall_trend': overall_trend,
            'consistency_score': consistency_score,
            'timeframe_analysis': timeframe_analysis,
            'conflicts': conflicts
        }

    def get_multi_timeframe_signal(self, symbol: str) -> Dict[str, Union[str, float, bool]]:
        """
        تولید سیگنال معاملاتی بر اساس تحلیل چند تایم‌فریم

        خروجی:
        - signal: سیگنال نهایی (BUY, SELL, HOLD)
        - strength: قدرت سیگنال (0-100)
        - is_confirmed: آیا سیگنال تأیید شده است
        - risk_level: سطح ریسک (LOW, MEDIUM, HIGH)
        """
        multi_tf_analysis = self.analyze_multi_timeframe_consistency(symbol)

        overall_trend = multi_tf_analysis['overall_trend']
        consistency_score = multi_tf_analysis['consistency_score']
        conflicts = multi_tf_analysis['conflicts']

        if overall_trend == 'UP' and consistency_score >= 60:
            signal = 'BUY'
            strength = consistency_score
        elif overall_trend == 'DOWN' and consistency_score >= 60:
            signal = 'SELL'
            strength = consistency_score
        else:
            signal = 'HOLD'
            strength = 0.0

        is_confirmed = (
            consistency_score >= 70 and  # سازگاری بالا
            len(conflicts) == 0 and      # بدون تعارض
            overall_trend != 'UNKNOWN'   # روند مشخص
        )

        if consistency_score >= 80 and len(conflicts) == 0:
            risk_level = 'LOW'
        elif consistency_score >= 60 and len(conflicts) <= 1:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'

        return {
            'signal': signal,
            'strength': strength,
            'is_confirmed': is_confirmed,
            'risk_level': risk_level,
            'analysis': multi_tf_analysis
        }

    def detect_market_regime_fractal(self, df: pd.DataFrame) -> str:
        """
        تشخیص رژیم بازار با استفاده از تحلیل فراکتالی

        خروجی:
        - 'TRENDING': بازار رونددار (H > 0.55)
        - 'MEAN_REVERTING': بازار میانگین‌برگشتی (H < 0.45)
        - 'RANDOM': بازار تصادفی (0.45 < H < 0.55)
        """
        fractal_analysis = self.analyze_fractal_dimension(df)
        hurst = fractal_analysis['hurst_exponent']

        if hurst > 0.55:
            return 'TRENDING'
        elif hurst < 0.45:
            return 'MEAN_REVERTING'
        else:
            return 'RANDOM'

    def find_nearest_node(self, df: pd.DataFrame, price: float, direction: str = "above", strategy: str = None) -> Optional[float]:
        nodes = self.identify_nodes_improved(df, min_amplitude=0.001)
        if not nodes:
            return None

        nodes.sort(key=lambda x: x[2], reverse=True)

        # انتخاب بهترین گره‌ها با توجه به استراتژی
        if strategy == "SUPER_SCALP" or strategy == "Super Scalp":
            # برای Super Scalp، فقط نزدیک‌ترین گره‌ها را در نظر بگیر
            top_nodes = nodes[:2]  # دو گره برتر
        else:
            top_nodes = nodes[:3]  # سه گره برتر

        node_prices = [n[1] for n in top_nodes]

        # برای Super Scalp، حداکثر فاصله گره را محدود کن
        if strategy == "SUPER_SCALP" or strategy == "Super Scalp":
            max_node_distance = 0.0005  # حداکثر 0.05% فاصله برای Super Scalp
            if direction == "above":
                valid_nodes = [p for p in node_prices if p > price and (p - price) / price <= max_node_distance]
                return min(valid_nodes) if valid_nodes else None
            else:
                valid_nodes = [p for p in node_prices if p < price and (price - p) / price <= max_node_distance]
                return max(valid_nodes) if valid_nodes else None
        else:
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
