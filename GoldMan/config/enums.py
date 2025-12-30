"""
Enum های پروژه
"""

from enum import Enum

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


class SymbolType(Enum):
    """انواع نمادهای معاملاتی"""
    XAUUSD = "XAUUSD"
    EURUSD = "EURUSD"
    YM = "YM"
    BTCUSD = "BTCUSD"


class StrategyType(Enum):
    """انواع استراتژی‌های معاملاتی"""
    DAY_TRADING = "Day Trading"
    SCALP = "Scalp"
    SUPER_SCALP = "Super Scalp"


class TimeFrame(Enum):
    """تایم‌فریم‌های معاملاتی"""
    if mt5:
        M1 = mt5.TIMEFRAME_M1
        M3 = mt5.TIMEFRAME_M3
        M5 = mt5.TIMEFRAME_M5
        M15 = mt5.TIMEFRAME_M15
        H1 = mt5.TIMEFRAME_H1
        H4 = mt5.TIMEFRAME_H4
    else:
        M1 = None
        M3 = None
        M5 = None
        M15 = None
        H1 = None
        H4 = None

