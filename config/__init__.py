"""
ماژول تنظیمات و Constants
"""

from .constants import (
    MAX_RISK_PER_TRADE,
    MIN_LOT_SIZE,
    MAX_LOT_SIZE,
    MIN_RR_RATIO,
    DAILY_LOSS_LIMIT,
    DRAWDOWN_PROTECTION_1,
    DRAWDOWN_PROTECTION_2,
    RL_OPTIMIZATION_INTERVAL,
    SELECTION_TIMEOUT
)

from .enums import SymbolType, StrategyType, TimeFrame

__all__ = [
    'MAX_RISK_PER_TRADE',
    'MIN_LOT_SIZE',
    'MAX_LOT_SIZE',
    'MIN_RR_RATIO',
    'DAILY_LOSS_LIMIT',
    'DRAWDOWN_PROTECTION_1',
    'DRAWDOWN_PROTECTION_2',
    'RL_OPTIMIZATION_INTERVAL',
    'SELECTION_TIMEOUT',
    'SymbolType',
    'StrategyType',
    'TimeFrame'
]

