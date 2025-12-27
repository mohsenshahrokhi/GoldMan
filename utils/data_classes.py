"""
Data Classes برای پروژه
"""

from dataclasses import dataclass
from typing import List
from datetime import datetime

from config.enums import TimeFrame


@dataclass
class OrderSignal:
    """سیگنال معاملاتی"""
    symbol: str
    direction: str  # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    timeframe: TimeFrame
    confidence: float
    entry_points: List[float]  # نقاط ورود از تایم‌فریم‌های مختلف


@dataclass
class MarketData:
    """داده‌های بازار"""
    symbol: str
    timeframe: TimeFrame
    open: float
    high: float
    low: float
    close: float
    volume: int
    time: datetime
    spread: float
    commission: float


@dataclass
class AccountInfo:
    """اطلاعات حساب"""
    balance: float
    equity: float
    margin: float
    free_margin: float
    profit: float
    margin_level: float

