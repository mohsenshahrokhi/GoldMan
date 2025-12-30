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
    entry_points: List[float] = None  # نقاط ورود از تایم‌فریم‌های مختلف
    trends: List[str] = None  # روندهای 3 تایم‌فریم اول
    timeframes: List[str] = None  # نام تایم‌فریم‌ها
    trend_strengths: List[float] = None  # قدرت روندهای هر تایم‌فریم
    technical_signal: str = None  # سیگنال ترکیبی اندیکاتورهای تکنیکال


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

