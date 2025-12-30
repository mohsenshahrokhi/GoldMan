"""
تنظیمات ثابت و Constants
"""

MAX_RISK_PER_TRADE = 0.005  # 0.5% از سرمایه
MIN_LOT_SIZE = 0.01
MAX_LOT_SIZE = 0.5
MIN_RR_RATIO = 1.0  # Changed to 1.0 for testing. Original: 1.5
DAILY_LOSS_LIMIT = 0.02  # 2%
DRAWDOWN_PROTECTION_1 = 0.03  # 3%
DRAWDOWN_PROTECTION_2 = 0.05  # 5%
RL_OPTIMIZATION_INTERVAL = 20  # بعد از هر 20 معامله
SELECTION_TIMEOUT = 10  # 10 ثانیه

