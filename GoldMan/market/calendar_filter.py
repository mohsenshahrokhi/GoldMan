from datetime import datetime

from config.enums import SymbolType


class CalendarFilter:
    
    def __init__(self):
        self.crypto_symbols = [SymbolType.BTCUSD]
        self.weekend_trading_symbols = [SymbolType.BTCUSD]
    
    def is_trading_time(self, symbol_type: SymbolType) -> bool:
        now = datetime.now()
        weekday = now.weekday()
        
        if symbol_type in self.crypto_symbols:
            return True
        
        if weekday >= 5:
            return False
        
        hour = now.hour
        
        if symbol_type == SymbolType.XAUUSD:
            return True
        
        if symbol_type == SymbolType.EURUSD:
            return 8 <= hour <= 22
        
        if symbol_type == SymbolType.YM:
            return 9 <= hour <= 16
        
        return True
    
    def should_switch_to_crypto(self, current_symbol: SymbolType) -> bool:
        if current_symbol in self.crypto_symbols:
            return False
        
        if not self.is_trading_time(current_symbol):
            return True
        
        return False
