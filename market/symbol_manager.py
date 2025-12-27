from typing import List

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from utils.logger import logger
from config.enums import SymbolType


class SymbolManager:
    
    def __init__(self, connection_manager):
        self.conn_mgr = connection_manager
        self.symbol_mapping = {}
        self.current_symbol = None
        self.symbol_prefixes = {}
        self.symbol_suffixes = {}
        self.detect_symbol_variations()
    
    def detect_symbol_variations(self):
        if mt5 is None:
            logger.warning("MetaTrader5 is not installed")
            return
            
        try:
            symbols = mt5.symbols_get()
            base_symbols = ["XAUUSD", "EURUSD", "YM", "BTCUSD"]
            
            for base in base_symbols:
                variations = [s.name for s in symbols if base in s.name]
                if variations:
                    for var in variations:
                        if var.startswith(base):
                            suffix = var[len(base):]
                            self.symbol_suffixes[base] = suffix
                        elif var.endswith(base):
                            prefix = var[:-len(base)]
                            self.symbol_prefixes[base] = prefix
                        else:
                            idx = var.find(base)
                            if idx > 0:
                                self.symbol_prefixes[base] = var[:idx]
                            if idx + len(base) < len(var):
                                self.symbol_suffixes[base] = var[idx + len(base):]
                    
                    self.symbol_mapping[base] = variations[0]
                    logger.info(f"Symbol {base} detected as {variations[0]}")
        except Exception as e:
            logger.error(f"Error detecting symbols: {e}")
    
    def get_symbol_name(self, symbol_type: SymbolType) -> str:
        base = symbol_type.value
        if base in self.symbol_mapping:
            return self.symbol_mapping[base]
        return base
    
    def is_symbol_active(self, symbol: str) -> bool:
        if mt5 is None:
            return False
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
            return symbol_info.visible and symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL
        except Exception as e:
            logger.error(f"Error checking symbol {symbol}: {e}")
            return False
    
    def switch_symbol(self, new_symbol_type: SymbolType) -> bool:
        new_symbol = self.get_symbol_name(new_symbol_type)
        if self.is_symbol_active(new_symbol):
            self.current_symbol = new_symbol
            logger.info(f"Switched to symbol {new_symbol}")
            return True
        return False
    
    def get_active_symbols(self) -> List[SymbolType]:
        active = []
        for symbol_type in SymbolType:
            symbol_name = self.get_symbol_name(symbol_type)
            if self.is_symbol_active(symbol_name):
                active.append(symbol_type)
        return active
