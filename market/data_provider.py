import time
from typing import Optional

try:
    import MetaTrader5 as mt5
    import pandas as pd
except ImportError:
    mt5 = None
    pd = None

from utils.logger import logger
from utils.data_classes import MarketData
from config.enums import TimeFrame


class MarketDataProvider:
    
    def __init__(self, connection_manager):
        self.conn_mgr = connection_manager
        self.cache = {}
        self.symbol_info_cache = {}
        self.symbol_info_cache_ttl = 300
        
        self.timeframe_ttl = {
            TimeFrame.M1: 5,
            TimeFrame.M3: 10,
            TimeFrame.M5: 15,
            TimeFrame.M15: 30,
            TimeFrame.H1: 60,
            TimeFrame.H4: 300
        }
    
    def get_ohlc_data(self, symbol: str, timeframe: TimeFrame, count: int = 1000) -> Optional[pd.DataFrame]:
        if mt5 is None or pd is None:
            logger.error("MetaTrader5 or pandas is not installed")
            return None
            
        cache_key = (symbol, timeframe)
        ttl = self.timeframe_ttl.get(timeframe, 60)
        
        if cache_key in self.cache:
            data, last_update = self.cache[cache_key]
            if time.time() - last_update < ttl:
                if len(data) >= count:
                    return data.tail(count)
                else:
                    return data
        
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe.value, 0, count)
            if rates is None or len(rates) == 0:
                logger.warning(f"No data received for {symbol} on {timeframe}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            if cache_key in self.cache:
                cached_df, _ = self.cache[cache_key]
                if not cached_df.empty and len(cached_df) > 0:
                    last_cached_time = cached_df.index[-1]
                    new_data = df[df.index > last_cached_time]
                    if not new_data.empty:
                        updated_df = pd.concat([cached_df, new_data])
                        updated_df = updated_df.tail(count * 2)
                        self.cache[cache_key] = (updated_df, time.time())
                        return updated_df.tail(count)
            
            self.cache[cache_key] = (df, time.time())
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")
            return None
    
    def get_latest_candle(self, symbol: str, timeframe: TimeFrame) -> Optional[MarketData]:
        df = self.get_ohlc_data(symbol, timeframe, 1)
        if df is None or df.empty:
            return None
        
        try:
            if mt5 is None:
                return None
                
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info is None:
                return None
            
            latest = df.iloc[-1]
            commission = getattr(symbol_info, 'commission', 0.0)
            if hasattr(symbol_info, 'commission_mode') and symbol_info.commission_mode == mt5.COMMISSION_MONEY:
                commission = commission / symbol_info.trade_contract_size if symbol_info.trade_contract_size > 0 else 0.0
            
            return MarketData(
                symbol=symbol,
                timeframe=timeframe,
                open=latest['open'],
                high=latest['high'],
                low=latest['low'],
                close=latest['close'],
                volume=int(latest['tick_volume']),
                time=latest.name,
                spread=symbol_info.spread * symbol_info.point,
                commission=commission
            )
        except Exception as e:
            logger.error(f"Error retrieving latest candle: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional:
        if mt5 is None:
            return None
        
        if symbol in self.symbol_info_cache:
            info, last_update = self.symbol_info_cache[symbol]
            if time.time() - last_update < self.symbol_info_cache_ttl:
                return info
        
        try:
            info = mt5.symbol_info(symbol)
            if info is not None:
                self.symbol_info_cache[symbol] = (info, time.time())
            return info
        except Exception as e:
            logger.error(f"Error retrieving symbol info for {symbol}: {e}")
            return None
    
    def is_new_candle(self, symbol: str, timeframe: TimeFrame) -> bool:
        if mt5 is None or pd is None:
            return False
        
        try:
            if timeframe.value is None:
                logger.warning(f"[CANDLE_DETECTION] Timeframe {timeframe} not supported by MT5")
                return False
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe.value, 0, 1)
            if rates is None or len(rates) == 0:
                logger.warning(f"[CANDLE_DETECTION] No data received for {symbol} on {timeframe.value}")
                return False
            
            current_candle_time = pd.to_datetime(rates[-1]['time'], unit='s')
            
            cache_key = (symbol, timeframe)
            if cache_key not in self.cache:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                self.cache[cache_key] = (df, time.time())
                logger.info(f"[CANDLE_DETECTION] First candle check: Symbol={symbol}, Timeframe={timeframe.value}, Time={current_candle_time}")
                return True
            
            cached_data, _ = self.cache[cache_key]
            if cached_data.empty:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                self.cache[cache_key] = (df, time.time())
                logger.info(f"[CANDLE_DETECTION] Cache was empty, updating: Symbol={symbol}, Timeframe={timeframe.value}, Time={current_candle_time}")
                return True
            
            if isinstance(cached_data.index, pd.DatetimeIndex):
                cached_time = cached_data.index[-1]
            elif 'time' in cached_data.columns:
                cached_time = pd.to_datetime(cached_data['time'].iloc[-1], unit='s')
            else:
                cached_time = pd.to_datetime(cached_data.index[-1], unit='s')
            is_new = current_candle_time > cached_time
            
            if is_new:
                if cache_key in self.cache:
                    cached_df, _ = self.cache[cache_key]
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    
                    if not cached_df.empty:
                        new_data = df[~df.index.isin(cached_df.index)]
                        if not new_data.empty:
                            updated_df = pd.concat([cached_df, new_data])
                            updated_df = updated_df.tail(1000)
                            self.cache[cache_key] = (updated_df, time.time())
                        else:
                            self.cache[cache_key] = (df, time.time())
                    else:
                        self.cache[cache_key] = (df, time.time())
                else:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    self.cache[cache_key] = (df, time.time())
                
                logger.info(f"[CANDLE_DETECTION] New candle detected: Symbol={symbol}, Timeframe={timeframe.value}, OldTime={cached_time}, NewTime={current_candle_time}")
            else:
                if time.time() % 30 < 1:
                    logger.debug(f"[CANDLE_DETECTION] No new candle yet: Symbol={symbol}, Timeframe={timeframe.value}, CurrentTime={current_candle_time}, CachedTime={cached_time}")
            
            return is_new
        except Exception as e:
            logger.error(f"Error checking for new candle: {e}")
            return False
    
    def clear_cache(self, symbol: str = None, timeframe: TimeFrame = None):
        if symbol and timeframe:
            key = (symbol, timeframe)
            if key in self.cache:
                del self.cache[key]
        else:
            self.cache.clear()
            self.symbol_info_cache.clear()
    
    def get_cache_stats(self) -> dict:
        return {
            'ohlc_cache_size': len(self.cache),
            'symbol_info_cache_size': len(self.symbol_info_cache),
            'total_memory_estimate': len(self.cache) * 100 + len(self.symbol_info_cache) * 1
        }
