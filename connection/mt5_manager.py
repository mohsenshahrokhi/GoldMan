import time
from typing import Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

from utils.logger import logger
from utils.data_classes import AccountInfo


class ConnectionManager:

    def __init__(self):
        self.connected = False
        self.account_info = None
        self.heartbeat_interval = 10
        self.last_heartbeat = None
        self.account_info_cache_ttl = 5
        self.account_info_cache_time = 0
    
    def connect(self, login: int = None, password: str = None, server: str = None) -> bool:
        if mt5 is None:
            logger.error("MetaTrader5 is not installed")
            return False
            
        try:
            if not mt5.initialize():
                logger.error(f"Error initializing MT5: {mt5.last_error()}")
                return False
            
            if login and password and server:
                authorized = mt5.login(login, password=password, server=server)
                if not authorized:
                    logger.error(f"Authentication error: {mt5.last_error()}")
                    return False
            
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Error retrieving account information")
                return False
            
            if not self.validate_account(account_info):
                return False
            
            self.account_info = account_info
            self.connected = True
            self.last_heartbeat = time.time()
            logger.info(f"[SENSITIVE] MT5 connection established: Account={account_info.login}, Balance={account_info.balance:.2f}, Equity={account_info.equity:.2f}, Server={server if server else 'Default'}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False
    
    def validate_account(self, account_info) -> bool:
        if account_info.balance < 500:
            logger.error(f"[SENSITIVE] Account validation failed - Insufficient balance: Balance={account_info.balance:.2f}, Required=500.00")
            return False
        
        if account_info is None:
            logger.error("[SENSITIVE] Account validation failed - Account info is None")
            return False
        
        logger.info(f"[SENSITIVE] Account validated: Login={account_info.login}, Balance={account_info.balance:.2f}, Equity={account_info.equity:.2f}, MarginLevel={account_info.margin_level:.2f}%")
        return True
    
    def check_connection(self) -> bool:
        if not self.connected or mt5 is None:
            return False
        
        try:
            account_info = mt5.account_info()
            if account_info is None:
                self.connected = False
                return False
            
            current_time = time.time()
            if current_time - self.last_heartbeat > self.heartbeat_interval:
                self.last_heartbeat = current_time
                self.account_info = account_info
            
            return True
        except Exception as e:
            logger.error(f"Error checking connection: {e}")
            self.connected = False
            return False
    
    def get_account_info(self) -> Optional[AccountInfo]:
        if not self.check_connection():
            return None

        try:
            current_time = time.time()
            if (self.account_info is not None and 
                current_time - self.account_info_cache_time < self.account_info_cache_ttl):
                return self.account_info
            
            account = mt5.account_info()
            if account is None:
                return None

            free_margin = getattr(account, 'free_margin', account.equity - account.margin)
            
            account_info = AccountInfo(
                balance=account.balance,
                equity=account.equity,
                margin=account.margin,
                free_margin=free_margin,
                profit=account.profit,
                margin_level=account.margin_level
            )
            
            self.account_info = account_info
            self.account_info_cache_time = current_time
            
            return account_info
        except Exception as e:
            logger.error(f"Error retrieving account info: {e}")
            return None
    
    def disconnect(self):
        if mt5:
            mt5.shutdown()
        self.connected = False
        logger.info("Disconnected from MT5")
