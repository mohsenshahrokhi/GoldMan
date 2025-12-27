import sqlite3
import json
from datetime import datetime
from typing import Dict, Optional, List
from contextlib import contextmanager

from utils.logger import logger


class DatabaseManager:
    
    def __init__(self, db_path: str = "goldman.db"):
        self.db_path = db_path
        self.conn = None
        self.init_database()
    
    def init_database(self):
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self.create_tables()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER UNIQUE,
                    symbol TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    lot_size REAL,
                    profit REAL,
                    entry_time DATETIME,
                    exit_time DATETIME,
                    strategy TEXT,
                    status TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rl_experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    strategy TEXT,
                    timeframe TEXT,
                    state TEXT,
                    action TEXT,
                    reward REAL,
                    next_state TEXT,
                    timestamp DATETIME,
                    trade_id INTEGER
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rl_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    strategy TEXT,
                    parameter_name TEXT,
                    parameter_value REAL,
                    updated_at DATETIME,
                    UNIQUE(symbol, strategy, parameter_name)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rl_entry_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    strategy TEXT,
                    entry_point_name TEXT,
                    weight REAL,
                    updated_at DATETIME,
                    UNIQUE(symbol, strategy, entry_point_name)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rl_trend_weights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    strategy TEXT,
                    trend_name TEXT,
                    weight REAL,
                    updated_at DATETIME,
                    UNIQUE(symbol, strategy, trend_name)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nds_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    timeframe TEXT,
                    parameter_name TEXT,
                    parameter_value REAL,
                    updated_at DATETIME,
                    UNIQUE(symbol, timeframe, parameter_name)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_date DATE,
                    symbol TEXT,
                    strategy TEXT,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    total_profit REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL
                )
            """)
            
            self.create_indexes(cursor)
            self.conn.commit()
            logger.info("Database tables and indexes created successfully")
        except sqlite3.Error as e:
            logger.error(f"Error creating database tables: {e}")
            self.conn.rollback()
            raise
    
    def create_indexes(self, cursor):
        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_strategy 
                ON trades(symbol, strategy)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_status 
                ON trades(status)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_entry_time 
                ON trades(entry_time)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_ticket 
                ON trades(ticket)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rl_experiences_symbol_strategy 
                ON rl_experiences(symbol, strategy)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rl_experiences_timestamp 
                ON rl_experiences(timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rl_experiences_trade_id 
                ON rl_experiences(trade_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rl_weights_symbol_strategy 
                ON rl_weights(symbol, strategy)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rl_entry_weights_symbol_strategy 
                ON rl_entry_weights(symbol, strategy)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_rl_trend_weights_symbol_strategy 
                ON rl_trend_weights(symbol, strategy)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_performance_reports_symbol_strategy 
                ON performance_reports(symbol, strategy)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_performance_reports_date 
                ON performance_reports(report_date DESC)
            """)
            
            logger.info("Database indexes created successfully")
        except sqlite3.Error as e:
            logger.error(f"Error creating database indexes: {e}")
            raise
    
    @contextmanager
    def get_cursor(self):
        cursor = self.conn.cursor()
        try:
            yield cursor
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            self.conn.rollback()
            raise
        finally:
            cursor.close()
    
    def save_trade(self, trade_data: Dict) -> bool:
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT OR REPLACE INTO trades 
                    (ticket, symbol, direction, entry_price, stop_loss, take_profit, 
                     lot_size, entry_time, strategy, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data.get('ticket'),
                    trade_data.get('symbol'),
                    trade_data.get('direction'),
                    trade_data.get('entry_price'),
                    trade_data.get('stop_loss'),
                    trade_data.get('take_profit'),
                    trade_data.get('lot_size'),
                    trade_data.get('entry_time'),
                    trade_data.get('strategy'),
                    trade_data.get('status', 'OPEN')
                ))
                self.conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error saving trade: {e}, TradeData: {trade_data}")
            return False
        except KeyError as e:
            logger.error(f"Missing required field in trade_data: {e}, TradeData: {trade_data}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving trade: {e}, TradeData: {trade_data}")
            return False
    
    def update_trade(self, ticket: int, updates: Dict) -> bool:
        if not updates:
            logger.warning(f"Empty updates dictionary for ticket {ticket}")
            return False
        
        allowed_fields = {
            'exit_price', 'stop_loss', 'take_profit', 'profit', 
            'exit_time', 'status', 'lot_size'
        }
        
        filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}
        if not filtered_updates:
            logger.warning(f"No valid fields in updates for ticket {ticket}")
            return False
        
        try:
            with self.get_cursor() as cursor:
                set_clause = ", ".join([f"{k} = ?" for k in filtered_updates.keys()])
                values = list(filtered_updates.values()) + [ticket]
                cursor.execute(
                    f"UPDATE trades SET {set_clause} WHERE ticket = ?",
                    values
                )
                if cursor.rowcount == 0:
                    logger.warning(f"No trade found with ticket {ticket}")
                    return False
                self.conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error updating trade {ticket}: {e}, Updates: {updates}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error updating trade {ticket}: {e}, Updates: {updates}")
            return False
    
    def get_rl_weights(self, symbol: str, strategy: str) -> Dict:
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT parameter_name, parameter_value 
                    FROM rl_weights 
                    WHERE symbol = ? AND strategy = ?
                """, (symbol, strategy))
                return {row['parameter_name']: row['parameter_value'] 
                        for row in cursor.fetchall()}
        except sqlite3.Error as e:
            logger.error(f"Error getting RL weights for {symbol}-{strategy}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting RL weights for {symbol}-{strategy}: {e}")
            return {}
    
    def save_rl_weights(self, symbol: str, strategy: str, weights: Dict) -> bool:
        if not weights:
            logger.warning(f"Empty weights dictionary for {symbol}-{strategy}")
            return False
        
        try:
            with self.get_cursor() as cursor:
                now = datetime.now()
                cursor.executemany("""
                    INSERT OR REPLACE INTO rl_weights 
                    (symbol, strategy, parameter_name, parameter_value, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    (symbol, strategy, param_name, param_value, now)
                    for param_name, param_value in weights.items()
                ])
                self.conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error saving RL weights for {symbol}-{strategy}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving RL weights for {symbol}-{strategy}: {e}")
            return False
    
    def save_experience(self, experience: Dict) -> bool:
        required_fields = ['symbol', 'strategy', 'timeframe', 'state', 'action', 'reward']
        if not all(field in experience for field in required_fields):
            missing = [f for f in required_fields if f not in experience]
            logger.error(f"Missing required fields in experience: {missing}")
            return False
        
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO rl_experiences 
                    (symbol, strategy, timeframe, state, action, reward, next_state, timestamp, trade_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experience['symbol'],
                    experience['strategy'],
                    experience['timeframe'],
                    json.dumps(experience['state']),
                    json.dumps(experience['action']),
                    experience['reward'],
                    json.dumps(experience.get('next_state', {})),
                    datetime.now(),
                    experience.get('trade_id')
                ))
                self.conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error saving experience: {e}, Experience: {experience}")
            return False
        except (json.JSONEncodeError, TypeError) as e:
            logger.error(f"Error serializing experience data: {e}, Experience: {experience}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving experience: {e}, Experience: {experience}")
            return False
    
    def get_trade_count(self, symbol: str, strategy: str) -> int:
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT COUNT(*) FROM trades 
                    WHERE symbol = ? AND strategy = ? AND status = 'CLOSED'
                """, (symbol, strategy))
                result = cursor.fetchone()
                return result[0] if result else 0
        except sqlite3.Error as e:
            logger.error(f"Error getting trade count for {symbol}-{strategy}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error getting trade count for {symbol}-{strategy}: {e}")
            return 0
    
    def get_rl_entry_weights(self, symbol: str, strategy: str) -> Dict:
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT entry_point_name, weight
                    FROM rl_entry_weights
                    WHERE symbol = ? AND strategy = ?
                """, (symbol, strategy))
                return {row['entry_point_name']: row['weight']
                        for row in cursor.fetchall()}
        except sqlite3.Error as e:
            logger.error(f"Error getting RL entry weights for {symbol}-{strategy}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting RL entry weights for {symbol}-{strategy}: {e}")
            return {}
    
    def save_rl_entry_weights(self, symbol: str, strategy: str, entry_weights: Dict) -> bool:
        if not entry_weights:
            logger.warning(f"Empty entry weights dictionary for {symbol}-{strategy}")
            return False
        
        try:
            with self.get_cursor() as cursor:
                now = datetime.now()
                cursor.executemany("""
                    INSERT OR REPLACE INTO rl_entry_weights
                    (symbol, strategy, entry_point_name, weight, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    (symbol, strategy, entry_name, weight, now)
                    for entry_name, weight in entry_weights.items()
                ])
                self.conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error saving RL entry weights for {symbol}-{strategy}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving RL entry weights for {symbol}-{strategy}: {e}")
            return False
    
    def get_rl_trend_weights(self, symbol: str, strategy: str) -> Dict:
        try:
            with self.get_cursor() as cursor:
                cursor.execute("""
                    SELECT trend_name, weight
                    FROM rl_trend_weights
                    WHERE symbol = ? AND strategy = ?
                """, (symbol, strategy))
                return {row['trend_name']: row['weight']
                        for row in cursor.fetchall()}
        except sqlite3.Error as e:
            logger.error(f"Error getting RL trend weights for {symbol}-{strategy}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting RL trend weights for {symbol}-{strategy}: {e}")
            return {}
    
    def save_rl_trend_weights(self, symbol: str, strategy: str, trend_weights: Dict) -> bool:
        if not trend_weights:
            logger.warning(f"Empty trend weights dictionary for {symbol}-{strategy}")
            return False
        
        try:
            with self.get_cursor() as cursor:
                now = datetime.now()
                cursor.executemany("""
                    INSERT OR REPLACE INTO rl_trend_weights
                    (symbol, strategy, trend_name, weight, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    (symbol, strategy, trend_name, weight, now)
                    for trend_name, weight in trend_weights.items()
                ])
                self.conn.commit()
                return True
        except sqlite3.Error as e:
            logger.error(f"Error saving RL trend weights for {symbol}-{strategy}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving RL trend weights for {symbol}-{strategy}: {e}")
            return False
    
    def close(self):
        try:
            if self.conn:
                self.conn.close()
                logger.info("Database connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
