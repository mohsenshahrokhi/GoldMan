"""
Helper module to import python-telegram-bot without namespace collision
"""

import sys
import importlib
from typing import TYPE_CHECKING, Any

TELEGRAM_AVAILABLE = False
Update = Any
Application = Any
CommandHandler = Any
CallbackQueryHandler = Any
ContextTypes = Any
InlineKeyboardButton = Any
InlineKeyboardMarkup = Any
Conflict = Exception

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
    InlineKeyboardButton = Any
    InlineKeyboardMarkup = Any

try:
    if 'telegram' in sys.modules:
        mod = sys.modules['telegram']
        if not hasattr(mod, 'Update'):
            del sys.modules['telegram']
            if 'telegram.ext' in sys.modules:
                del sys.modules['telegram.ext']
    
    import importlib
    
    telegram_mod = importlib.import_module('telegram')
    Update = getattr(telegram_mod, 'Update', None)
    InlineKeyboardButton = getattr(telegram_mod, 'InlineKeyboardButton', None)
    InlineKeyboardMarkup = getattr(telegram_mod, 'InlineKeyboardMarkup', None)
    
    if Update:
        telegram_ext = importlib.import_module('telegram.ext')
        Application = getattr(telegram_ext, 'Application', None)
        CommandHandler = getattr(telegram_ext, 'CommandHandler', None)
        CallbackQueryHandler = getattr(telegram_ext, 'CallbackQueryHandler', None)
        ContextTypes = getattr(telegram_ext, 'ContextTypes', None)
        
        try:
            telegram_error = importlib.import_module('telegram.error')
            Conflict = getattr(telegram_error, 'Conflict', Exception)
        except:
            Conflict = Exception
        
        if Application:
            TELEGRAM_AVAILABLE = True
except Exception as e:
    pass

