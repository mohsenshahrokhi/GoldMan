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
    
    import importlib.util
    import site
    
    for site_pkg in site.getsitepackages():
        telegram_path = __import__('pathlib').Path(site_pkg) / 'telegram' / '__init__.py'
        if telegram_path.exists():
            spec = importlib.util.spec_from_file_location('_telegram_pkg', str(telegram_path))
            if spec and spec.loader:
                telegram_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(telegram_mod)
                
                Update = getattr(telegram_mod, 'Update', None)
                InlineKeyboardButton = getattr(telegram_mod, 'InlineKeyboardButton', None)
                InlineKeyboardMarkup = getattr(telegram_mod, 'InlineKeyboardMarkup', None)
                
                if Update:
                    ext_path = __import__('pathlib').Path(site_pkg) / 'telegram' / 'ext' / '__init__.py'
                    if ext_path.exists():
                        ext_spec = importlib.util.spec_from_file_location('_telegram_ext', str(ext_path))
                        if ext_spec and ext_spec.loader:
                            ext_mod = importlib.util.module_from_spec(ext_spec)
                            ext_spec.loader.exec_module(ext_mod)
                            
                            Application = getattr(ext_mod, 'Application', None)
                            CommandHandler = getattr(ext_mod, 'CommandHandler', None)
                            CallbackQueryHandler = getattr(ext_mod, 'CallbackQueryHandler', None)
                            ContextTypes = getattr(ext_mod, 'ContextTypes', None)
                            
                            if Application:
                                TELEGRAM_AVAILABLE = True
                                break
except Exception:
    pass

