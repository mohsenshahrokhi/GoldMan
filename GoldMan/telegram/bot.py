"""
ربات تلگرام برای مانیتورینگ و کنترل
"""

import asyncio
import threading
from typing import Optional, TYPE_CHECKING, Any

from ._telegram_import import (
    TELEGRAM_AVAILABLE,
    Update,
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)

from utils.logger import logger
from config.enums import SymbolType, StrategyType
from config.constants import SELECTION_TIMEOUT


class TelegramBot:
    """ربات تلگرام برای مانیتورینگ و کنترل"""
    
    def __init__(self, token: str, main_controller):
        if not TELEGRAM_AVAILABLE:
            raise ImportError("python-telegram-bot is not installed. Please install it with: pip install python-telegram-bot")
            
        self.token = token
        self.main_controller = main_controller
        self.application = None
        self.selected_symbol = None
        self.selected_strategy = None
        self.selection_timeout = SELECTION_TIMEOUT
        self.selection_timer = None
    
    async def start(self):
        """شروع ربات تلگرام"""
        self.application = Application.builder().token(self.token).build()
        
        # اضافه کردن handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("report", self.report_command))
        self.application.add_handler(CommandHandler("params", self.params_command))
        self.application.add_handler(CommandHandler("stop", self.stop_command))
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
        logger.info("Telegram bot started")
    
    async def stop(self):
        """توقف ربات تلگرام"""
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
    
    async def start_command(self, update: 'Update', context: Any):
        """دستور /start"""
        if self.main_controller.is_running():
            await update.message.reply_text(
                "🤖 ربات در حال اجرا است!\n\n"
                "دستورات موجود:\n"
                "/status - وضعیت فعلی\n"
                "/report - گزارش عملکرد\n"
                "/params - مشاهده پارامترها\n"
                "/stop - توقف اضطراری"
            )
            return
        
        keyboard = [
            [InlineKeyboardButton("Day Trading", callback_data="strategy_DAY_TRADING")],
            [InlineKeyboardButton("Scalp", callback_data="strategy_SCALP")],
            [InlineKeyboardButton("Super Scalp", callback_data="strategy_SUPER_SCALP")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 ربات GoldMan\n\n"
            "لطفاً استراتژی معاملاتی را انتخاب کنید:\n"
            f"⏱️ زمان باقیمانده: {self.selection_timeout} ثانیه\n"
            "در صورت عدم انتخاب، Day Trading به عنوان پیش‌فرض انتخاب می‌شود.",
            reply_markup=reply_markup
        )
        
        self.selection_timer = threading.Timer(
            self.selection_timeout,
            lambda: asyncio.create_task(self.default_strategy_selection(update))
        )
        self.selection_timer.start()
    
    async def default_strategy_selection(self, update: 'Update'):
        """انتخاب استراتژی پیش‌فرض در صورت عدم انتخاب"""
        if not self.selected_strategy:
            self.selected_strategy = StrategyType.DAY_TRADING
            await update.message.reply_text(
                f"✅ استراتژی: {self.selected_strategy.value} (پیش‌فرض)\n\n"
                "در حال نمایش منوی انتخاب نماد..."
            )
            await self.show_symbol_menu(update)
    
    async def button_callback(self, update: 'Update', context: Any):
        """پردازش کلیک دکمه"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data.startswith("strategy_"):
            strategy_name = data.split("_")[1]
            if strategy_name == "DAY_TRADING":
                self.selected_strategy = StrategyType.DAY_TRADING
            elif strategy_name == "SCALP":
                self.selected_strategy = StrategyType.SCALP
            elif strategy_name == "SUPER_SCALP":
                self.selected_strategy = StrategyType.SUPER_SCALP
            
            if self.selection_timer:
                self.selection_timer.cancel()
            
            await query.edit_message_text(
                f"✅ استراتژی: {self.selected_strategy.value}\n\n"
                "در حال نمایش منوی انتخاب نماد..."
            )
            await self.show_symbol_menu(query)
        
        elif data.startswith("symbol_"):
            symbol_name = data.split("_")[1]
            self.selected_symbol = SymbolType[symbol_name]
            if self.selection_timer:
                self.selection_timer.cancel()
            
            await query.edit_message_text(
                f"✅ استراتژی: {self.selected_strategy.value}\n"
                f"✅ نماد: {self.selected_symbol.value}\n\n"
                "🚀 در حال راه‌اندازی ربات..."
            )
            
            await self.main_controller.start_trading(
                self.selected_symbol,
                self.selected_strategy
            )
        
        elif data.startswith("report_"):
            report_type = data.split("_")[1]
            reporter = self.main_controller.reporter
            
            if report_type == "daily":
                report_text = reporter.generate_daily_report()
            elif report_type == "weekly":
                report_text = reporter.generate_weekly_report()
            elif report_type == "monthly":
                report_text = reporter.generate_monthly_report()
            else:
                report_text = "❌ نوع گزارش نامعتبر است."
            
            await query.edit_message_text(report_text)
        
        elif data == "stop_confirm":
            await query.edit_message_text("⏹️ در حال توقف ربات...")
            await self.main_controller.stop()
        
        elif data == "stop_cancel":
            await query.edit_message_text("✅ توقف لغو شد.")
    
    async def show_symbol_menu(self, query):
        """نمایش منوی انتخاب نماد"""
        keyboard = [
            [
                InlineKeyboardButton("XAUUSD (طلا)", callback_data="symbol_XAUUSD"),
                InlineKeyboardButton("EURUSD", callback_data="symbol_EURUSD")
            ],
            [
                InlineKeyboardButton("YM (Dow Jones)", callback_data="symbol_YM"),
                InlineKeyboardButton("BTCUSD (بیت‌کوین)", callback_data="symbol_BTCUSD")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            f"✅ استراتژی: {self.selected_strategy.value}\n\n"
            "لطفاً نماد معاملاتی را انتخاب کنید:\n"
            f"⏱️ زمان باقیمانده: {self.selection_timeout} ثانیه\n"
            "در صورت عدم انتخاب، BTCUSD به عنوان پیش‌فرض انتخاب می‌شود.",
            reply_markup=reply_markup
        )
        
        self.selection_timer = threading.Timer(
            self.selection_timeout,
            lambda: asyncio.create_task(self.default_symbol_selection(query))
        )
        self.selection_timer.start()
    
    async def default_symbol_selection(self, query):
        """انتخاب نماد پیش‌فرض در صورت عدم انتخاب"""
        if not self.selected_symbol:
            self.selected_symbol = SymbolType.BTCUSD
        
        await query.edit_message_text(
            f"✅ استراتژی: {self.selected_strategy.value}\n"
            f"✅ نماد: {self.selected_symbol.value} (پیش‌فرض)\n\n"
            "🚀 در حال راه‌اندازی ربات..."
        )
        
        await self.main_controller.start_trading(
            self.selected_symbol,
            self.selected_strategy
        )
    
    async def status_command(self, update: 'Update', context: Any):
        """دستور /status"""
        if not self.main_controller.is_running():
            await update.message.reply_text("❌ ربات در حال اجرا نیست.")
            return
        
        account_info = self.main_controller.conn_mgr.get_account_info()
        if account_info:
            status_text = f"""
📊 وضعیت فعلی ربات:

💰 حساب:
• موجودی: ${account_info.balance:.2f}
• Equity: ${account_info.equity:.2f}
• سود/زیان: ${account_info.profit:.2f}
• Margin Level: {account_info.margin_level:.2f}%

📈 معاملات:
• نماد فعلی: {self.main_controller.current_symbol.value if self.main_controller.current_symbol else 'N/A'}
• استراتژی: {self.main_controller.current_strategy.value if self.main_controller.current_strategy else 'N/A'}
• معامله باز: {'✅ بله' if self.main_controller.trade_executor.has_open_position() else '❌ خیر'}
"""
        else:
            status_text = "❌ خطا در دریافت اطلاعات حساب"
        
        await update.message.reply_text(status_text)
    
    async def report_command(self, update: 'Update', context: Any):
        """دستور /report"""
        keyboard = [
            [InlineKeyboardButton("📅 روزانه", callback_data="report_daily")],
            [InlineKeyboardButton("📆 هفتگی", callback_data="report_weekly")],
            [InlineKeyboardButton("📊 ماهانه", callback_data="report_monthly")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "لطفاً نوع گزارش را انتخاب کنید:",
            reply_markup=reply_markup
        )
    
    async def params_command(self, update: 'Update', context: Any):
        """دستور /params - نمایش پارامترها"""
        if not self.main_controller.is_running():
            await update.message.reply_text("❌ ربات در حال اجرا نیست.")
            return
        
        symbol = self.main_controller.current_symbol.value if self.main_controller.current_symbol else "N/A"
        strategy = self.main_controller.current_strategy.value if self.main_controller.current_strategy else "N/A"
        
        # دریافت وزن‌های RL
        if symbol != "N/A" and strategy != "N/A":
            rl_engine = self.main_controller.rl_engine
            weights = rl_engine.get_weights(symbol, strategy)
            
            from config.constants import (
                MAX_RISK_PER_TRADE, MIN_RR_RATIO, DAILY_LOSS_LIMIT,
                DRAWDOWN_PROTECTION_1, DRAWDOWN_PROTECTION_2
            )
            
            params_text = f"""
⚙️ پارامترهای فعلی:

📊 نماد: {symbol}
📈 استراتژی: {strategy}

🎯 وزن‌های RL (SL/TP Methods):
• Node-Based: {weights.get('node', 0.25):.2%}
• ATR-Based: {weights.get('atr', 0.25):.2%}
• GARCH-Based: {weights.get('garch', 0.25):.2%}
• Fixed RR: {weights.get('fixed_rr', 0.25):.2%}

📊 پارامترهای ریسک:
• Max Risk Per Trade: {MAX_RISK_PER_TRADE:.2%}
• Min R/R Ratio: {MIN_RR_RATIO}
• Daily Loss Limit: {DAILY_LOSS_LIMIT:.2%}
• Drawdown Protection 1: {DRAWDOWN_PROTECTION_1:.2%}
• Drawdown Protection 2: {DRAWDOWN_PROTECTION_2:.2%}
"""
        else:
            params_text = "❌ اطلاعات پارامترها در دسترس نیست."
        
        await update.message.reply_text(params_text)
    
    async def stop_command(self, update: 'Update', context: Any):
        """دستور /stop - توقف اضطراری"""
        keyboard = [
            [InlineKeyboardButton("✅ تایید توقف", callback_data="stop_confirm")],
            [InlineKeyboardButton("❌ انصراف", callback_data="stop_cancel")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "⚠️ آیا مطمئن هستید که می‌خواهید ربات را متوقف کنید؟",
            reply_markup=reply_markup
        )
    
    async def send_message(self, chat_id: int, text: str):
        """ارسال پیام به کاربر"""
        if self.application:
            await self.application.bot.send_message(chat_id=chat_id, text=text)

