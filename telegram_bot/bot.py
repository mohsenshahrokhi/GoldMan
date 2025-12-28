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

from ._telegram_import import Conflict

from utils.logger import logger
from config.enums import SymbolType, StrategyType
from config.constants import SELECTION_TIMEOUT


class TelegramBot:
    """ربات تلگرام برای مانیتورینگ و کنترل"""
    
    def __init__(self, token: str, main_controller):
        if not TELEGRAM_AVAILABLE:
            error_msg = "python-telegram-bot is not installed. Please install it with: pip install python-telegram-bot"
            logger.error(error_msg)
            raise ImportError(error_msg)
            
        self.token = token
        self.main_controller = main_controller
        self.application = None
        self.selected_symbol = None
        self.selected_strategy = None
        self.selection_timeout = SELECTION_TIMEOUT
        self.selection_timer = None
        self.event_loop = None
        self.chat_ids = set()
        
        import os
        channel_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        if channel_chat_id:
            try:
                self.chat_ids.add(int(channel_chat_id))
                logger.info(f"Channel chat_id loaded from environment: {channel_chat_id}")
            except ValueError:
                logger.warning(f"Invalid TELEGRAM_CHAT_ID format: {channel_chat_id}")
        
        logger.info("TelegramBot initialized successfully")
    
    async def start(self):
        """شروع ربات تلگرام"""
        self.application = Application.builder().token(self.token).build()
        
        # حذف webhook در صورت وجود و صبر برای اطمینان
        try:
            await self.application.bot.delete_webhook(drop_pending_updates=True)
            logger.info("Webhook deleted (if existed)")
            await asyncio.sleep(2)  # صبر 2 ثانیه برای اطمینان از حذف webhook
        except Exception as e:
            logger.warning(f"Error deleting webhook: {e}")
        
        # اضافه کردن handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("report", self.report_command))
        self.application.add_handler(CommandHandler("params", self.params_command))
        self.application.add_handler(CommandHandler("stop", self.stop_command))
        self.application.add_handler(CommandHandler("add_channel", self.add_channel_command))
        self.application.add_handler(CommandHandler("get_chat_id", self.get_chat_id_command))
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        
        await self.application.initialize()
        await self.application.start()
        
        # ذخیره event loop برای استفاده در timer callbacks
        try:
            self.event_loop = self.application.updater._network_loop._loop
        except:
            self.event_loop = asyncio.get_event_loop()
        
        # بررسی conflict قبل از شروع polling
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # تست کردن getUpdates قبل از شروع polling
                test_updates = await self.application.bot.get_updates(limit=1, timeout=1)
                logger.info("Bot is ready. No conflicts detected.")
                break
            except Exception as e:
                if "Conflict" in str(e):
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = retry_count * 3
                        logger.warning(f"Conflict detected. Waiting {wait_time} seconds before retry... (Attempt {retry_count}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        # حذف webhook دوباره
                        try:
                            await self.application.bot.delete_webhook(drop_pending_updates=True)
                            await asyncio.sleep(1)
                        except:
                            pass
                    else:
                        logger.error(f"Failed to resolve conflict after {max_retries} attempts")
                        logger.warning("Please:")
                        logger.warning("1. Stop all other bot instances")
                        logger.warning("2. Wait 20-30 seconds")
                        logger.warning("3. Run 'python check_bot_instances.py' to verify")
                        logger.warning("4. Restart the bot")
                        raise Exception("Telegram bot conflict could not be resolved. Please check for other running instances.")
                else:
                    logger.warning(f"Unexpected error during conflict check: {e}")
                    break
        
        # اضافه کردن error handler برای conflict
        async def error_handler(update: object, context: Any) -> None:
            error = context.error
            if isinstance(error, Conflict):
                logger.error("Conflict detected during polling. This usually means another bot instance is running.")
                logger.warning("The bot will continue trying to reconnect. Please stop other instances.")
            else:
                logger.error(f"Unhandled error in Telegram bot: {error}")
        
        self.application.add_error_handler(error_handler)
        
        # صبر کوتاه قبل از شروع polling
        await asyncio.sleep(2)
        
        # شروع polling
        try:
            await self.application.updater.start_polling(
                drop_pending_updates=True,
                allowed_updates=["message", "callback_query"],
                poll_interval=1.0,
                timeout=10
            )
            logger.info("Telegram bot started successfully")
        except Exception as e:
            if "Conflict" in str(e) or isinstance(e, Conflict):
                logger.error("Conflict detected during polling startup.")
                logger.warning("Please stop all other bot instances and wait 20-30 seconds before restarting.")
            logger.error(f"Error starting polling: {e}")
            raise
    
    async def stop(self):
        """توقف ربات تلگرام"""
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
    
    async def start_command(self, update: 'Update', context: Any):
        """دستور /start"""
        if update.message:
            self.chat_ids.add(update.message.chat_id)
        
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
        
        def timer_callback():
            try:
                if self.event_loop and self.event_loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.default_strategy_selection(update),
                        self.event_loop
                    )
                else:
                    logger.warning("Event loop not available for timer callback")
            except Exception as e:
                logger.error(f"Error in timer callback: {e}")
        
        self.selection_timer = threading.Timer(
            self.selection_timeout,
            timer_callback
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
        if query is None:
            logger.error("Callback query is None")
            return
        
        await query.answer()
        
        data = query.data
        if data is None:
            logger.error("Callback data is None")
            return
        
        logger.info(f"Button callback received: {data}")
        
        if data.startswith("strategy_"):
            strategy_name = data.replace("strategy_", "")
            logger.info(f"Strategy selected: {strategy_name}")
            
            if strategy_name == "DAY_TRADING":
                self.selected_strategy = StrategyType.DAY_TRADING
            elif strategy_name == "SCALP":
                self.selected_strategy = StrategyType.SCALP
            elif strategy_name == "SUPER_SCALP":
                self.selected_strategy = StrategyType.SUPER_SCALP
            else:
                logger.error(f"Unknown strategy: {strategy_name}")
                await query.edit_message_text(f"❌ استراتژی نامعتبر: {strategy_name}")
                return
            
            if self.selection_timer:
                self.selection_timer.cancel()
            
            try:
                await query.edit_message_text(
                    f"✅ استراتژی: {self.selected_strategy.value}\n\n"
                    "در حال نمایش منوی انتخاب نماد..."
                )
                await self.show_symbol_menu(query)
            except Exception as e:
                logger.error(f"Error in strategy selection: {e}", exc_info=True)
                await query.edit_message_text(f"❌ خطا در انتخاب استراتژی: {e}")
        
        elif data.startswith("symbol_"):
            symbol_name = data.replace("symbol_", "")
            logger.info(f"Symbol selected: {symbol_name}")
            
            try:
                self.selected_symbol = SymbolType[symbol_name]
            except KeyError:
                logger.error(f"Unknown symbol: {symbol_name}")
                await query.edit_message_text(f"❌ نماد نامعتبر: {symbol_name}")
                return
            
            if self.selected_strategy is None:
                logger.error("Strategy not selected before symbol selection")
                await query.edit_message_text("❌ لطفاً ابتدا استراتژی را انتخاب کنید.")
                return
            
            if self.selection_timer:
                self.selection_timer.cancel()
            
            try:
                await query.edit_message_text(
                    f"✅ استراتژی: {self.selected_strategy.value}\n"
                    f"✅ نماد: {self.selected_symbol.value}\n\n"
                    "🚀 در حال راه‌اندازی ربات..."
                )
                
                await self.main_controller.start_trading(
                    self.selected_symbol,
                    self.selected_strategy
                )
                
                if query.message:
                    self.chat_ids.add(query.message.chat_id)
                    await self.send_status_message(query.message.chat_id, is_start=True)
            except Exception as e:
                logger.error(f"Error starting trading: {e}", exc_info=True)
                await query.edit_message_text(f"❌ خطا در راه‌اندازی ربات: {e}")
        
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
            chat_id = query.message.chat_id if query.message else None
            await self.main_controller.stop()
            if chat_id:
                await self.send_status_message(chat_id, is_start=False)
        
        elif data == "stop_cancel":
            await query.edit_message_text("✅ توقف لغو شد.")
    
    async def show_symbol_menu(self, query_or_update):
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
        
        if hasattr(query_or_update, 'edit_message_text'):
            await query_or_update.edit_message_text(
                f"✅ استراتژی: {self.selected_strategy.value}\n\n"
                "لطفاً نماد معاملاتی را انتخاب کنید:\n"
                f"⏱️ زمان باقیمانده: {self.selection_timeout} ثانیه\n"
                "در صورت عدم انتخاب، BTCUSD به عنوان پیش‌فرض انتخاب می‌شود.",
                reply_markup=reply_markup
            )
        elif hasattr(query_or_update, 'message'):
            await query_or_update.message.reply_text(
                f"✅ استراتژی: {self.selected_strategy.value}\n\n"
                "لطفاً نماد معاملاتی را انتخاب کنید:\n"
                f"⏱️ زمان باقیمانده: {self.selection_timeout} ثانیه\n"
                "در صورت عدم انتخاب، BTCUSD به عنوان پیش‌فرض انتخاب می‌شود.",
                reply_markup=reply_markup
            )
        
        def timer_callback():
            try:
                if self.event_loop and self.event_loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.default_symbol_selection(query_or_update),
                        self.event_loop
                    )
                else:
                    logger.warning("Event loop not available for timer callback")
            except Exception as e:
                logger.error(f"Error in timer callback: {e}")
        
        self.selection_timer = threading.Timer(
            self.selection_timeout,
            timer_callback
        )
        self.selection_timer.start()
    
    async def default_symbol_selection(self, query_or_update):
        """انتخاب نماد پیش‌فرض در صورت عدم انتخاب"""
        if not self.selected_symbol:
            self.selected_symbol = SymbolType.BTCUSD
        
        message_text = (
            f"✅ استراتژی: {self.selected_strategy.value}\n"
            f"✅ نماد: {self.selected_symbol.value} (پیش‌فرض)\n\n"
            "🚀 در حال راه‌اندازی ربات..."
        )
        
        if hasattr(query_or_update, 'edit_message_text'):
            await query_or_update.edit_message_text(message_text)
        elif hasattr(query_or_update, 'message'):
            await query_or_update.message.reply_text(message_text)
        
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
• معامله باز: {'✅ بله' if self.main_controller.order_executor.has_open_position() else '❌ خیر'}
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
    
    async def add_channel_command(self, update: 'Update', context: Any):
        """دستور /add_channel - اضافه کردن کانال برای دریافت اعلان‌ها"""
        if not update.message:
            return
        
        chat_id = update.message.chat_id
        self.chat_ids.add(chat_id)
        
        chat_type = "channel" if update.message.chat.type == "channel" else "group" if update.message.chat.type == "group" else "private"
        
        await update.message.reply_text(
            f"✅ Chat ID اضافه شد:\n"
            f"• Chat ID: `{chat_id}`\n"
            f"• Type: {chat_type}\n"
            f"• Title: {update.message.chat.title if hasattr(update.message.chat, 'title') else 'N/A'}\n\n"
            f"از این به بعد تمام اعلان‌های ربات به این chat ارسال می‌شود.",
            parse_mode='Markdown'
        )
        
        logger.info(f"Chat ID added: {chat_id} (Type: {chat_type})")
    
    async def get_chat_id_command(self, update: 'Update', context: Any):
        """دستور /get_chat_id - دریافت Chat ID"""
        if not update.message:
            return
        
        chat_id = update.message.chat_id
        chat_type = update.message.chat.type if hasattr(update.message.chat, 'type') else "unknown"
        chat_title = update.message.chat.title if hasattr(update.message.chat, 'title') else 'N/A'
        chat_username = update.message.chat.username if hasattr(update.message.chat, 'username') else 'N/A'
        
        message = f"""📋 <b>Chat Information:</b>

• <b>Chat ID:</b> <code>{chat_id}</code>
• <b>Type:</b> {chat_type}
• <b>Title:</b> {chat_title}
• <b>Username:</b> @{chat_username if chat_username != 'N/A' else 'N/A'}

💡 <b>برای اضافه کردن این chat به لیست اعلان‌ها:</b>
دستور <code>/add_channel</code> را ارسال کنید.

📝 <b>برای استفاده در .env:</b>
<code>TELEGRAM_CHAT_ID={chat_id}</code>"""
        
        await update.message.reply_text(message, parse_mode='HTML')
        
        logger.info(f"Chat ID requested: {chat_id} (Type: {chat_type})")
    
    async def send_message(self, chat_id: int, text: str):
        """ارسال پیام به کاربر"""
        if self.application:
            await self.application.bot.send_message(chat_id=chat_id, text=text)
    
    async def send_notification(self, message: str, parse_mode: str = 'HTML'):
        """ارسال اعلان به همه کاربران و کانال‌ها"""
        if not self.application or not self.chat_ids:
            logger.warning("No chat_ids registered. Use /add_channel command or set TELEGRAM_CHAT_ID in .env")
            return
        
        success_count = 0
        for chat_id in self.chat_ids:
            try:
                await self.application.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode=parse_mode
                )
                success_count += 1
                logger.debug(f"Notification sent successfully to chat_id: {chat_id}")
            except Exception as e:
                error_msg = str(e)
                if "chat not found" in error_msg.lower() or "bot was blocked" in error_msg.lower():
                    logger.warning(f"Chat {chat_id} not accessible. Removing from list. Error: {e}")
                    self.chat_ids.discard(chat_id)
                else:
                    logger.error(f"Error sending notification to {chat_id}: {e}")
        
        if success_count > 0:
            logger.info(f"Notification sent to {success_count} chat(s)")
        else:
            logger.warning("No notifications were sent successfully")
    
    async def send_status_message(self, chat_id: int, is_start: bool = True):
        """ارسال پیام وضعیت با اطلاعات حساب و معاملات"""
        if not self.application:
            return
        
        try:
            controller = self.main_controller
            account_info = controller.conn_mgr.get_account_info()
            if account_info is None:
                return
            
            cursor = controller.db_manager.conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'OPEN'")
            open_orders = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'CLOSED'")
            closed_orders = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(profit) FROM trades WHERE status = 'CLOSED'")
            total_profit_result = cursor.fetchone()[0]
            total_profit = total_profit_result if total_profit_result else 0.0
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'CLOSED' AND profit > 0")
            winning_orders = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'CLOSED' AND profit < 0")
            losing_orders = cursor.fetchone()[0]
            
            win_rate = (winning_orders / closed_orders * 100) if closed_orders > 0 else 0.0
            
            if is_start:
                status_emoji = "🟢"
                status_text = "Bot Started"
            else:
                status_emoji = "🔴"
                status_text = "Bot Stopped"
            
            message = f"""{status_emoji} <b>{status_text}</b>

💰 <b>Account Information:</b>
• Login: {account_info.login}
• Balance: ${account_info.balance:.2f}
• Equity: ${account_info.equity:.2f}
• Margin: ${account_info.margin:.2f}
• Free Margin: ${getattr(account_info, 'free_margin', account_info.equity - account_info.margin):.2f}
• Margin Level: {account_info.margin_level:.2f}%

📊 <b>Order Statistics:</b>
• Open Orders: {open_orders}
• Closed Orders: {closed_orders}
• Winning Orders: {winning_orders}
• Losing Orders: {losing_orders}
• Win Rate: {win_rate:.2f}%

💵 <b>Performance:</b>
• Total Profit/Loss: ${total_profit:.2f}
• Current Balance: ${account_info.balance:.2f}
• Equity: ${account_info.equity:.2f}"""
            
            if controller.current_strategy and controller.current_symbol:
                message += f"""

📈 <b>Current Trading:</b>
• Symbol: {controller.current_symbol.value}
• Strategy: {controller.current_strategy.value}"""
            
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Error sending status message: {e}")

