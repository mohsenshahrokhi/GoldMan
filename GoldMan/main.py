import os
import sys
import asyncio
from pathlib import Path

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger, logger
from core.bot import GoldManBot
from config.enums import StrategyType, SymbolType


def create_env_file_if_not_exists():
    env_path = Path('.env')
    env_example_path = Path('.env.example')
    
    if env_path.exists():
        return
    
    logger.info("Creating .env file from template...")
    
    env_content = """# GoldMan Trading Bot Configuration

TELEGRAM_BOT_TOKEN=

TELEGRAM_CHAT_ID=

MT5_LOGIN=
MT5_PASSWORD=
MT5_SERVER=
"""
    
    try:
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        logger.info(".env file created successfully. Please fill in your credentials.")
    except Exception as e:
        logger.error(f"Error creating .env file: {e}")
    
    if env_example_path.exists():
        return
    
    try:
        with open(env_example_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        logger.info(".env.example file created successfully.")
    except Exception as e:
        logger.warning(f"Error creating .env.example file: {e}")


def load_environment():
    create_env_file_if_not_exists()
    
    if DOTENV_AVAILABLE:
        env_path = Path('.env')
        if env_path.exists():
            load_dotenv(env_path)
            logger.info("Environment variables loaded from .env file")
        else:
            logger.warning(".env file not found. Using system environment variables.")
    else:
        logger.warning("python-dotenv not installed. Install it with: pip install python-dotenv")
        logger.info("Using system environment variables only.")


def get_strategy_input():
    try:
        return input().strip()
    except (EOFError, KeyboardInterrupt, OSError):
        return ""

def get_symbol_input():
    try:
        return input().strip()
    except (EOFError, KeyboardInterrupt, OSError):
        return ""

async def get_user_strategy_selection() -> StrategyType:
    print("\n" + "="*50)
    print("Select Trading Strategy:")
    print("1. Day Trading")
    print("2. Scalp")
    print("3. Super Scalp")
    print("="*50)
    print("You have 10 seconds to select. Default: Super Scalp")
    print("Enter choice (1-3): ", end="", flush=True)
    
    try:
        selection = await asyncio.wait_for(
            asyncio.to_thread(get_strategy_input),
            timeout=10.0
        )
        
        if selection == "1":
            return StrategyType.DAY_TRADING
        elif selection == "2":
            return StrategyType.SCALP
        elif selection == "3":
            return StrategyType.SUPER_SCALP
        else:
            logger.info("Invalid selection. Using default: Super Scalp")
            return StrategyType.SUPER_SCALP
    except asyncio.TimeoutError:
        print("\nTimeout - Using default: Super Scalp")
        logger.info("Timeout - Using default strategy: Super Scalp")
        return StrategyType.SUPER_SCALP
    except Exception as e:
        logger.warning(f"Error getting strategy selection: {e}. Using default: Super Scalp")
        return StrategyType.SUPER_SCALP


async def get_console_selection(bot: GoldManBot):
    """دریافت انتخاب از console با timeout"""
    try:
        logger.info("Console input available. You can select strategy and symbol now.")
        selected_strategy = await get_user_strategy_selection()
        logger.info(f"Selected strategy: {selected_strategy.value}")
        
        selected_symbol = await get_user_symbol_selection(bot)
        logger.info(f"Selected symbol: {selected_symbol.value}")
        
        return (selected_strategy, selected_symbol)
    except asyncio.TimeoutError:
        logger.info("Console input timeout. Waiting for Telegram selection...")
        return None
    except Exception as e:
        logger.warning(f"Error in console selection: {e}")
        return None

async def get_user_symbol_selection(bot: GoldManBot) -> SymbolType:
    print("\n" + "="*50)
    print("Select Trading Symbol:")
    print("1. XAUUSD (Gold)")
    print("2. EURUSD")
    print("3. YM (Dow Jones)")
    print("4. BTCUSD (Bitcoin)")
    print("="*50)
    print("You have 10 seconds to select. Default: BTCUSD")
    print("Enter choice (1-4): ", end="", flush=True)
    
    try:
        selection = await asyncio.wait_for(
            asyncio.to_thread(get_symbol_input),
            timeout=10.0
        )
        
        if selection == "1":
            return SymbolType.XAUUSD
        elif selection == "2":
            return SymbolType.EURUSD
        elif selection == "3":
            return SymbolType.YM
        elif selection == "4":
            return SymbolType.BTCUSD
        else:
            logger.info("Invalid selection. Using default: BTCUSD")
            return SymbolType.BTCUSD
    except asyncio.TimeoutError:
        print("\nTimeout - Using default: BTCUSD")
        logger.info("Timeout - Using default symbol: BTCUSD")
        return SymbolType.BTCUSD
    except Exception as e:
        logger.warning(f"Error getting symbol selection: {e}. Using default: BTCUSD")
        return SymbolType.BTCUSD


async def main():
    setup_logger()
    
    load_environment()
    
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_token = telegram_token.strip() if telegram_token and telegram_token.strip() else None
    
    mt5_login = os.getenv('MT5_LOGIN')
    mt5_login = int(mt5_login) if mt5_login and mt5_login.strip() else None
    mt5_password = os.getenv('MT5_PASSWORD')
    mt5_server = os.getenv('MT5_SERVER')
    
    bot = GoldManBot(
        telegram_token=telegram_token,
        mt5_login=mt5_login,
        mt5_password=mt5_password,
        mt5_server=mt5_server
    )
    
    try:
        await bot.initialize()
        
        selected_strategy = None
        selected_symbol = None
        console_task = None
        
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        telegram_chat_id = telegram_chat_id.strip() if telegram_chat_id and telegram_chat_id.strip() else None
        
        logger.info(f"Telegram token status: {'SET' if telegram_token else 'NOT SET'}, Telegram Bot status: {'ACTIVE' if bot.telegram_bot else 'NOT ACTIVE'}")
        logger.info(f"Telegram chat_id status: {'SET' if telegram_chat_id else 'NOT SET'}")
        
        if telegram_token and bot.telegram_bot:
            logger.info("Telegram Bot is active. You can select via Telegram OR console.")
            logger.info("Send /start command to your Telegram bot OR use console input below.")
            logger.info("Console input will timeout in 10 seconds if not used.")
            
            console_task = None
            try:
                console_task = asyncio.create_task(get_console_selection(bot))
                console_result = await console_task
            except asyncio.CancelledError:
                logger.info("Console selection cancelled")
                if console_task and not console_task.done():
                    console_task.cancel()
                    try:
                        await console_task
                    except asyncio.CancelledError:
                        pass
                console_result = None
            
            if console_result:
                selected_strategy, selected_symbol = console_result
                logger.info("Selection completed via Console")
                await bot.start_operating(selected_symbol, selected_strategy)
                logger.info("Bot started successfully!")
                
                try:
                    while True:
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logger.info("Stop signal received...")
            else:
                if telegram_chat_id:
                    logger.info("Console input timeout. Using default values: Super Scalp + BTCUSD")
                    logger.info("Starting bot automatically with default settings...")
                    selected_strategy = StrategyType.SUPER_SCALP
                    selected_symbol = SymbolType.BTCUSD
                    await bot.start_operating(selected_symbol, selected_strategy)
                    logger.info("Bot started successfully with default settings!")
                    
                    try:
                        while True:
                            await asyncio.sleep(1)
                    except asyncio.CancelledError:
                        logger.info("Stop signal received...")
                else:
                    logger.info("Console input timeout. Waiting for Telegram selection...")
                    logger.info("Send /start command to your Telegram bot to begin.")
                    try:
                        while True:
                            await asyncio.sleep(1)
                            if bot.is_running():
                                logger.info("Bot started via Telegram!")
                                break
                    except asyncio.CancelledError:
                        logger.info("Stop signal received...")
        else:
            if telegram_token:
                logger.warning("Telegram token provided but Telegram Bot failed to initialize. Using console input instead.")
            else:
                logger.info("Telegram token not set. Using console input for selection.")
            
            selected_strategy = await get_user_strategy_selection()
            logger.info(f"Selected strategy: {selected_strategy.value}")
            
            selected_symbol = await get_user_symbol_selection(bot)
            logger.info(f"Selected symbol: {selected_symbol.value}")
            
            await bot.start_trading(selected_symbol, selected_strategy)
            logger.info("Bot started successfully!")
            
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.info("Stop signal received...")
    
    except KeyboardInterrupt:
        logger.info("Stop signal received (KeyboardInterrupt)...")
    except asyncio.CancelledError:
        logger.info("Stop signal received (CancelledError)...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        try:
            await bot.stop()
        except Exception as e:
            logger.error(f"Error during bot shutdown: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
