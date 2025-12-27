"""
Script to fix Telegram bot conflict by deleting webhook
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def delete_webhook():
    try:
        from telegram_bot._telegram_import import TELEGRAM_AVAILABLE, Application
        
        if not TELEGRAM_AVAILABLE:
            print("python-telegram-bot is not installed")
            return
        
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not token:
            print("TELEGRAM_BOT_TOKEN not found in .env file")
            return
        
        app = Application.builder().token(token).build()
        await app.initialize()
        
        result = await app.bot.delete_webhook(drop_pending_updates=True)
        print(f"Webhook deleted: {result}")
        
        await app.shutdown()
        print("Done!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(delete_webhook())

