"""
Script to check if there are other bot instances running
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def check_bot_status():
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
        
        webhook_info = await app.bot.get_webhook_info()
        print(f"Webhook URL: {webhook_info.url}")
        print(f"Webhook pending updates: {webhook_info.pending_update_count}")
        
        if webhook_info.url:
            print("\nWARNING: A webhook is configured!")
            print("Deleting webhook...")
            result = await app.bot.delete_webhook(drop_pending_updates=True)
            print(f"Webhook deleted: {result}")
        else:
            print("\nOK: No webhook configured")
        
        try:
            updates = await app.bot.get_updates(limit=1, timeout=1)
            print(f"\nOK: Bot is ready. No conflicts detected.")
            print(f"Pending updates: {len(updates)}")
        except Exception as e:
            if "Conflict" in str(e):
                print(f"\nERROR: CONFLICT DETECTED: {e}")
                print("\nPlease:")
                print("1. Stop all other bot instances")
                print("2. Wait 10-20 seconds")
                print("3. Run this script again")
            else:
                print(f"\nWARNING: Error getting updates: {e}")
        
        await app.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_bot_status())

