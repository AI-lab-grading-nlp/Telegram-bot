from telegram.ext import ContextTypes, Application

import os


async def callback_minute(context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id='1105758757', text='One message every minute')

application = Application.builder().token(os.getenv('BOT_TOKEN')).build()
job_queue = application.job_queue
job_minute = job_queue.run_repeating(callback_minute, interval=60, first=10)

application.run_polling()
