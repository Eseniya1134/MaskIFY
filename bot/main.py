# main.py
import logging
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from config import BOT_TOKEN
from handlers import (
    start, help_command,
    model_selection, model_callback, test_dataset,
    handle_photo, handle_document
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    app = Application.builder().token(BOT_TOKEN).build()

    # Командные обработчики
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("model", model_selection))
    app.add_handler(CommandHandler("test_dataset", test_dataset))

    # Обработчики медиа
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Callback для inline-кнопок
    app.add_handler(CallbackQueryHandler(model_callback, pattern="^model_"))

    logger.info("Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()
