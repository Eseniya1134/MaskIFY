# handlers.py
from telegram import Update
from telegram.ext import ContextTypes

# ===== Командные обработчики =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Вот список команд...")

async def model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Выберите модель...")

async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(f"Вы выбрали: {query.data}")

async def test_dataset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Тест датасета выполнен!")

# ===== Обработчики медиа =====
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Фото получено!")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Документ получен!")