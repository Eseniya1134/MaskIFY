import os
import cv2
import numpy as np
import pickle
from telegram import ReplyKeyboardMarkup
from io import BytesIO
from PIL import Image

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

import tensorflow as tf
from tensorflow import keras
from skimage.feature import hog


# ============================================================
# КЛАСС HOG + SVM (должен быть определен ДО загрузки модели!)
# ============================================================

class HOG_SVM_Model:
    def __init__(self):
        self.model = None
        self.img_size = (128, 128)

    def extract_features(self, image):
        """Извлечение HOG признаков"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.img_size)
        features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)
        return features

    def predict(self, image):
        """Предсказание для одного изображения"""
        if self.model is None:
            raise ValueError("Модель не загружена!")
        features = self.extract_features(image).reshape(1, -1)
        pred = self.model.predict(features)[0]
        prob = self.model.predict_proba(features)[0]
        return pred, prob


# ============================================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ============================================================

class ModelManager:
    def __init__(self):
        self.models = {}
        self.load_models()
        self.class_names = ['Без маски', 'В маске']

    def load_models(self):
        """Загрузка всех трех моделей"""
        # Модель 1: HOG + SVM
        try:
            with open('models/hog_svm_model.pkl', 'rb') as f:
                self.models['hog_svm'] = pickle.load(f)
            print("✓ HOG + SVM загружена")
        except FileNotFoundError:
            print("✗ Файл HOG + SVM не найден: models/hog_svm_model.pkl")
        except Exception as e:
            print(f"✗ Ошибка загрузки HOG + SVM: {e}")

        # Модель 2: Simple CNN
        try:
            self.models['simple_cnn'] = keras.models.load_model('models/simple_cnn_model.h5')
            print("✓ Simple CNN загружена")
        except FileNotFoundError:
            print("✗ Файл Simple CNN не найден: models/simple_cnn_model.h5")
        except Exception as e:
            print(f"✗ Ошибка загрузки Simple CNN: {e}")

        # Модель 3: MobileNetV2
        try:
            self.models['mobilenet'] = keras.models.load_model('models/mobilenet_model.h5')
            print("✓ MobileNetV2 загружена")
        except FileNotFoundError:
            print("✗ Файл MobileNetV2 не найден: models/mobilenet_model.h5")
        except Exception as e:
            print(f"✗ Ошибка загрузки MobileNetV2: {e}")

    def predict_hog_svm(self, image):
        """Предсказание HOG + SVM"""
        if 'hog_svm' not in self.models:
            raise ValueError("HOG + SVM модель не загружена")

        model = self.models['hog_svm']
        pred, prob = model.predict(image)
        return pred, prob[pred]

    def predict_cnn(self, image, model_name='simple_cnn'):
        """Предсказание CNN моделей"""
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не загружена")

        model = self.models[model_name]

        # Подготовка изображения
        img = cv2.resize(image, (128, 128))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Предсказание
        pred = model.predict(img, verbose=0)
        class_idx = np.argmax(pred[0])
        confidence = pred[0][class_idx]

        return class_idx, confidence

    def predict_all(self, image):
        """Предсказание всеми моделями"""
        results = {}

        if 'hog_svm' in self.models:
            try:
                pred, conf = self.predict_hog_svm(image)
                results['HOG + SVM'] = {
                    'class': self.class_names[pred],
                    'confidence': conf * 100
                }
            except Exception as e:
                print(f"Ошибка HOG + SVM: {e}")

        if 'simple_cnn' in self.models:
            try:
                pred, conf = self.predict_cnn(image, 'simple_cnn')
                results['Simple CNN'] = {
                    'class': self.class_names[pred],
                    'confidence': conf * 100
                }
            except Exception as e:
                print(f"Ошибка Simple CNN: {e}")

        if 'mobilenet' in self.models:
            try:
                pred, conf = self.predict_cnn(image, 'mobilenet')
                results['MobileNetV2'] = {
                    'class': self.class_names[pred],
                    'confidence': conf * 100
                }
            except Exception as e:
                print(f"Ошибка MobileNetV2: {e}")

        return results


# Глобальный менеджер моделей
model_manager = ModelManager()


# ============================================================
# ОБРАБОТЧИКИ КОМАНД БОТА
# ============================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    welcome_text = """
🎭 <b>Добро пожаловать в бот определения масок!</b> 🎭 

Этот бот использует три различные модели машинного обучения для определения, есть ли на лице защитная маска:

🔹 <b>Модель 1:</b> HOG + SVM (классический метод) 🔹
🔹 <b>Модель 2:</b> Simple CNN (сверточная нейросеть) 🔹
🔹 <b>Модель 3:</b> MobileNetV2 (transfer learning) 🔹 

📸 <b>Как использовать:</b> 📸
1. Отправьте фотографию с лицом
2. Выберите модель для анализа
3. Получите результат!

Используйте /help для получения справки.
    """
    await update.message.reply_text(welcome_text, parse_mode='HTML')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """
ℹ️ <b>Справка</b>

<b>Доступные команды:</b>
/start - Начать работу с ботом
/help - Показать эту справку
/about - Информация о моделях

<b>Как пользоваться:</b>
1. Отправьте фото с лицом человека
2. Выберите модель для анализа
3. Бот определит, есть ли маска на лице

<b>Поддерживаемые форматы:</b>
JPG, PNG

<b>Доступные модели:</b>
"""

    # Добавляем информацию о доступных моделях
    if 'hog_svm' in model_manager.models:
        help_text += "✅ HOG + SVM\n"
    else:
        help_text += "❌ HOG + SVM (не загружена)\n"

    if 'simple_cnn' in model_manager.models:
        help_text += "✅ Simple CNN\n"
    else:
        help_text += "❌ Simple CNN (не загружена)\n"

    if 'mobilenet' in model_manager.models:
        help_text += "✅ MobileNetV2\n"
    else:
        help_text += "❌ MobileNetV2 (не загружена)\n"

    await update.message.reply_text(help_text, parse_mode='HTML')


async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Информация о моделях"""
    about_text = """
📊 <b>Информация о моделях</b>

<b>1. HOG + SVM</b>
Классический метод компьютерного зрения:
• HOG (Histogram of Oriented Gradients) - извлечение признаков
• SVM (Support Vector Machine) - классификация
• Быстрая работа, но меньшая точность

<b>2. Simple CNN</b>
Сверточная нейронная сеть:
• 3 сверточных блока
• Batch Normalization и Dropout
• Обучена с нуля на датасете

<b>3. MobileNetV2</b>
Transfer Learning с предобученной моделью:
• Использует веса ImageNet
• Высокая точность
• Оптимизирована для мобильных устройств

<b>Метрики качества:</b>
• Accuracy (точность)
• Confidence (уверенность модели)

<b>Функция потерь:</b>
Categorical Crossentropy (для CNN моделей)
Hinge Loss (для SVM)
    """
    await update.message.reply_text(about_text, parse_mode='HTML')


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка фотографии"""

    try:
        # Получение фото
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        # Преобразование в numpy array
        image = Image.open(BytesIO(photo_bytes))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Сохранение в контексте
        context.user_data['current_image'] = image

        # Создание клавиатуры с доступными моделями
        keyboard = []

        if 'hog_svm' in model_manager.models:
            keyboard.append([InlineKeyboardButton("🔹 HOG + SVM 🔹", callback_data='model_hog_svm')])

        if 'simple_cnn' in model_manager.models:
            keyboard.append([InlineKeyboardButton("🔹 Simple CNN 🔹", callback_data='model_simple_cnn')])

        if 'mobilenet' in model_manager.models:
            keyboard.append([InlineKeyboardButton("🔹 MobileNetV2 🔹", callback_data='model_mobilenet')])

        # Кнопка "Все модели" только если есть хотя бы 2 модели
        if len(model_manager.models) >= 2:
            keyboard.append([InlineKeyboardButton("🎯 Все модели 🎯", callback_data='model_all')])

        if not keyboard:
            await update.message.reply_text("❌ Ни одна модель не загружена! Проверьте файлы в папке models/")
            return

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "✅ Фото получено! Выберите модель для анализа:",
            reply_markup=reply_markup
        )

    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка обработки фото: {str(e)}")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка нажатий кнопок"""
    query = update.callback_query
    await query.answer()

    image = context.user_data.get('current_image')

    if image is None:
        await query.edit_message_text("❌ Сначала отправьте фото!")
        return

    # Показать процесс обработки
    await query.edit_message_text("⏳ Анализирую изображение...")

    result_text = ""

    try:
        # Выбор модели
        if query.data == 'model_hog_svm':
            if 'hog_svm' not in model_manager.models:
                result_text = "❌ HOG + SVM модель не загружена!"
            else:
                pred, conf = model_manager.predict_hog_svm(image)
                result_text = format_result("HOG + SVM", model_manager.class_names[pred], conf * 100)

        elif query.data == 'model_simple_cnn':
            if 'simple_cnn' not in model_manager.models:
                result_text = "❌ Simple CNN модель не загружена!"
            else:
                pred, conf = model_manager.predict_cnn(image, 'simple_cnn')
                result_text = format_result("Simple CNN", model_manager.class_names[pred], conf * 100)

        elif query.data == 'model_mobilenet':
            if 'mobilenet' not in model_manager.models:
                result_text = "❌ MobileNetV2 модель не загружена!"
            else:
                pred, conf = model_manager.predict_cnn(image, 'mobilenet')
                result_text = format_result("MobileNetV2", model_manager.class_names[pred], conf * 100)

        elif query.data == 'model_all':
            results = model_manager.predict_all(image)
            if not results:
                result_text = "❌ Ни одна модель не смогла обработать изображение!"
            else:
                result_text = format_all_results(results)

        else:
            result_text = "❌ Неизвестная команда!"

    except Exception as e:
        result_text = f"❌ Ошибка при обработке: {str(e)}"



    await query.edit_message_text(
        result_text,
        parse_mode='HTML'
    )

def start_keyboard():
    return ReplyKeyboardMarkup(
        [["▶️ Старт"]],
        resize_keyboard=True,
        one_time_keyboard=True
    )


def format_result(model_name, prediction, confidence):
    """Форматирование результата одной модели"""
    emoji = "✅" if "маске" in prediction else "❌"

    return f"""
{emoji} <b>Результат анализа</b>

<b>Модель:</b> {model_name}
<b>Результат:</b> {prediction}
<b>Уверенность:</b> {confidence:.2f}%

{'🎭 Маска обнаружена!' if "маске" in prediction else '⚠️ Маска не обнаружена!'}
    """


def format_all_results(results):
    """Форматирование результатов всех моделей"""
    text = "🎯 <b>Результаты всех моделей</b>\n\n"

    for model_name, result in results.items():
        emoji = "✅" if "маске" in result['class'] else "❌"
        text += f"{emoji} <b>{model_name}</b>\n"
        text += f"   Результат: {result['class']}\n"
        text += f"   Уверенность: {result['confidence']:.2f}%\n\n"

    # Консенсус
    mask_count = sum(1 for r in results.values() if "маске" in r['class'])
    total = len(results)
    text += f"<b>Консенсус:</b> {mask_count}/{total} модел"
    if total == 1:
        text += "ь определила маску"
    elif total in [2, 3, 4]:
        text += "и определили маску"
    else:
        text += "ей определили маску"

    return text


# ============================================================
# ЗАПУСК БОТА
# ============================================================

def main():
    """Запуск бота"""

    # ВАЖНО: Замените на ваш токен от @BotFather
    TOKEN = "не наш токен"

    if TOKEN == "не наш токен":
        print("=" * 50)
        print("ОШИБКА: Не указан токен бота!")
        print("=" * 50)
        print("1. Откройте @BotFather в Telegram")
        print("2. Создайте бота командой /newbot")
        print("3. Скопируйте токен")
        print("4. Вставьте токен в переменную TOKEN в этом файле")
        print("=" * 50)
        return

    # Проверка наличия хотя бы одной модели
    if not model_manager.models:
        print("=" * 50)
        print("ПРЕДУПРЕЖДЕНИЕ: Ни одна модель не загружена!")
        print("=" * 50)
        print("Убедитесь, что файлы моделей находятся в папке models/:")
        print("  - models/hog_svm_model.pkl")
        print("  - models/simple_cnn_model.h5")
        print("  - models/mobilenet_model.h5")
        print("=" * 50)
        print("Бот будет запущен, но не сможет обрабатывать фото!")
        print("=" * 50)

    # Создание приложения
    application = Application.builder().token(TOKEN).build()

    # Регистрация обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("about", about))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(CallbackQueryHandler(button_callback))

    # Запуск бота
    print("🤖 Бот запущен!")
    print(f"Загружено моделей: {len(model_manager.models)}")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()