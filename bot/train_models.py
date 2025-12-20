import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import hog
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# МОДЕЛЬ 1: Классический метод - HOG + SVM
# ============================================================

class HOG_SVM_Model:
    def __init__(self):
        self.model = SVC(kernel='rbf', probability=True, random_state=42)
        self.img_size = (128, 128)

    def extract_features(self, image):
        """Извлечение HOG признаков"""
        # Преобразование в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Изменение размера
        resized = cv2.resize(gray, self.img_size)
        # HOG признаки
        features = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False)
        return features

    def train(self, X_train, y_train):
        """Обучение SVM"""
        print("Извлечение HOG признаков...")
        X_features = [self.extract_features(img) for img in X_train]
        print(f"Размер признаков: {X_features[0].shape}")

        print("Обучение SVM...")
        self.model.fit(X_features, y_train)
        print("Обучение завершено!")

    def predict(self, image):
        """Предсказание для одного изображения"""
        features = self.extract_features(image).reshape(1, -1)
        pred = self.model.predict(features)[0]
        prob = self.model.predict_proba(features)[0]
        return pred, prob

    def save(self, path='models/hog_svm_model.pkl'):
        """Сохранение модели"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path='models/hog_svm_model.pkl'):
        """Загрузка модели"""
        with open(path, 'rb') as f:
            return pickle.load(f)


# ============================================================
# МОДЕЛЬ 2: CNN с нуля
# ============================================================

def create_simple_cnn(input_shape=(128, 128, 3), num_classes=2):
    """Создание простой CNN"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================
# МОДЕЛЬ 3: Transfer Learning - MobileNetV2
# ============================================================

def create_mobilenet_model(input_shape=(128, 128, 3), num_classes=2):
    """Transfer Learning с MobileNetV2"""
    # Загрузка предобученной модели
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Заморозка базовых слоев
    base_model.trainable = False

    # Добавление собственных слоев
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================

def load_dataset(dataset_path, img_size=(128, 128), max_samples=None):
    """Загрузка датасета"""
    X, y = [], []

    class_names = ['WithoutMask', 'WithMask']

    for label, class_name in enumerate(class_names):
        class_path = dataset_path / "Train" / class_name
        images = os.listdir(str(class_path))

        if max_samples:
            images = images[:max_samples]

        print(f"Загрузка {class_name}: {len(images)} изображений")

        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    X.append(img)
                    y.append(label)
            except:
                continue

    return np.array(X), np.array(y)


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# ============================================================

def train_all_models():
    """Обучение всех трех моделей"""

    BASE_DIR = Path(__file__).resolve().parent.parent
    dataset_path = BASE_DIR / "Face Mask Dataset"
    img_size = (128, 128)

    print("=" * 50)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 50)
    X, y = load_dataset(dataset_path, img_size=img_size, max_samples=2000)
    print(f"Загружено: {len(X)} изображений")
    print(f"Распределение классов: {np.bincount(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ========== МОДЕЛЬ 1: HOG + SVM ==========
    print("\n" + "=" * 50)
    print("МОДЕЛЬ 1: HOG + SVM")
    print("=" * 50)

    model1 = HOG_SVM_Model()
    model1.train(X_train, y_train)

    correct = 0
    for img, true_label in zip(X_test, y_test):
        pred, _ = model1.predict(img)
        if pred == true_label:
            correct += 1

    accuracy1 = correct / len(X_test)
    print(f"Accuracy: {accuracy1:.4f}")
    model1.save()

    # ========== МОДЕЛЬ 2: Simple CNN ==========
    print("\n" + "=" * 50)
    print("МОДЕЛЬ 2: Simple CNN")
    print("=" * 50)

    X_train_norm = X_train.astype('float32') / 255.0
    X_test_norm = X_test.astype('float32') / 255.0

    y_train_cat = keras.utils.to_categorical(y_train, 2)
    y_test_cat = keras.utils.to_categorical(y_test, 2)

    model2 = create_simple_cnn()

    history2 = model2.fit(
        X_train_norm, y_train_cat,
        validation_data=(X_test_norm, y_test_cat),
        epochs=20,
        batch_size=32,
        verbose=1
    )

    loss2, accuracy2 = model2.evaluate(X_test_norm, y_test_cat, verbose=0)
    print(f"Test Accuracy: {accuracy2:.4f}")
    model2.save('models/simple_cnn_model.h5')

    # ========== МОДЕЛЬ 3: MobileNetV2 ==========
    print("\n" + "=" * 50)
    print("МОДЕЛЬ 3: MobileNetV2 (Transfer Learning)")
    print("=" * 50)

    model3 = create_mobilenet_model()

    history3 = model3.fit(
        X_train_norm, y_train_cat,
        validation_data=(X_test_norm, y_test_cat),
        epochs=15,
        batch_size=32,
        verbose=1
    )

    loss3, accuracy3 = model3.evaluate(X_test_norm, y_test_cat, verbose=0)
    print(f"Test Accuracy: {accuracy3:.4f}")
    model3.save('models/mobilenet_model.h5')

    # ========== СРАВНЕНИЕ МОДЕЛЕЙ ==========
    print("\n" + "=" * 50)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 50)
    print(f"1. HOG + SVM:         {accuracy1:.4f}")
    print(f"2. Simple CNN:        {accuracy2:.4f}")
    print(f"3. MobileNetV2:       {accuracy3:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    train_all_models()