import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
import pickle


# ============================================================
# КЛАСС HOG + SVM
# ============================================================

class HOG_SVM_Model:
    def __init__(self):
        self.model = SVC(kernel='rbf', probability=True, random_state=42)
        self.img_size = (128, 128)

    def extract_features(self, image):
        """Извлечение HOG признаков"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.img_size)
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

    def evaluate(self, X_test, y_test):
        """Оценка модели"""
        print("Оценка модели...")
        predictions = []
        for img in X_test:
            pred, _ = self.predict(img)
            predictions.append(pred)

        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy


# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_dataset(dataset_path, img_size=(128, 128), max_samples=2000):
    """Загрузка датасета"""
    X, y = [], []

    class_names = ['WithoutMask', 'WithMask']

    for label, class_name in enumerate(class_names):
        class_path = dataset_path / "Train" / class_name

        if not os.path.exists(class_path):
            print(f" Папка не найдена: {class_path}")
            continue

        images = os.listdir(str(class_path))

        if max_samples:
            images = images[:max_samples]

        print(f"Загрузка {class_name}: {len(images)} изображений")

        for i, img_name in enumerate(images):
            if i % 200 == 0:
                print(f"  Загружено: {i}/{len(images)}")

            img_path = os.path.join(class_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    X.append(img)
                    y.append(label)
            except Exception as e:
                continue

    print(f"Всего загружено: {len(X)} изображений")
    return np.array(X), np.array(y)


# ============================================================
# ОБУЧЕНИЕ
# ============================================================

def train_hog_svm():
    """Обучение HOG + SVM модели"""

    print("=" * 50)
    print("ОБУЧЕНИЕ HOG + SVM")
    print("=" * 50)

    # Параметры
    BASE_DIR = Path(__file__).resolve().parent.parent
    dataset_path = BASE_DIR / "Face Mask Dataset"
    img_size = (128, 128)
    max_samples = 2000

    print("\n1. Загрузка данных...")
    X, y = load_dataset(dataset_path, img_size=img_size, max_samples=max_samples)
    print(f"Распределение классов: {np.bincount(y)}")

    print("\n2. Разделение на train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} изображений")
    print(f"Test: {len(X_test)} изображений")

    print("\n3. Обучение модели...")
    model = HOG_SVM_Model()
    model.train(X_train, y_train)

    print("\n4. Оценка модели...")
    accuracy = model.evaluate(X_test, y_test)

    print("\n5. Сохранение модели...")
    os.makedirs('models', exist_ok=True)
    save_path = 'models/hog_svm_model.pkl'

    with open(save_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"✓ Модель сохранена: {save_path}")
    print(f"✓ Итоговая точность: {accuracy:.4f}")

    print("\n" + "=" * 50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 50)

    return model, accuracy


# ============================================================
# ТЕСТИРОВАНИЕ ЗАГРУЗКИ
# ============================================================

def test_model_loading():
    """Тест загрузки модели"""
    print("\n" + "=" * 50)
    print("ТЕСТ ЗАГРУЗКИ МОДЕЛИ")
    print("=" * 50)

    try:
        with open('models/hog_svm_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        print("✓ Модель успешно загружена!")
        print(f"✓ Тип модели: {type(loaded_model)}")
        print(f"✓ Размер изображения: {loaded_model.img_size}")

        test_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        pred, prob = loaded_model.predict(test_img)
        print(f"✓ Тестовое предсказание работает!")
        print(f"  Предсказание: {pred}")
        print(f"  Вероятности: {prob}")

        return True

    except Exception as e:
        print(f"✗ Ошибка загрузки: {e}")
        return False


# ============================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================

def main():
    """Главная функция"""

    model, accuracy = train_hog_svm()

    test_model_loading()

    print("\n ИТОГИ:")
    print(f"  Модель обучена с точностью: {accuracy:.2%}")
    print(f"  Модель сохранена в: models/hog_svm_model.pkl")
    print(f"  Модель готова к использованию в боте!")


if __name__ == "__main__":
    main()