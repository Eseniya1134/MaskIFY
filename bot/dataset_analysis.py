import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =======================
# Пути
# =======================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "Face Mask Dataset"


def analyze_dataset(dataset_path: Path):
    """Анализ датасета: количество, размеры, битые файлы"""

    stats = {}

    for class_name in ["WithMask", "WithoutMask"]:
        class_path = dataset_path / "Train" / class_name

        if not class_path.exists():
            print(f" Папка не найдена: {class_path}")
            continue

        images = list(class_path.glob("*"))
        print(f"\n{class_name}: {len(images)} изображений")

        widths, heights = [], []
        corrupted = 0

        for img_path in images[:500]:
            img = cv2.imread(str(img_path))
            if img is None:
                corrupted += 1
                continue

            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)

        stats[class_name] = {
            "count": len(images),
            "avg_width": np.mean(widths) if widths else 0,
            "avg_height": np.mean(heights) if heights else 0,
            "min_width": np.min(widths) if widths else 0,
            "max_width": np.max(widths) if widths else 0,
            "min_height": np.min(heights) if heights else 0,
            "max_height": np.max(heights) if heights else 0,
            "corrupted": corrupted,
        }

    return stats


def visualize_samples(dataset_path: Path, n_samples=5):
    """Визуализация примеров изображений"""

    fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))

    for i, class_name in enumerate(["WithMask", "WithoutMask"]):
        class_path = dataset_path / "Train" / class_name
        images = list(class_path.glob("*"))[:n_samples]

        for j, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            axes[i, j].set_title(class_name if j == 0 else "")

    plt.tight_layout()
    plt.savefig("dataset_samples.png", dpi=150)
    print("Примеры изображений сохранены в dataset_samples.png")


def check_class_balance(stats: dict):
    """Проверка баланса классов"""

    total = sum(info["count"] for info in stats.values())

    print("\n" + "=" * 50)
    print("БАЛАНС КЛАССОВ")
    print("=" * 50)

    classes = []
    counts = []

    for class_name, info in stats.items():
        percentage = (info["count"] / total) * 100
        print(f"{class_name}: {info['count']} ({percentage:.1f}%)")
        classes.append(class_name)
        counts.append(info["count"])

    plt.figure(figsize=(6, 5))
    plt.bar(classes, counts)
    plt.title("Распределение классов")
    plt.ylabel("Количество изображений")
    plt.savefig("class_balance.png", dpi=150)
    print("График сохранён в class_balance.png")


def print_detailed_stats(stats: dict):
    """Детальная статистика"""

    print("\n" + "=" * 50)
    print("ДЕТАЛЬНАЯ СТАТИСТИКА")
    print("=" * 50)

    for class_name, info in stats.items():
        print(f"\n{class_name}:")
        print(f"  Количество: {info['count']}")
        print(f"  Средний размер: {info['avg_width']:.0f}x{info['avg_height']:.0f}")
        print(f"  Мин размер: {info['min_width']}x{info['min_height']}")
        print(f"  Макс размер: {info['max_width']}x{info['max_height']}")
        print(f"  Поврежденных файлов: {info['corrupted']}")


def main():
    print("=" * 50)
    print("АНАЛИЗ ДАТАСЕТА FACE MASK")
    print("=" * 50)

    stats = analyze_dataset(DATASET_PATH)

    if not stats:
        print("Данные не найдены. Проверь путь к датасету.")
        return

    print_detailed_stats(stats)
    check_class_balance(stats)
    visualize_samples(DATASET_PATH)

    print("\nВЫВОДЫ:")
    print("1. Датасет содержит 2 класса")
    print("2. Баланс классов близок к равномерному")
    print("3. Размеры изображений различаются → нужна нормализация")
    print("4. Рекомендуемый размер: 128×128 или 224×224")


if __name__ == "__main__":
    main()
