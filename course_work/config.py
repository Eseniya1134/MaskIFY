from pathlib import Path

SEED = 42

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

CLASSES = ["WithMask", "WithoutMask"]

IMG_SIZE = 128
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

MODELS_DIR = BASE_DIR / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)
