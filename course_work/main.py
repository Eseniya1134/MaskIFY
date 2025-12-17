from config import IMG_SHAPE, EPOCHS
from data import create_generators
from models import build_custom_cnn, build_mobilenet
from training import compile_and_train
from utils import plot_history

train, val, test = create_generators()

cnn = build_custom_cnn(IMG_SHAPE)
cnn_hist = compile_and_train(cnn, train, val, EPOCHS, "cnn")
plot_history(cnn_hist, "CNN")

mobilenet, _ = build_mobilenet(IMG_SHAPE)
mb_hist = compile_and_train(mobilenet, train, val, EPOCHS, "mobilenet")
plot_history(mb_hist, "MobileNetV2")
