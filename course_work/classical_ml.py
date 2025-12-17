import cv2
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from features import extract_hog, extract_lbp, extract_gabor, extract_combined, build_gabor_bank
from config import CLASSES

def load_images(directory, size):
    images, labels = [], []
    for idx, cls in enumerate(CLASSES):
        for img_path in (directory / cls).glob("*.png"):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size))
            images.append(img)
            labels.append(idx)
    return np.array(images), np.array(labels)

def train_svm(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = SVC(kernel="rbf", probability=True)
    model.fit(Xs, y)
    return model, scaler

def train_rf(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=200)
    model.fit(Xs, y)
    return model, scaler

def train_gb(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = GradientBoostingClassifier(n_estimators=150)
    model.fit(Xs, y)
    return model, scaler
