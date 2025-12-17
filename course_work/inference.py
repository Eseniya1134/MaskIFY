import cv2
import numpy as np

def predict_mask(image_path, model, size):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size)) / 255.0
    pred = model.predict(img[None])[0][0]
    return "WithoutMask" if pred > 0.5 else "WithMask", float(pred)
