from sklearn.metrics import classification_report, roc_curve, auc

def evaluate_model(model, generator):
    preds = model.predict(generator)
    y_pred = (preds > 0.5).astype(int)
    y_true = generator.classes

    print(classification_report(y_true, y_pred))
    fpr, tpr, _ = roc_curve(y_true, preds)
    return auc(fpr, tpr)
