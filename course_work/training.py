from tensorflow.keras import callbacks, optimizers
from config import MODELS_DIR, LEARNING_RATE

def get_callbacks(name):
    return [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(MODELS_DIR / f"{name}.keras", save_best_only=True)
    ]

def compile_and_train(model, train, val, epochs, name):
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    history = model.fit(
        train,
        validation_data=val,
        epochs=epochs,
        callbacks=get_callbacks(name)
    )
    return history
