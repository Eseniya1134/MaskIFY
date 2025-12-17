import matplotlib.pyplot as plt

def plot_history(history, name):
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title(name)
    plt.legend()
    plt.show()
