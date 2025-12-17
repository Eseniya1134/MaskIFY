from tensorflow.keras import layers, models, applications

def build_custom_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_vgg_like(input_shape):
    model = models.Sequential([
        layers.Conv2D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_mobilenet(input_shape):
    base = applications.MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(base.input, out), base

def build_resnet(input_shape):
    base = applications.ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(base.input, out)
