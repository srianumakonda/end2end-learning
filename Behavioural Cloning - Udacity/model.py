import keras
from keras import layers


def model():
    model = keras.Sequential([
        layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu', input_shape=(80, 160, 3)),
        layers.Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(1164, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(50, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1)
    ])
    return model
    
def train(model, x, y, lr, epochs, batch_size, val_pct, save_model, model_name):
    # callback = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-3),
    #             keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)]
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=lr))
    model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=val_pct, shuffle=True, verbose=1)

    if save_model:
        model.save("models/"+model_name)
    return