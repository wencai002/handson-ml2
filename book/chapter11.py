import keras
import numpy as np

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds

def exponential_decay_fn(epoch):
    return 0.01*0.1**(epoch/20)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train_scaled, y_train,
                    epochs=25,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[lr_scheduler])

# def split_dataset(X, y):
#     y_5_or_6 = (y == 5) | (y == 6) # sandals or shirts
#     y_A = y[~y_5_or_6]
#     y_A[y_A > 6] = y_A[y_A>6]-2 # class indices 7, 8, 9 should be moved to 5, 6, 7
#     y_B = (y[y_5_or_6] == 6).astype(np.float32) # binary classification task: is it a shirt (class 6)?
#     return ((X[~y_5_or_6], y_A),
#             (X[y_5_or_6], y_B))
#
# (X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
# (X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
# (X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)
# X_train_B = X_train_B[:200]
# y_train_B = y_train_B[:200]

# model_A = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28,28]),
#     keras.layers.Dense(300, activation="selu"),
#     keras.layers.Dense(100, activation="selu"),
#     keras.layers.Dense(50, activation="selu"),
#     keras.layers.Dense(50, activation="selu"),
#     keras.layers.Dense(50, activation="selu"),
#     keras.layers.Dense(8, activation="softmax")
# ])
#
# model_A.compile(loss="sparse_categorical_crossentropy",
#                 optimizer=keras.optimizers.SGD(lr=1e-3),
#                 metrics=["accuracy"])
#
# history = model_A.fit(X_train_A, y_train_A, epochs=20, validation_data=(X_valid_A, y_valid_A))
# model_A.save("modelC11/my_model_A.h5")

# model_A = keras.models.load_model("modelC11/my_model_A.h5")
# model_A_clone = keras.models.clone_model(model_A)
# model_A_clone.set_weights(model_A.get_weights())
# model_B_on_A = keras.models.Sequential(model_A_clone.layers[:-1])
# model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid", name="dense_output_B"))
#
# for layer in model_B_on_A.layers[:-1]:
#     layer.trainable = False
#
# model_B_on_A.compile(loss="binary_crossentropy",
#                      optimizer="sgd",
#                      metrics=["accuracy"])
#
# history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4, validation_data=(X_valid_B, y_valid_B))
# for layer in model_B_on_A.layers[:-1]:
#     layer.trainable = True
#
# optimizer = keras.optimizers.SGD(lr=1e-4)
# model_B_on_A.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
# model_B_on_A.fit(X_train_B, y_train_B, epochs=20, validation_data=(X_valid_B, y_valid_B))
# print(model_B_on_A.evaluate(X_test_B, y_test_B))


#
# print(model.summary())