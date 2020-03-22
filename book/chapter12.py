import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1,1), random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

input_shape = X_train.shape[1:]

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",input_shape=input_shape),
    keras.layers.Dense(1)
])
# def huber_fn(y_true, y_pred):
#     error = y_true-y_pred
#     is_small_error = tf.abs(error)<1
#     squared_loss = tf.square(error) / 2
#     linear_loss = tf.abs(error)-0.5
#     return tf.where(is_small_error,squared_loss,linear_loss)

from tensorflow.keras.losses import Huber
huber=Huber()
model.compile(loss=huber,optimizer="nadam",metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid_scaled,y_valid))

def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true-y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error)-threshold**2/2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn

model.compile(loss=create_huber(2.0), optimizer="nadam")

class