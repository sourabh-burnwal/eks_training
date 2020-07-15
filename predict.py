import tensorflow as tf
import numpy as np

hours_studied    = np.array([10, 12, 20, 17, 25, 35, 31, 26],  dtype=float)
marks = np.array([160, 165, 190, 180, 205, 232, 225, 210],  dtype=float)


layer_0 = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([layer_0])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
trained_model = model.fit(hours_studied, marks, epochs=1000, verbose=False)

def predict(num):
    print(model.predict(num))
