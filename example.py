import tensorflow as tf
from filter_response_normalization import FilterResponseNormalization

if __name__ == "__main__":
    shape = (32, 32, 3)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(16, 3, input_shape=shape),
            FilterResponseNormalization(),
            tf.keras.layers.Conv2D(32, 3),
            FilterResponseNormalization(),
            tf.keras.layers.Conv2D(64, 3),
            FilterResponseNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.summary()

