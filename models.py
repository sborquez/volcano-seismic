import tensorflow as tf


def get_loss(name, parameters, **kwargs):
    if name == "categorical_crossentropy":
        return tf.keras.losses.CategoricalCrossentropy()
    else:
        raise ValueError(f"Invalid loss function {name}")

def get_optimizer(name, parameters, **kwargs):
    if name == "adam":
        return tf.keras.optimizers.Adam(**parameters)
    elif name == "RMSprop":
        return tf.keras.optimizers.RMSprop(**parameters)
    else:
        raise ValueError(f"Invalid optimizer {name}")


def mlp_builder(input_shape, num_classes, layers_units=[256, 128], **kwargs):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(inputs)
    for i, l_units in enumerate(layers_units):
        x = tf.keras.layers.Dense(units=l_units, activation="relu")(x)
    classification_head = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=classification_head)
    return model


builders = {
    "mlp_builder": mlp_builder
}

__all__ = [
    "get_loss", "get_optimizer",
    "mlp_builder", "builders"
]