import tensorflow as tf
import numpy as np
import wandb

# y_true = tf.keras.utils.to_categorical(np.array([0,1,2]), num_classes=3)
# y_pred = np.array([[1.0, 0, 0], [1.0,0, 0], [0.0, 0.2,0.8]])
# alpha=0.33
# get_loss("categorical_focal_loss")(alpha, gamma=2.)(y_true, y_pred)
def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = np.array(alpha, dtype=np.float32).reshape(1, -1)
    K = tf.keras.backend
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))
    return categorical_focal_loss_fixed


"""
========
"""
def get_loss(name, parameters, **kwargs):
    if name == "categorical_crossentropy":
        return tf.keras.losses.CategoricalCrossentropy()
    elif name == "categorical_focal_loss":
        return categorical_focal_loss(**parameters)
    else:
        raise ValueError(f"Invalid loss function {name}")

def get_optimizer(name, parameters, **kwargs):
    if name == "adam":
        return tf.keras.optimizers.Adam(**parameters)
    elif name == "RMSprop":
        return tf.keras.optimizers.RMSprop(**parameters)
    else:
        raise ValueError(f"Invalid optimizer {name}")

def get_callbacks(labels, verbosity, early_stop=None, **kwargs):
    callbacks=[
        wandb.keras.WandbCallback(verbose=verbosity, labels=labels, save_weights_only=True),
    ]
    if early_stop is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=early_stop))
    return callbacks
"""
Models
======
"""
builders = {}
def add_model_builder(func):
    """Register a model builder."""
    global builders
    builder_name = func.__name__
    builders[builder_name] = func
    return func

@add_model_builder
def mlp_builder(input_shape, num_classes, layers_units=[375], dropout_rate=0.75, **kwargs):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(inputs)
    for i, l_units in enumerate(layers_units):
        x = tf.keras.layers.Dense(units=l_units)(x)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    classification_head = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=classification_head)
    return model

@add_model_builder
def seismicnet_builder(input_shape, num_classes, **kwargs):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=32, strides=2, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool1D(pool_size=3, strides=3)(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=16, strides=2, activation="relu")(x)
    x = tf.keras.layers.MaxPool1D(pool_size=3, strides=3)(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Conv1D(filters=128, kernel_size=8, strides=2, activation="relu")(x)
    x = tf.keras.layers.MaxPool1D(pool_size=3, strides=3)(x)
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=4, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv1D(filters=1401, kernel_size=8, strides=2, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1500, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.75)(x)
    classification_head = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=classification_head, name="SeismicNet")
    return model

@add_model_builder
def multihead_builder(input_shape, num_classes, filters=(64, 32, 32), **kwargs):
    assert len(filters) == 3, "filters size not equals to 3"
    inputs = tf.keras.layers.Input(shape=(input_shape))
    x = inputs
    # Head 1
    f1 = 64
    head_1 = tf.keras.layers.Conv1D(f1, 3, activation="relu")(x)
    head_1 = tf.keras.layers.Conv1D(f1, 3, activation="relu")(head_1)
    head_1 = tf.keras.layers.MaxPool1D(2)(head_1)
    head_1 = tf.keras.layers.Conv1D(f1, 3, activation="relu")(head_1)
    head_1 = tf.keras.layers.MaxPool1D(2)(head_1)
    head_1 = tf.keras.layers.Conv1D(f1, 3, activation="relu")(head_1)
    head_1 = tf.keras.layers.MaxPool1D(2)(head_1)
    # 748, 16
    # Head 2
    f2 = 32
    head_2 = tf.keras.layers.Conv1D(f2, 7, activation="relu")(x)
    head_2 = tf.keras.layers.MaxPool1D(2)(head_2)
    head_2 = tf.keras.layers.Conv1D(f2, 7, activation="relu")(head_2)
    head_2 = tf.keras.layers.MaxPool1D(2)(head_2)
    head_2 = tf.keras.layers.Conv1D(f2, 7, activation="relu", padding="same")(head_2)
    head_2 = tf.keras.layers.MaxPool1D(2, padding="same")(head_2)
    # 748, 32
    f3 = 32
    head_3 = tf.keras.layers.Conv1D(f3, 19, activation="relu", strides=4, padding="valid")(x)
    head_3 = tf.keras.layers.Conv1D(f3, 7, activation="relu", strides=1, padding="same")(head_3)
    head_3 = tf.keras.layers.MaxPool1D(2, padding="same")(head_3)
    # 748, 32
    x = tf.keras.layers.Concatenate()([head_1, head_2, head_3])
    x = tf.keras.layers.Flatten()(x)
    classification_head = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=classification_head, name="MultiHead")
    return model

@add_model_builder
def lstm_builder(input_shape, num_classes, lstm_units=[128, 128], dropout_rate=0.35, **kwargs):
    n_timesteps, n_features = input_shape
    inputs = tf.keras.layers.Input(shape=(n_timesteps, n_features))
    x = inputs
    for i, lstm_i_units in enumerate(lstm_units):
        if i + 1 != len(lstm_units):
            x = tf.keras.layers.LSTM(lstm_i_units, activation="tanh", return_sequences=True)(x)
        else:
            x = tf.keras.layers.LSTM(lstm_i_units, activation="tanh", return_sequences=False)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(units=256, activation="relu")(x)
    classification_head = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=classification_head, name="LSTM")
    return model


@add_model_builder
def gru_builder(input_shape, num_classes, gru_units=[128, 128], dropout_rate=0.35, **kwargs):
    n_timesteps, n_features = input_shape
    inputs = tf.keras.layers.Input(shape=(n_timesteps, n_features))
    x = inputs
    for i, lstm_i_units in enumerate(gru_units):
        if i + 1 != len(gru_units):
            x = tf.keras.layers.GRU(lstm_i_units, return_sequences=True)(x)
        else:
            x = tf.keras.layers.GRU(lstm_i_units, return_sequences=False)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(units=256, activation="relu")(x)
    classification_head = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=classification_head, name="GRU")
    return model


"""
Transformer
===========
"""
class FeatureAndPositionalEmbedding(tf.keras.layers.Layer):
    def get_angles(self, pos, i, d_model):
        angle_rates = 1/np.power(10000, (2*(i//2)) / np.float32(d_model))
        return angle_rates * pos

    def positional_embedding(self):
        maxlen = self.maxlen
        embedding_size = self.embedding_size
        angle_rads = self.get_angles(np.arange(maxlen)[:,np.newaxis],
                                np.arange(embedding_size)[np.newaxis, :],
                                embedding_size)
        angle_rads[:,0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:,1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def __init__(self, input_lenght=6000, embedding_size=128):
        super(FeatureAndPositionalEmbedding, self).__init__()
        self.maxlen = int(((input_lenght-21)/18)+1)
        self.embedding_size = embedding_size
        self.positional_embedding = self.positional_embedding()
        self.feature_embedding = tf.keras.Sequential(
            [tf.keras.layers.Conv1D(self.embedding_size/2, 21, activation="relu", strides=18),
            tf.keras.layers.Conv1D(self.embedding_size, 1, activation="tanh")]
        )
        
    def call(self, x):
        x = self.feature_embedding(x)
        return x + self.positional_embedding

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_size=128, num_heads=16, ff_dim=64, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_size)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"),
             tf.keras.layers.Dense(embedding_size),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training):
        attn_output = self.att(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

@add_model_builder
def transformer_builder(input_shape, num_classes, embedding_size=128, num_heads=32, ff_dim=128, dropout_rate=0.1, **kwargs):
    input_lenght = input_shape[0]
    inputs = tf.keras.layers.Input(shape=(input_shape))
    # Feature + Positional Embedding
    x = FeatureAndPositionalEmbedding(input_lenght, embedding_size)(inputs)
    x = TransformerBlock(embedding_size, num_heads, ff_dim, dropout_rate)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(units=ff_dim//2, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    classification_head = tf.keras.layers.Dense(units=num_classes, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=classification_head, name="Transformer")
    return model


__all__ = ["get_loss", "get_optimizer", "get_callbacks", "builders"] + list(builders.keys())