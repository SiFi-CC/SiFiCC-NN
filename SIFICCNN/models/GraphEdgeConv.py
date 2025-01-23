import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Input
from spektral.layers import EdgeConv, GlobalMaxPool

from SIFICCNN.utils.layers import *

import inspect
import types


def SiFiECRNShort(
    F=10, nFilter=32, activation="relu", n_out=1, activation_out="sigmoid", dropout=0.0, task=None,
):
    """
    Graph EdgeConv model used in the paper: "To be released"

    Args:
        F (int):                Number of node attributes
        nFilter (int):          Number of filters in the starting layer
        activation (str):       Activation function used
        n_out (int):            Number of output nodes
        activation_out (str):   Output activation function
        dropout (float):        Dropout percentage

    Returns:
        Keras model
    """

    X_in = Input(shape=(F,))
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), dtype=tf.int64)

    x = EdgeConv(channels=nFilter, activation="relu")([X_in, A_in])

    x = EdgeConvResNetBlockV2(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNetBlockV2(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNetBlockV2(*[x, A_in], n_filter=nFilter)

    x = GlobalMaxPool()([x, I_in])

    if dropout > 0:
        x = Dropout(dropout)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name="SiFiECRNShort")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # compile model based on the number of output nodes
    # Since the model can be universally used for classification and regression it is helpful to
    # make an exception here
    # binary classification setup
    if task == "classification":
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
    # regression setup
    else:
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)

    return model


def SiFiECRNShortOld(
    F=10, nFilter=32, activation="relu", n_out=1, activation_out="sigmoid", dropout=0.0
):
    """
    Graph EdgeConv model used in the paper: "To be released"

    Args:
        F (int):                Number of node attributes
        nFilter (int):          Number of filters in the starting layer
        activation (str):       Activation function used
        n_out (int):            Number of output nodes
        activation_out (str):   Output activation function
        dropout (float):        Dropout percentage

    Returns:
        Keras model
    """

    X_in = Input(shape=(F,))
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), dtype=tf.int64)

    x = EdgeConv(channels=nFilter, activation="relu")([X_in, A_in])

    x = EdgeConvResNetBlock(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNetBlock(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNetBlock(*[x, A_in], n_filter=nFilter)

    x = GlobalMaxPool()([x, I_in])

    if dropout > 0:
        x = Dropout(dropout)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name="SiFiECRNShort")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # compile model based on the number of output nodes
    # Since the model can be universally used for classification and regression it is helpful to
    # make an exception here
    # binary classification setup
    if n_out == 1:
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
    # regression setup
    else:
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)

    return model


def SiFiECRN4(
    F=10, nFilter=32, activation="relu", n_out=1, activation_out="sigmoid", dropout=0.0
):
    """
    Graph EdgeConv model used in the paper: "To be released"

    Args:
        F (int):                Number of node attributes
        nFilter (int):          Number of filters in the starting layer
        activation (str):       Activation function used
        n_out (int):            Number of output nodes
        activation_out (str):   Output activation function
        dropout (float):        Dropout percentage

    Returns:
        Keras model
    """

    X_in = Input(shape=(F,))
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), dtype=tf.int64)

    x = EdgeConv(channels=nFilter, activation="relu")([X_in, A_in])

    x = EdgeConvResNetBlockV2(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNetBlockV2(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNetBlockV2(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNetBlockV2(*[x, A_in], n_filter=nFilter)

    x = GlobalMaxPool()([x, I_in])

    if dropout > 0:
        x = Dropout(dropout)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name="SiFiECRN4")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # compile model based on the number of output nodes
    # Since the model can be universally used for classification and regression it is helpful to
    # make an exception here
    # binary classification setup
    if n_out == 1:
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
    # regression setup
    else:
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)

    return model


def SiFiECRN5(
    F=10, nFilter=32, activation="relu", n_out=1, activation_out="sigmoid", dropout=0.0
):
    """
    Graph EdgeConv model used in the paper: "To be released"

    Args:
        F (int):                Number of node attributes
        nFilter (int):          Number of filters in the starting layer
        activation (str):       Activation function used
        n_out (int):            Number of output nodes
        activation_out (str):   Output activation function
        dropout (float):        Dropout percentage

    Returns:
        Keras model
    """

    X_in = Input(shape=(F,))
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), dtype=tf.int64)

    x = EdgeConv(channels=nFilter, activation="relu")([X_in, A_in])

    x = EdgeConvResNetBlockV2(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNetBlockV2(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNetBlockV2(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNetBlockV2(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNetBlockV2(*[x, A_in], n_filter=nFilter)

    x = GlobalMaxPool()([x, I_in])

    if dropout > 0:
        x = Dropout(dropout)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name="SiFiECRN5")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # compile model based on the number of output nodes
    # Since the model can be universally used for classification and regression it is helpful to
    # make an exception here
    # binary classification setup
    if n_out == 1:
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
    # regression setup
    else:
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)

    return model


def SiFiECRNShortBN(
    F=10, nFilter=32, activation="relu", n_out=1, activation_out="sigmoid", dropout=0.0
):
    """
    Graph EdgeConv model used in the paper: "To be released"

    Args:
        F (int):                Number of node attributes
        nFilter (int):          Number of filters in the starting layer
        activation (str):       Activation function used
        n_out (int):            Number of output nodes
        activation_out (str):   Output activation function
        dropout (float):        Dropout percentage

    Returns:
        Keras model
    """

    X_in = Input(shape=(F,))
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), dtype=tf.int64)

    x = EdgeConv(channels=nFilter, activation="relu")([X_in, A_in])

    x = EdgeConvResNetBlockV2BN(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNetBlockV2BN(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNetBlockV2BN(*[x, A_in], n_filter=nFilter)

    x = GlobalMaxPool()([x, I_in])

    if dropout > 0:
        x = Dropout(dropout)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name="SiFiECRNShort")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # compile model based on the number of output nodes
    # Since the model can be universally used for classification and regression it is helpful to
    # make an exception here
    # binary classification setup
    if n_out == 1:
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
    # regression setup
    else:
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)

    return model


def SiFiECRNOldBN(
    F=10, nFilter=32, activation="relu", n_out=1, activation_out="sigmoid", dropout=0.0
):
    """
    Graph EdgeConv model used in the paper: "To be released"

    Args:
        F (int):                Number of node attributes
        nFilter (int):          Number of filters in the starting layer
        activation (str):       Activation function used
        n_out (int):            Number of output nodes
        activation_out (str):   Output activation function
        dropout (float):        Dropout percentage

    Returns:
        Keras model
    """

    X_in = Input(shape=(F,))
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), dtype=tf.int64)

    x = EdgeConv(channels=nFilter, activation="relu")([X_in, A_in])

    x = EdgeConvResNetBlockBN(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNetBlockBN(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNetBlockBN(*[x, A_in], n_filter=nFilter)

    x = GlobalMaxPool()([x, I_in])

    if dropout > 0:
        x = Dropout(dropout)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name="SiFiECRNShort")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # compile model based on the number of output nodes
    # Since the model can be universally used for classification and regression it is helpful to
    # make an exception here
    # binary classification setup
    if n_out == 1:
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
    # regression setup
    else:
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)

    return model


def create_model_edgeconv_res2net(
    F, nFilter, activation, n_out, activation_out, dropout
):
    """
    Create a model using EdgeConvRes2NetBlock layers.
    """
    X_in = Input(shape=(F,))
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), dtype=tf.int64)

    x = EdgeConv(channels=nFilter, activation="relu")([X_in, A_in])

    x = EdgeConvRes2NetBlock(*[x, A_in], n_filter=nFilter)
    x = EdgeConvRes2NetBlock(*[x, A_in], n_filter=nFilter)
    x = EdgeConvRes2NetBlock(*[x, A_in], n_filter=nFilter)

    x = GlobalMaxPool()([x, I_in])

    if dropout > 0:
        x = Dropout(dropout)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name="EdgeConvRes2Net")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    if n_out == 1:
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
    else:
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)

    return model


def create_model_edgeconv_res2net_bn(
    F, nFilter, activation, n_out, activation_out, dropout
):
    """
    Create a model using EdgeConvRes2NetBlockBN layers.
    """
    X_in = Input(shape=(F,))
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), dtype=tf.int64)

    x = EdgeConv(channels=nFilter, activation="relu")([X_in, A_in])

    x = EdgeConvRes2NetBlockBN(*[x, A_in], n_filter=nFilter)
    x = EdgeConvRes2NetBlockBN(*[x, A_in], n_filter=nFilter)
    x = EdgeConvRes2NetBlockBN(*[x, A_in], n_filter=nFilter)

    x = GlobalMaxPool()([x, I_in])

    if dropout > 0:
        x = Dropout(dropout)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name="EdgeConvRes2NetBN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    if n_out == 1:
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
    else:
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)

    return model


def create_model_edgeconv_res2net_v2(
    F, nFilter, activation, n_out, activation_out, dropout
):
    """
    Create a model using EdgeConvRes2NetBlockV2 layers.
    """
    X_in = Input(shape=(F,))
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), dtype=tf.int64)

    x = EdgeConv(channels=nFilter, activation="relu")([X_in, A_in])

    x = EdgeConvRes2NetBlockV2(*[x, A_in], n_filter=nFilter)
    x = EdgeConvRes2NetBlockV2(*[x, A_in], n_filter=nFilter)
    x = EdgeConvRes2NetBlockV2(*[x, A_in], n_filter=nFilter)

    x = GlobalMaxPool()([x, I_in])

    if dropout > 0:
        x = Dropout(dropout)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name="EdgeConvRes2NetV2")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    if n_out == 1:
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
    else:
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)

    return model


def create_model_edgeconv_res2net_v2_bn(
    F, nFilter, activation, n_out, activation_out, dropout
):
    """
    Create a model using EdgeConvRes2NetBlockV2BN layers.
    """
    X_in = Input(shape=(F,))
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), dtype=tf.int64)

    x = EdgeConv(channels=nFilter, activation="relu")([X_in, A_in])

    x = EdgeConvRes2NetBlockV2BN(*[x, A_in], n_filter=nFilter)
    x = EdgeConvRes2NetBlockV2BN(*[x, A_in], n_filter=nFilter)
    x = EdgeConvRes2NetBlockV2BN(*[x, A_in], n_filter=nFilter)

    x = GlobalMaxPool()([x, I_in])

    if dropout > 0:
        x = Dropout(dropout)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name="EdgeConvRes2NetV2BN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    if n_out == 1:
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
    else:
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)

    return model


def create_model_edgeconv_resnext_v2(
    F, nFilter, activation, n_out, activation_out, dropout
):
    """
    Create a model using EdgeConvResNeXtBlockV2 layers.
    """
    X_in = Input(shape=(F,))
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), dtype=tf.int64)

    x = EdgeConv(channels=nFilter, activation="relu")([X_in, A_in])

    x = EdgeConvResNeXtBlockV2(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNeXtBlockV2(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNeXtBlockV2(*[x, A_in], n_filter=nFilter)

    x = GlobalMaxPool()([x, I_in])

    if dropout > 0:
        x = Dropout(dropout)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name="EdgeConvResNeXtV2")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    if n_out == 1:
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
    else:
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)

    return model


def create_model_edgeconv_resnext_v2_bn(
    F, nFilter, activation, n_out, activation_out, dropout
):
    """
    Create a model using EdgeConvResNeXtBlockV2BN layers.
    """
    X_in = Input(shape=(F,))
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), dtype=tf.int64)

    x = EdgeConv(channels=nFilter, activation="relu")([X_in, A_in])

    x = EdgeConvResNeXtBlockV2BN(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNeXtBlockV2BN(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNeXtBlockV2BN(*[x, A_in], n_filter=nFilter)

    x = GlobalMaxPool()([x, I_in])

    if dropout > 0:
        x = Dropout(dropout)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name="EdgeConvResNeXtV2BN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    if n_out == 1:
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
    else:
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)

    return model


def create_model_edgeconv_resnext(
    F, nFilter, activation, n_out, activation_out, dropout
):
    """
    Create a model using EdgeConvResNeXtBlock layers.
    """
    X_in = Input(shape=(F,))
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), dtype=tf.int64)

    x = EdgeConv(channels=nFilter, activation="relu")([X_in, A_in])

    x = EdgeConvResNeXtBlock(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNeXtBlock(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNeXtBlock(*[x, A_in], n_filter=nFilter)

    x = GlobalMaxPool()([x, I_in])

    if dropout > 0:
        x = Dropout(dropout)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name="EdgeConvResNeXt")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    if n_out == 1:
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
    else:
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)

    return model


def create_model_edgeconv_resnext_bn(
    F, nFilter, activation, n_out, activation_out, dropout
):
    """
    Create a model using EdgeConvResNeXtBlockBN layers.
    """
    X_in = Input(shape=(F,))
    A_in = Input(shape=(None,), sparse=True)
    I_in = Input(shape=(), dtype=tf.int64)

    x = EdgeConv(channels=nFilter, activation="relu")([X_in, A_in])

    x = EdgeConvResNeXtBlockBN(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNeXtBlockBN(*[x, A_in], n_filter=nFilter)
    x = EdgeConvResNeXtBlockBN(*[x, A_in], n_filter=nFilter)

    x = GlobalMaxPool()([x, I_in])

    if dropout > 0:
        x = Dropout(dropout)(x)

    out = Dense(n_out, activation=activation_out)(x)

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name="EdgeConvResNeXtBN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    if n_out == 1:
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
    else:
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
    model.compile(optimizer=optimizer, loss=loss, metrics=list_metrics)

    return model


def get_models():
    current_module = inspect.getmodule(inspect.currentframe())
    functions = {
        name: obj
        for name, obj in globals().items()
        if isinstance(obj, types.FunctionType)
        and inspect.getmodule(obj) == current_module
        and name != "get_models"
    }
    return functions
