import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Input
from spektral.layers import EdgeConv, GlobalMaxPool

from SIFICCNN.utils.layers import EdgeConvResNetBlockV2, EdgeConvResNetBlock

import inspect
import types



def SiFiECRNShort(F=10,
                  nFilter=32,
                  activation="relu",
                  n_out=1,
                  activation_out="sigmoid",
                  dropout=0.0):
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

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name = "SiFiECRNShort")

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

def SiFiECRNShortOld(F=10,
                  nFilter=32,
                  activation="relu",
                  n_out=1,
                  activation_out="sigmoid",
                  dropout=0.0):
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

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name = "SiFiECRNShort")

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


def SiFiECRN4(F=10,
                  nFilter=32,
                  activation="relu",
                  n_out=1,
                  activation_out="sigmoid",
                  dropout=0.0):
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

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name = "SiFiECRN4")

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

def SiFiECRN5(F=10,
                  nFilter=32,
                  activation="relu",
                  n_out=1,
                  activation_out="sigmoid",
                  dropout=0.0):
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

    model = Model(inputs=[X_in, A_in, I_in], outputs=out, name = "SiFiECRN5")

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



def get_models():
    current_module = inspect.getmodule(inspect.currentframe())
    functions = {name: obj for name, obj in globals().items() 
                 if isinstance(obj, types.FunctionType) and inspect.getmodule(obj) == current_module and name != 'get_models'}
    return functions

