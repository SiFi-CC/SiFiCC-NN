import tensorflow as tf
import spektral as sp


class ReZero(tf.keras.layers.Layer):
    r"""ReZero layer based on the paper
    'ReZero is All You Need: Fast Convergence at Large Depth' (arXiv:2003.04887v2)

    This layer computes:
        x_(i+1) = x_i + alpha_i*F(x_i)
    where alpha_i is the trainable residual weight, initialized with zero

    Inputs:
      List of layers with same shape.

    Returns:
      Output of ReZero layer with same shape as individual inputs

    Example:

      ReZero layer can by used in ResNet block
      #convolution layer with some prior layer. x serves as short cut path
      x = Conv2D(32, activation="relu", use_bias=True)(someOtherLayer)

      #two convolution layers in ResNetBlock
      fx = Conv2D(32, activation="relu", use_bias=True)(x)
      fx = Conv2D(32, use_bias=False)(fx)

      #addition like in normal ResNet block, but weighted with trainable residual weight
      x = ReZero()([x, fx])
      #final activation function of this ReZero/ResNet block
      x = Activation('relu')(x)
    """

    def __init__(self, **kwargs):
        super(ReZero, self).__init__(**kwargs)

    def build(self, input_shape):
        # create residual weight
        assert isinstance(input_shape, list)
        self.residualWeight = self.add_weight(name="residualWeight",
                                              shape=(1,),
                                              initializer=tf.keras.initializers.Zeros(),
                                              trainable=True)
        super(ReZero, self).build(input_shape)

    def call(self, inputs):
        assert isinstance(inputs, list)
        inputX, inputFx = inputs
        return inputX + self.residualWeight * inputFx


def adjustChannelSize(xInput, fxInput):
    r"""
    Expand channels of shortcut to match residual.
    If number of filters do not match an additional Conv1D
    with kernel size 1 is applied to the shortcut.

    Args:
        xInput: input layer to ResNetBlock with certain number of filters
        fxInput: output layer of convolution block of ResNetBlock with potentially different number
                 of filters
    Returns:
        A keras layer.
    """

    # get shapes of input layers
    inputShape = tf.keras.backend.int_shape(xInput)
    convBlockShape = tf.keras.backend.int_shape(fxInput)

    # check if number of filters are the same
    equalChannels = inputShape[-1] == convBlockShape[-1]

    # 1 X 1 conv if shape is different. Else identity.
    if not equalChannels:
        x = tf.keras.layers.Dense(convBlockShape[-1],
                                  activation="relu")(xInput)
        return x
    else:
        return xInput


def GCNConvResNetBlock(x,
                       A,
                       n_filter=64,
                       activation="relu",
                       conv_activation="relu",
                       kernel_initializer="glorot_uniform"):
    r"""
    ResNetBlock implementation used in the master thesis??????????
    'Vertex Reconstruction for Neutrino Events in the Double Chooz Experiment using Graph Neural
    Networks'
    Utilizes graph convolutions (Kipf & Welling) instead of conventional layers.Conv...
    Instead of the standard short cut the ReZero approach is used.

    Args:
        x: A keras layers object which is compatible with
           sp.layers.GCNConv with output shape [batch, nodes, features]
        A: Adjacency matrix (Modified Laplacian of shape [batch, nodes, nodes])
           e.g. in form of an Input layer:
           A = tf.keras.layers.Input(tensor = sp.layers.ops.sp_matrix_to_sp_tensor(sp.utils.gcn_filter(Adj)))
        n_filter: number of convolution filters
        conv_activation: activation function of the first convolution layer in the block
        kernel_initializer: initializer for the weights
        activation: activation function which is applied after adding identity to conv block
    Returns:
        A keras layer. Graph after GCNConvResNetBlock with shape [batch, n_nodes, n_filter]
    """

    # two convolution layers in ResNetBlock
    fx = sp.layers.GCNConv(n_filter,
                           activation=conv_activation,
                           use_bias=True,
                           kernel_initializer=kernel_initializer)([x, A])
    fx = sp.layers.GCNConv(n_filter,
                           use_bias=False,
                           kernel_initializer=kernel_initializer)([fx, A])

    # adjust the number of filters in the short cut path by eventually applying a Conv1D with kernel size 1
    x = adjustChannelSize(x, fx)

    # apply rezero layer
    x = ReZero()([x, fx])

    # apply activation function at the end
    x = tf.keras.layers.Activation(activation)(x)

    return x


def EdgeConvResNetBlock(x,
                        A,
                        n_filter=64,
                        activation="relu",
                        conv_activation="relu",
                        kernel_initializer="glorot_uniform"):
    r"""
    ResNetBlock implementation used in the master thesis
    'Vertex Reconstruction for Neutrino Events in the Double Chooz Experiment using Graph Neural
    Networks'
    Utilizes graph convolutions (Kipf & Welling) instead of conventional layers.Conv...
    Instead of the standard short cut the ReZero approach is used.

    Args:
        x: A keras layers object which is compatible with
           sp.layers.GCNConv with output shape [batch, nodes, features]
        A: Adjacency matrix (Modified Laplacian of shape [batch, nodes, nodes])
           e.g. in form of an Input layer:
           A = tf.keras.layers.Input(tensor = sp.layers.ops.sp_matrix_to_sp_tensor(sp.utils.gcn_filter(Adj)))
        n_filter: number of convolution filters
        conv_activation: activation function of the first convolution layer in the block
        kernel_initializer: initializer for the weights
        activation: activation function which is applied after adding identity to conv block
    Returns:
        A keras layer. Graph after GCNConvResNetBlock with shape [batch, n_nodes, n_filter]
    """

    # two convolution layers in ResNetBlock
    fx = sp.layers.EdgeConv(channels=n_filter,
                            activation=conv_activation,
                            use_bias=True,
                            kernel_initializer=kernel_initializer)([x, A])
    fx = sp.layers.EdgeConv(channels=n_filter,
                            use_bias=False,
                            kernel_initializer=kernel_initializer)([fx, A])

    # adjust the number of filters in the short cut path by eventually applying a Conv1D with kernel size 1
    x = adjustChannelSize(x, fx)

    # apply rezero layer
    x = ReZero()([x, fx])

    # apply activation function at the end
    x = tf.keras.layers.Activation(activation)(x)

    return x


def EdgeConvResNetBlockV2(x,
                          A,
                          n_filter=64,
                          activation="relu",
                          kernel_initializer="glorot_uniform"):
    """
    Updated ResNetBlock implementation from https://arxiv.org/pdf/1603.05027v3.pdf
    Compared to the standard ResNet implementation the order of BN and ReLU is changed for better
    identify mapping. Currently, Batch Normalization is not included!
    Instead of the standard short cut the ReZero approach is used.

    Args:
        x: A keras layers object which is compatible with
           sp.layers.GCNConv with output shape [batch, nodes, features]
        A: Adjacency matrix (Modified Laplacian of shape [batch, nodes, nodes])
           e.g. in form of an Input layer:
           A = tf.keras.layers.Input(tensor = sp.layers.ops.sp_matrix_to_sp_tensor(sp.utils.gcn_filter(Adj)))
        n_filter: number of convolution filters
        kernel_initializer: initializer for the weights
        activation: activation function which is applied after adding identity to conv block
    Returns:
        A keras layer. Graph after GCNConvResNetBlock with shape [batch, n_nodes, n_filter]
    """

    # two weight layers in ResNetBlock
    # fx = tf.keras.layers.BatchNormalization(axis=0)(x)
    fx = tf.keras.layers.Activation(activation)(x)
    fx = sp.layers.EdgeConv(channels=n_filter,
                            use_bias=True,
                            kernel_initializer=kernel_initializer)([fx, A])
    # fx = tf.keras.layers.BatchNormalization(axis=0)(fx)
    fx = tf.keras.layers.Activation(activation)(fx)
    fx = sp.layers.EdgeConv(channels=n_filter,
                            use_bias=False,
                            kernel_initializer=kernel_initializer)([fx, A])

    # adjust the number of filters in the short cut path by eventually applying a Conv1D with kernel size 1
    x = adjustChannelSize(x, fx)

    # apply rezero layer
    x = ReZero()([x, fx])
    return x


def resNetBlocks(implementation, **kwargs):
    if implementation == "GCNResNet":
        return GCNConvResNetBlock(**kwargs)


def EdgeConvResNetBlockV2BN(x,
                            A,
                            n_filter=64,
                            activation="relu",
                            kernel_initializer="glorot_uniform"):
    """
    Updated ResNetBlock implementation from https://arxiv.org/pdf/1603.05027v3.pdf
    Compared to the standard ResNet implementation the order of BN and ReLU is changed for better
    identify mapping. Currently, Batch Normalization is not included!
    Instead of the standard short cut the ReZero approach is used.

    Args:
        x: A keras layers object which is compatible with
           sp.layers.GCNConv with output shape [batch, nodes, features]
        A: Adjacency matrix (Modified Laplacian of shape [batch, nodes, nodes])
           e.g. in form of an Input layer:
           A = tf.keras.layers.Input(tensor = sp.layers.ops.sp_matrix_to_sp_tensor(sp.utils.gcn_filter(Adj)))
        n_filter: number of convolution filters
        kernel_initializer: initializer for the weights
        activation: activation function which is applied after adding identity to conv block
    Returns:
        A keras layer. Graph after GCNConvResNetBlock with shape [batch, n_nodes, n_filter]
    """

    # two weight layers in ResNetBlock
    fx = tf.keras.layers.BatchNormalization(axis=1)(x)
    fx = tf.keras.layers.Activation(activation)(x)
    fx = sp.layers.EdgeConv(channels=n_filter,
                            use_bias=True,
                            kernel_initializer=kernel_initializer)([fx, A])
    fx = tf.keras.layers.BatchNormalization(axis=1)(fx)
    fx = tf.keras.layers.Activation(activation)(fx)
    fx = sp.layers.EdgeConv(channels=n_filter,
                            use_bias=False,
                            kernel_initializer=kernel_initializer)([fx, A])

    # adjust the number of filters in the short cut path by eventually applying a Conv1D with kernel size 1
    x = adjustChannelSize(x, fx)

    # apply rezero layer
    x = ReZero()([x, fx])
    return x


def EdgeConvResNetBlockBN(x,
                          A,
                          n_filter=64,
                          activation="relu",
                          conv_activation="relu",
                          kernel_initializer="glorot_uniform"):
    r"""
    ResNetBlock implementation used in the master thesis
    'Vertex Reconstruction for Neutrino Events in the Double Chooz Experiment using Graph Neural
    Networks'
    Utilizes graph convolutions (Kipf & Welling) instead of conventional layers.Conv...
    Instead of the standard short cut the ReZero approach is used.

    Args:
        x: A keras layers object which is compatible with
           sp.layers.GCNConv with output shape [batch, nodes, features]
        A: Adjacency matrix (Modified Laplacian of shape [batch, nodes, nodes])
           e.g. in form of an Input layer:
           A = tf.keras.layers.Input(tensor = sp.layers.ops.sp_matrix_to_sp_tensor(sp.utils.gcn_filter(Adj)))
        n_filter: number of convolution filters
        conv_activation: activation function of the first convolution layer in the block
        kernel_initializer: initializer for the weights
        activation: activation function which is applied after adding identity to conv block
    Returns:
        A keras layer. Graph after GCNConvResNetBlock with shape [batch, n_nodes, n_filter]
    """

    # two convolution layers in ResNetBlock
    fx = sp.layers.EdgeConv(channels=n_filter,
                            activation=conv_activation,
                            use_bias=True,
                            kernel_initializer=kernel_initializer)([x, A])
    fx = sp.layers.EdgeConv(channels=n_filter,
                            use_bias=False,
                            kernel_initializer=kernel_initializer)([fx, A])

    # adjust the number of filters in the short cut path by eventually applying a Conv1D with kernel size 1
    x = adjustChannelSize(x, fx)

    # apply rezero layer
    x = ReZero()([x, fx])

    # apply activation function at the end
    x = tf.keras.layers.BatchNormalization(axis=1)(x)
    x = tf.keras.layers.Activation(activation)(x)

    return x


def EdgeConvRes2NetBlock(x, A, n_filter=64, scales=4, activation="relu", conv_activation="relu", kernel_initializer="glorot_uniform"):
    """
    Res2Net-inspired Block for Graph Neural Networks using progressive feature reuse.
    Each scale builds on the original input and the output of the previous scale.

    Args:
        x: Tensor of shape [batch_size, nodes, features] (Node features of the graph)
        A: Tensor of shape [batch_size, nodes, nodes] (Adjacency matrix or modified Laplacian of the graph)
        n_filter: Total number of output filters for the block
        scale: Number of scales for multi-scale feature extraction
        activation: Activation function (e.g., "relu")
        kernel_initializer: Weight initializer for convolution layers

    Returns:
        Tensor with shape [batch_size, nodes, n_filter] after applying multi-scale feature reuse.
    """

    outputs = []  # To store scale outputs

    # Initialize the first scale directly from the input
    current_output = x
    for i in range(scales):
        # Apply EdgeConv on the current output + original input
        current_output = sp.layers.EdgeConv(
            channels=n_filter,
            use_bias=True,
            kernel_initializer=kernel_initializer
        )([current_output, A])

        # Add the original input into the computation for progressive reuse
        if i > 0:
            current_output = adjustChannelSize(x, current_output)
            print("current_output.shape", current_output.shape)
            print("x.shape", x.shape)

            # apply rezero layer
            current_output = ReZero()([x, current_output])

        outputs.append(current_output)

    # Step 2: Concatenate all scale outputs
    # Concatenate along the feature dimension
    concatenated = tf.keras.layers.Concatenate(axis=-1)(outputs)

    # Step 3: Apply final EdgeConv for feature merging
    merged = sp.layers.EdgeConv(
        channels=n_filter,
        use_bias=False,
        kernel_initializer=kernel_initializer
    )([concatenated, A])

    x = adjustChannelSize(x, merged)

    # apply rezero layer
    x = ReZero()([x, merged])

    # Step 5: Apply Activation
    x_res = tf.keras.layers.Activation(activation)(x)

    return x_res


def EdgeConvRes2NetBlockBN(x, A, n_filter=64, scales=4, activation="relu", conv_activation="relu", kernel_initializer="glorot_uniform"):
    """
    Res2Net-inspired Block for Graph Neural Networks using progressive feature reuse.
    Each scale builds on the original input and the output of the previous scale.

    Args:
        x: Tensor of shape [batch_size, nodes, features] (Node features of the graph)
        A: Tensor of shape [batch_size, nodes, nodes] (Adjacency matrix or modified Laplacian of the graph)
        n_filter: Total number of output filters for the block
        scale: Number of scales for multi-scale feature extraction
        activation: Activation function (e.g., "relu")
        kernel_initializer: Weight initializer for convolution layers

    Returns:
        Tensor with shape [batch_size, nodes, n_filter] after applying multi-scale feature reuse.
    """
    outputs = []  # To store scale outputs

    # Initialize the first scale directly from the input
    current_output = x
    for i in range(scales):
        # Apply EdgeConv on the current output + original input
        current_output = sp.layers.EdgeConv(
            channels=n_filter,
            use_bias=True,
            kernel_initializer=kernel_initializer
        )([current_output, A])

        # Add the original input into the computation for progressive reuse
        if i > 0:
            current_output = adjustChannelSize(x, current_output)

            # apply rezero layer
            current_output = ReZero()([x, current_output])

        outputs.append(current_output)

    # Step 2: Concatenate all scale outputs
    # Concatenate along the feature dimension
    concatenated = tf.keras.layers.Concatenate(axis=-1)(outputs)

    # Step 3: Apply final EdgeConv for feature merging
    merged = sp.layers.EdgeConv(
        channels=n_filter,
        use_bias=False,
        kernel_initializer=kernel_initializer
    )([concatenated, A])

    x = adjustChannelSize(x, merged)

    # apply rezero layer
    x = ReZero()([x, merged])

    # Step 5: Apply Activation
    x = tf.keras.layers.BatchNormalization(axis=1)(x)
    x = tf.keras.layers.Activation(activation)(x)

    return x


def EdgeConvRes2NetBlockV2(x, A, n_filter=64, scales=4, activation="relu", conv_activation="relu", kernel_initializer="glorot_uniform"):
    """
    Res2Net-inspired Block for Graph Neural Networks using progressive feature reuse.
    Each scale builds on the original input and the output of the previous scale.

    Args:
        x: Tensor of shape [batch_size, nodes, features] (Node features of the graph)
        A: Tensor of shape [batch_size, nodes, nodes] (Adjacency matrix or modified Laplacian of the graph)
        n_filter: Total number of output filters for the block
        scale: Number of scales for multi-scale feature extraction
        activation: Activation function (e.g., "relu")
        kernel_initializer: Weight initializer for convolution layers

    Returns:
        Tensor with shape [batch_size, nodes, n_filter] after applying multi-scale feature reuse.
    """
    outputs = []  # To store scale outputs

    # Initialize the first scale directly from the input
    current_output = x
    for i in range(scales):
        # Apply EdgeConv on the current output + original input
        current_output = tf.keras.layers.Activation(activation)(current_output)
        current_output = sp.layers.EdgeConv(
            channels=n_filter,
            use_bias=True,
            kernel_initializer=kernel_initializer
        )([current_output, A])

        # Add the original input into the computation for progressive reuse
        if i > 0:
            current_output = adjustChannelSize(x, current_output)

            # apply rezero layer
            current_output = ReZero()([x, current_output])

        outputs.append(current_output)

    # Step 2: Concatenate all scale outputs
    # Concatenate along the feature dimension
    concatenated = tf.keras.layers.Concatenate(axis=-1)(outputs)

    # Step 3: Apply final EdgeConv for feature merging
    merged = sp.layers.EdgeConv(
        channels=n_filter,
        use_bias=False,
        kernel_initializer=kernel_initializer
    )([concatenated, A])

    x = adjustChannelSize(x, merged)

    # apply rezero layer
    x = ReZero()([x, merged])

    return x


def EdgeConvRes2NetBlockV2BN(x, A, n_filter=64, scales=4, activation="relu", conv_activation="relu", kernel_initializer="glorot_uniform"):
    """
    Res2Net-inspired Block for Graph Neural Networks using progressive feature reuse.
    Each scale builds on the original input and the output of the previous scale.

    Args:
        x: Tensor of shape [batch_size, nodes, features] (Node features of the graph)
        A: Tensor of shape [batch_size, nodes, nodes] (Adjacency matrix or modified Laplacian of the graph)
        n_filter: Total number of output filters for the block
        scale: Number of scales for multi-scale feature extraction
        activation: Activation function (e.g., "relu")
        kernel_initializer: Weight initializer for convolution layers

    Returns:
        Tensor with shape [batch_size, nodes, n_filter] after applying multi-scale feature reuse.
    """
    outputs = []  # To store scale outputs

    # Initialize the first scale directly from the input
    current_output = x
    for i in range(scales):
        # Apply EdgeConv on the current output + original input
        current_output = tf.keras.layers.BatchNormalization(
            axis=1)(current_output)
        current_output = tf.keras.layers.Activation(activation)(current_output)
        current_output = sp.layers.EdgeConv(
            channels=n_filter,
            use_bias=True,
            kernel_initializer=kernel_initializer
        )([current_output, A])

        # Add the original input into the computation for progressive reuse
        if i > 0:
            current_output = adjustChannelSize(x, current_output)

            # apply rezero layer
            current_output = ReZero()([x, current_output])

        outputs.append(current_output)

    # Step 2: Concatenate all scale outputs
    # Concatenate along the feature dimension
    concatenated = tf.keras.layers.Concatenate(axis=-1)(outputs)

    # Step 3: Apply final EdgeConv for feature merging
    merged = sp.layers.EdgeConv(
        channels=n_filter,
        use_bias=False,
        kernel_initializer=kernel_initializer
    )([concatenated, A])

    x = adjustChannelSize(x, merged)

    # apply rezero layer
    x = ReZero()([x, merged])

    return x


def EdgeConvResNeXtBlockV2(x, A, n_filter=64, cardinality=4, activation="relu", conv_activation="relu", kernel_initializer="glorot_uniform"):
    """
    ResNeXt block with EdgeConv layers and improved activation function preceding the convolution layer.

    Args:
        x: Input tensor.
        A: Adjacency matrix.
        n_filter: Number of filters.
        cardinality: Number of independent paths.
        activation: Activation function applied at the end.
        conv_activation: Activation function applied in convolution layers.
        kernel_initializer: Initializer for the weights.

    Returns:
        A keras layer. Graph after EdgeConvResNeXtBlockV2 with shape [batch, n_nodes, n_filter].
    """

    # Split the input into cardinality paths
    split_channels = n_filter // cardinality
    splits = tf.split(x, num_or_size_splits=cardinality, axis=-1)

    # Apply EdgeConv to each path and concatenate
    paths = []
    for i in range(cardinality):
        y = tf.keras.layers.Activation(conv_activation)(splits[i])
        y = sp.layers.EdgeConv(channels=split_channels,
                               use_bias=True,
                               kernel_initializer=kernel_initializer)([y, A])
        paths.append(y)

    # Concatenate all paths
    y = tf.concat(paths, axis=-1)

    # Adjust the number of filters in the short cut path by eventually applying a Conv1D with kernel size 1
    x = adjustChannelSize(x, y)

    # Apply rezero layer
    x = ReZero()([x, y])

    return x


def EdgeConvResNeXtBlockV2BN(x, A, n_filter=64, cardinality=4, activation="relu", conv_activation="relu", kernel_initializer="glorot_uniform"):
    """
    ResNeXt block with EdgeConv layers, improved activation function preceding the convolution layer, and batch normalization.

    Args:
        x: Input tensor.
        A: Adjacency matrix.
        n_filter: Number of filters.
        cardinality: Number of independent paths.
        activation: Activation function applied at the end.
        conv_activation: Activation function applied in convolution layers.
        kernel_initializer: Initializer for the weights.

    Returns:
        A keras layer. Graph after EdgeConvResNeXtBlockV2BN with shape [batch, n_nodes, n_filter].
    """

    # Split the input into cardinality paths
    split_channels = n_filter // cardinality
    splits = tf.split(x, num_or_size_splits=cardinality, axis=-1)

    # Apply EdgeConv to each path and concatenate
    paths = []
    for i in range(cardinality):
        y = tf.keras.layers.BatchNormalization(axis=1)(splits[i])
        y = tf.keras.layers.Activation(conv_activation)(y)
        y = sp.layers.EdgeConv(channels=split_channels,
                               use_bias=True,
                               kernel_initializer=kernel_initializer)([y, A])
        paths.append(y)

    # Concatenate all paths
    y = tf.concat(paths, axis=-1)

    # Adjust the number of filters in the short cut path by eventually applying a Conv1D with kernel size 1
    x = adjustChannelSize(x, y)

    # Apply rezero layer
    x = ReZero()([x, y])

    return x


def EdgeConvResNeXtBlock(x, A, n_filter=64, cardinality=4, activation="relu", conv_activation="relu", kernel_initializer="glorot_uniform"):
    """
    ResNeXt block with EdgeConv layers and improved activation function preceding the convolution layer.

    Args:
        x: Input tensor.
        A: Adjacency matrix.
        n_filter: Number of filters.
        cardinality: Number of independent paths.
        activation: Activation function applied at the end.
        conv_activation: Activation function applied in convolution layers.
        kernel_initializer: Initializer for the weights.

    Returns:
        A keras layer. Graph after EdgeConvResNeXtBlockV2 with shape [batch, n_nodes, n_filter].
    """

    # Split the input into cardinality paths
    split_channels = n_filter // cardinality
    splits = tf.split(x, num_or_size_splits=cardinality, axis=-1)

    # Apply EdgeConv to each path and concatenate
    paths = []
    for i in range(cardinality):
        y = sp.layers.EdgeConv(channels=split_channels,
                               use_bias=True,
                               kernel_initializer=kernel_initializer)([splits[i], A])
        paths.append(y)

    # Concatenate all paths
    y = tf.concat(paths, axis=-1)

    # Adjust the number of filters in the short cut path by eventually applying a Conv1D with kernel size 1
    x = adjustChannelSize(x, y)

    # Apply rezero layer
    x = ReZero()([x, y])

    x = tf.keras.layers.Activation(activation)(x)

    return x


def EdgeConvResNeXtBlockBN(x, A, n_filter=64, cardinality=4, activation="relu", conv_activation="relu", kernel_initializer="glorot_uniform"):
    """
    ResNeXt block with EdgeConv layers, improved activation function preceding the convolution layer, and batch normalization.

    Args:
        x: Input tensor.
        A: Adjacency matrix.
        n_filter: Number of filters.
        cardinality: Number of independent paths.
        activation: Activation function applied at the end.
        conv_activation: Activation function applied in convolution layers.
        kernel_initializer: Initializer for the weights.

    Returns:
        A keras layer. Graph after EdgeConvResNeXtBlockV2BN with shape [batch, n_nodes, n_filter].
    """

    # Split the input into cardinality paths
    split_channels = n_filter // cardinality
    splits = tf.split(x, num_or_size_splits=cardinality, axis=-1)

    # Apply EdgeConv to each path and concatenate
    paths = []
    for i in range(cardinality):
        y = sp.layers.EdgeConv(channels=split_channels,
                               use_bias=True,
                               kernel_initializer=kernel_initializer)([splits[i], A])
        paths.append(y)

    # Concatenate all paths
    y = tf.concat(paths, axis=-1)

    # Adjust the number of filters in the short cut path by eventually applying a Conv1D with kernel size 1
    x = adjustChannelSize(x, y)

    # Apply rezero layer
    x = ReZero()([x, y])

    x = tf.keras.layers.BatchNormalization(axis=1)(x)
    x = tf.keras.layers.Activation(activation)(x)

    return x
