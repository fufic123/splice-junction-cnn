import keras
from keras import layers


def build_baseline_cnn(
    input_length: int = 60,
    n_classes: int = 3,
    n_filters: int = 64,
    kernel_size: int = 7,
    n_blocks: int = 1,
    dropout: float = 0.3,
    pool_type: str = "global",  # "global" or "local"
    l2_reg: float = 0.0,
) -> keras.Model:
    """
    Model A

    Architecture: one-hot (L x 4) -> [Conv1D + ReLU + Pool] x n_blocks
                  -> GlobalMaxPool -> Dense -> Softmax.
    """
    reg = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    inp = layers.Input(shape=(input_length, 4), name="sequence")
    x = inp
    for i in range(n_blocks):
        x = layers.Conv1D(
            n_filters, kernel_size, padding="same",
            activation="relu", kernel_regularizer=reg,
            name=f"conv_{i+1}",
        )(x)
        if pool_type == "local" and i < n_blocks - 1:
            x = layers.MaxPooling1D(pool_size=2, name=f"pool_{i+1}")(x)

    if pool_type == "global":
        x = layers.GlobalMaxPooling1D(name="global_pool")(x)
    else:
        x = layers.Flatten(name="flatten")(x)

    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=reg, name="dense_1")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout_2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="BaselineCNN")
    return model


def _residual_block(x, filters, kernel_size, dilation_rate, dropout, reg, block_id):
    """Single residual block with dilated convolution."""
    shortcut = x
    y = layers.Conv1D(
        filters, kernel_size, padding="same",
        dilation_rate=dilation_rate, activation="relu",
        kernel_regularizer=reg, name=f"res{block_id}_conv1",
    )(x)
    if dropout > 0:
        y = layers.Dropout(dropout, name=f"res{block_id}_drop")(y)
    y = layers.Conv1D(
        filters, kernel_size, padding="same",
        dilation_rate=1, activation=None,
        kernel_regularizer=reg, name=f"res{block_id}_conv2",
    )(y)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters, 1, padding="same",
            name=f"res{block_id}_proj",
        )(shortcut)
    y = layers.Add(name=f"res{block_id}_add")([shortcut, y])
    y = layers.Activation("relu", name=f"res{block_id}_relu")(y)
    return y


def build_dilated_residual_cnn(
    input_length: int = 60,
    n_classes: int = 3,
    n_filters: int = 64,
    kernel_size: int = 5,
    dilation_rates: tuple = (1, 2, 4),
    dropout: float = 0.3,
    l2_reg: float = 0.0,
) -> keras.Model:
    """
    Model B - CNN with dilated convolutions and residual connections.
    """
    reg = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    inp = layers.Input(shape=(input_length, 4), name="sequence")
    x = layers.Conv1D(
        n_filters, kernel_size, padding="same",
        activation="relu", kernel_regularizer=reg, name="stem_conv",
    )(inp)

    for i, dr in enumerate(dilation_rates):
        x = _residual_block(x, n_filters, kernel_size, dr, dropout, reg, i + 1)

    x = layers.GlobalMaxPooling1D(name="global_pool")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="head_drop")(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=reg, name="head_dense")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="DilatedResCNN")
    return model
