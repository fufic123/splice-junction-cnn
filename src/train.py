import keras

def get_callbacks(patience_es: int = 10, patience_lr: int = 5, min_lr: float = 1e-6):
    """Standard callbacks: EarlyStopping + ReduceLROnPlateau."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience_es,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=patience_lr, min_lr=min_lr, verbose=1,
        ),
    ]


def compile_model(model: keras.Model, lr: float = 1e-3):
    """Compile with Adam + categorical crossentropy."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: keras.Model,
    X_train, y_train,
    X_val, y_val,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience_es: int = 10,
    patience_lr: int = 5,
    verbose: int = 0,
):
    """Compile and train. Returns keras History object."""
    compile_model(model, lr=lr)
    callbacks = get_callbacks(patience_es=patience_es, patience_lr=patience_lr)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    return history
