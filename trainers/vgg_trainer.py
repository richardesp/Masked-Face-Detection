from base.base_trainer import BaseTrain
import os
from tensorflow.keras.callbacks import ModelCheckpoint


class ModelTrainer(BaseTrain):
    def __init__(self, model, training_data, validation_data, config):
        super(ModelTrainer, self).__init__(model, training_data, validation_data, config)
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.load_callbacks()

    def load_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                      '%s-{epoch:02d}-{val_acc:.4f}.h5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                verbose=self.config.callbacks.checkpoint_verbose,
                save_format="h5",
            )
        )

    def get_callbacks(self) -> list:
        return self.callbacks
