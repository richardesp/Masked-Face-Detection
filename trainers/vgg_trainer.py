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
        for index in range(len(self.config.callbacks)):
            self.callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(self.config.callbacks[index].checkpoint_dir,
                                          self.config.exp.name + "-" + self.config.callbacks[index].checkpoint_monitor),
                    monitor=self.config.callbacks[index].checkpoint_monitor,
                    mode=self.config.callbacks[index].checkpoint_mode,
                    save_best_only=self.config.callbacks[index].checkpoint_save_best_only,
                    verbose=self.config.callbacks[index].checkpoint_verbose,
                    save_format="h5",
                )
            )

            print(f"Callback {index + 1}: {self.config.callbacks[index].checkpoint_monitor}, "
                  f"{self.config.callbacks[index].checkpoint_mode}, "
                  f"{self.config.callbacks[index].checkpoint_save_best_only}, "
                  f"{self.config.callbacks[index].checkpoint_verbose}")

    def get_callbacks(self) -> list:
        return self.callbacks
