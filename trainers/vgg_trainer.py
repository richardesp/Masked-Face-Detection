from base.base_trainer import BaseTrain
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from optimizers.learning_rate_schedules import LearningRateSchedules
import matplotlib.pyplot as plt


class ModelTrainer(BaseTrain):
    def __init__(self, model, training_data, validation_data, config):
        super(ModelTrainer, self).__init__(model, training_data, validation_data, config)
        self.callbacks = []

        # Previously loading callbacks for prepare the model for training
        self.load_callbacks()

    def load_callbacks(self, verbose=True):
        for index in range(len(self.config.callbacks)):

            # Appending callback checkpoint types
            if self.config.callbacks[index].callback_type == "ModelCheckpoint":
                self.callbacks.append(
                    ModelCheckpoint(
                        filepath=os.path.join(self.config.callbacks[index].checkpoint_dir,
                                              self.config.exp.name + "-" + self.config.callbacks[
                                                  index].checkpoint_monitor + ".h5"),
                        monitor=self.config.callbacks[index].checkpoint_monitor,
                        mode=self.config.callbacks[index].checkpoint_mode,
                        save_best_only=self.config.callbacks[index].checkpoint_save_best_only,
                        verbose=self.config.callbacks[index].checkpoint_verbose,
                        save_format="h5",
                    )
                )

                if verbose:
                    print(f"Callback {index + 1}: {self.config.callbacks[index].callback_type}, "
                          f"{self.config.callbacks[index].checkpoint_mode}, "
                          f"{self.config.callbacks[index].checkpoint_save_best_only}, "
                          f"{self.config.callbacks[index].checkpoint_verbose}")

            # Appending callback earlyStopping types
            elif self.config.callbacks[index].callback_type == "EarlyStopping":
                self.callbacks.append(
                    EarlyStopping(
                        monitor=self.config.callbacks[index].early_stopping_monitor,
                        mode=self.config.callbacks[index].early_stopping_mode,
                        verbose=self.config.callbacks[index].early_stopping_verbose,
                        patience=self.config.callbacks[index].early_stopping_patience
                    )
                )

                if verbose:
                    print(f"Callback {index + 1}: {self.config.callbacks[index].callback_type}, "
                          f"{self.config.callbacks[index].early_stopping_monitor}, "
                          f"{self.config.callbacks[index].early_stopping_mode}, "
                          f"{self.config.callbacks[index].early_stopping_verbose}")

        # Appending callback for a learning rate scheduler, if it's specified
        if self.config.model.learning_rate_schedules != "constant":
            lr_scheduler = None
            learning_rates_class = LearningRateSchedules(initial_learning_rate=self.config.model.learning_rate)

            if self.config.model.learning_rate_schedules == "time_based_decay":
                lr_scheduler = learning_rates_class.get_lr_time_based_decay()

            self.callbacks.append(
                LearningRateScheduler(lr_scheduler, verbose=1)
            )

            if verbose:
                print(
                    f"Callback for learning rate scheduler: "
                    f"{self.config.model.learning_rate_schedules}")

    def train(self, plot=True):
        history = self.model.fit(
            self.training_data,
            epochs=self.config.trainer.num_epochs,
            batch_size=self.config.trainer.batch_size,
            validation_data=self.validation_data,
            callbacks=self.callbacks,
            verbose=self.config.trainer.verbose_training
        )

        # If the user specified that a graph of the training phase is saved
        if plot:
            # summarize history for accuracy
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')

            if not os.path.exists(self.config.trainer.plots_dir):
                os.mkdir(self.config.trainer.plots_dir)

            # Saving history for accuracy
            plots_path = os.path.join(self.config.trainer.plots_dir,
                                      self.config.exp.name + "-accuracy.png")

            plt.savefig(plots_path)

            # Removing previously plot
            plt.clf()

            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')

            # Saving history for loss
            plots_path = os.path.join(self.config.trainer.plots_dir,
                                      self.config.exp.name + "-loss.png")

            plt.savefig(plots_path)

        # Saving the final model
        self.model.save(os.path.join(self.config.trainer.models_dir,
                                     self.config.exp.name + "-final_model.h5"))
