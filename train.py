from utils import process_args, config, factory
import os
import sys


def main():
    try:
        args = process_args.get_args()
        config_files = os.listdir(args.config)
        if not len(config_files):
            raise Exception("No config files were found")

        # Processing every config file

        for config_file in config_files:
            if args.verbose:
                print(f"Processing config file: {config_file}")
            path_to_config = os.path.join(args.config, config_file)
            config_object = config.process_config(path_to_config)

            if args.verbose:
                print("Importing the dataset")
            data_loader = factory.create("data_loader." + config_object.data_loader.name)(config_object,
                                                                                          config_object.exp.data_dir)

            if args.verbose:
                print("Creating the model")
            model = factory.create("models." + config_object.model.name)(config_object)

            if args.verbose:
                print("Creating the trainer")
            trainer = factory.create("trainers." + config_object.trainer.name)(model.get_model(),
                                                                               data_loader.get_training_data(),
                                                                               data_loader.get_validation_data(),
                                                                               config=config_object)

            trainer.train()

        if args.evaluate == 't':
            pass

    except Exception as error:
        print(error)
        sys.exit(1)


if __name__ == '__main__':
    main()
