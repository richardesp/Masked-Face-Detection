"""
Functions for extract training and models params from a .json file

"""

import json
from dotmap import DotMap
import os
import time


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict, json_file


def process_config(json_file):
    config, _, json_file = get_config_from_json(json_file)
    json_string_name = json_file.rsplit('_', 1)[1].rsplit('.', 1)[0]  # e.g. mid-01
    dataloader_string_name = 'dl' + config.data_loader.name.rsplit('_', 1)[1].rsplit('.', 1)[0]  # e.g. dl01
    model_string_name = 'm' + config.model.name.rsplit('_', 1)[1].rsplit('.', 1)[0]  # e.g. m01

    for index in range(len(config.callbacks)):

        # We can process any type of callback from .json file
        # In this case we are going to process the checkpoints' callback type
        # for saving all checkpoints generated in a specified directory
        if config.callbacks[index].callback_type == "ModelCheckpoint":
            config.callbacks[index].exp_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/", time.localtime()))
            config.callbacks[index].checkpoint_dir = os.path.join("experiments",
                                                                  time.strftime("%Y-%m-%d/", time.localtime()),
                                                                  "{}-{}-{}-checkpoints/".format(
                                                                      model_string_name,
                                                                      json_string_name,
                                                                      dataloader_string_name

                                                                  ))

        # Path for saving all graphs during training phase
        config.trainer.plots_dir = os.path.join("experiments",
                                                time.strftime("%Y-%m-%d/", time.localtime()),
                                                "{}-{}-{}-plots/".format(
                                                    model_string_name,
                                                    json_string_name,
                                                    dataloader_string_name
                                                ))

        # Path for saving final training models
        config.trainer.models_dir = os.path.join("experiments",
                                                 time.strftime("%Y-%m-%d/", time.localtime()),
                                                 "{}-{}-{}-models/".format(
                                                     model_string_name,
                                                     json_string_name,
                                                     dataloader_string_name
                                                 ))

    return config
