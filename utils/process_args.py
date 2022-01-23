import argparse
import os


def get_args():
    arg_parser = argparse.ArgumentParser(description=__doc__)

    # Argument to process the different configuration files
    arg_parser.add_argument(
        '-cd', '--config-dir',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')

    # Additional argument to evaluate the resulting models
    arg_parser.add_argument(
        '-e', '--evaluate',
        dest='evaluate',
        metavar='C',
        default='false',
        help='Option to evaluate on test data')

    # Additional verbose argument
    arg_parser.add_argument(
        '-v', '--verbose',
        dest='verbose',
        metavar='C',
        default='true',
        help='Option for verbose mode'

    )

    args = arg_parser.parse_args()
    return args
