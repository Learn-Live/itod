import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    help_ = "Load h5 ndm trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()

    print(f'args: {args}')
