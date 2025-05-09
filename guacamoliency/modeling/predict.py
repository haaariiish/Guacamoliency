from pathlib import Path

from tqdm import tqdm



import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default='moses',
                    help="on which datasets to use for the training", required=True)
    parser.add_argument('--output_dir', type = str, default='reports',
                        help="where save our outputs", required=False)
    args = parser.parse_args()


if __name__ == "__main__":
    main()
