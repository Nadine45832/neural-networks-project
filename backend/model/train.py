import tensorflow as tf
import pandas as pd


def main():
    df_training = pd.read_csv("./data/Student data.csv")

    print(df_training.head())
    print(tf.__version__)


if __name__ == "__main__":
    main()
