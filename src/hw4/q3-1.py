from hw4.naive_bayes import NaiveBayes
import numpy as np
import os


def main():
    characters = [" "] + [chr(i) for i in range(ord("a"), ord("z") + 1)]
    labels = ["English", "Spanish", "Japanese"]
    short_labels = ["e", "s", "j"]
    data_dir = "./data/"
    train_files = (
        [f"{data_dir}{short_labels[0]}{x}.txt" for x in range(0, 10)]
        + [f"{data_dir}{short_labels[1]}{x}.txt" for x in range(0, 10)]
        + [f"{data_dir}{short_labels[2]}{x}.txt" for x in range(0, 10)]
    )

    nb = NaiveBayes(characters=characters, labels=short_labels, use_log_prob=True)
    nb.load_and_train(files=train_files)
    X_test = nb.file_to_characters(f"{data_dir}e10.txt")
    nb.predict(X_test=X_test[1:])


if __name__ == "__main__":
    main()
