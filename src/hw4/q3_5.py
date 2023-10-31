from hw4.naive_bayes import NaiveBayes
import numpy as np
import os


def main():
    # differ from 1, move space to end
    characters = [chr(i) for i in range(ord("a"), ord("z") + 1)] + [" "]
    labels = ["e", "s", "j"]
    data_dir = "./data/"
    train_files = (
        [f"{data_dir}{labels[0]}{x}.txt" for x in range(0, 10)]
        + [f"{data_dir}{labels[1]}{x}.txt" for x in range(0, 10)]
        + [f"{data_dir}{labels[2]}{x}.txt" for x in range(0, 10)]
    )

    nb = NaiveBayes(characters=characters, labels=labels, use_log_prob=True)
    nb.load_and_train(files=train_files)
    nb.predict_by_file(f"{data_dir}e10.txt")

    for label_idx, label in enumerate(labels):
        print(f"p(x_i | {label}): {nb.char_prob[label_idx]}")
        print(f"\hat p(x | {label}): {nb.likelihood[label_idx]}\n")


if __name__ == "__main__":
    main()
