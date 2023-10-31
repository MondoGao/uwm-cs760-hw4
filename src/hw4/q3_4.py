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
    # nb_smooth.load_and_train(files=train_files)
    bow = nb.file_to_characters(f"{data_dir}e10.txt")

    print(f"e10.txt bag-of-words vector: \n{bow[1:]} (length: {len(bow[1:])})\n")


if __name__ == "__main__":
    main()
