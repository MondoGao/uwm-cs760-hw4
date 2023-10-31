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

    nb_smooth = NaiveBayes(characters=characters, labels=labels, use_log_prob=True)
    nb_smooth.load_and_train(files=train_files)

    print(
        f"conditoinal prob. for Japanese (smoothed):\n{nb_smooth.char_prob[labels.index('j')]}\n"
    )
    print(
        f"conditoinal prob. for Spanish (smoothed):\n{nb_smooth.char_prob[labels.index('s')]}"
    )


if __name__ == "__main__":
    main()
