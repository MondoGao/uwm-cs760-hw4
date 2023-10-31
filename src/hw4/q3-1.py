# from hw4.naive_bayes import NaiveBayes
import numpy as np
import os


def main():
    charactors = [" "] + [chr(i) for i in range(ord("a"), ord("z") + 1)]
    labels = ["English", "Spanish", "Japanese"]
    short_labels = ["e", "s", "j"]
    data_dir = "./data/"
    train_files = (
        [f"{data_dir}{short_labels[0]}{x}.txt" for x in range(0, 10)]
        + [f"{data_dir}{short_labels[1]}{x}.txt" for x in range(0, 10)]
        + [f"{data_dir}{short_labels[2]}{x}.txt" for x in range(0, 10)]
    )
    data = [file_to_charactors(file_path, charactors) for file_path in train_files]
    print(data)


def file_to_charactors(file_path, charactors):
    """
    Loads data file and transforms to charactor vector.
    Returns [label, space_count, a_count, ..., z_count]
    """
    char_count = np.zeros(len(charactors))

    with open(file_path, "r") as f:
        document = f.read().replace("\n", "")
        label = os.path.basename(f.name)[0]
    for character in document:
        char_count[charactors.index(character)] += 1

    return [label] + [char_count]


if __name__ == "__main__":
    main()
