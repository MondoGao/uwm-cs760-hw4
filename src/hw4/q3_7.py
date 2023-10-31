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

    # [real, pred]
    confusion_matrix = np.zeros((len(labels), len(labels)))

    for label_idx, label in enumerate(labels):
        for num in range(10, 20):
            pred = nb.predict_by_file(f"{data_dir}{label}{num}.txt")
            print(
                f"prediction for {label}{num}.txt: {pred}, log prob: {nb.predict_probs}   "
            )
            confusion_matrix[label_idx][labels.index(pred)] += 1
    
    print(f"confusion matrix: \n      pred {',   '.join(labels)}")
    for actual_idx, pred_arr in enumerate(confusion_matrix):
        print(f"actual {labels[actual_idx]}: {pred_arr}")


if __name__ == "__main__":
    main()
