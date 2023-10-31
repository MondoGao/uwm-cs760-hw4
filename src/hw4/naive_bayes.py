import os
import numpy as np
from dataclasses import dataclass, field


@dataclass
class NaiveBayes:
    characters: list[str]
    labels: list[str]
    use_log_prob: bool = True
    data: np.ndarray = field(default=None, init=False)

    def load_and_train(self, files: list[str]):
        self.data = np.array(
            [self.file_to_characters(file_path) for file_path in files]
        )
        self.fit(self.data)

    def file_to_characters(self, file_path):
        """
        Loads data file and transforms to charactor vector.
        Returns [label_idx, space_count, a_count, ..., z_count]
        """
        char_count = np.zeros(len(self.characters), dtype=int)
        with open(file_path, "r") as f:
            # ignore \n
            document = f.read().replace("\n", "")
            label = os.path.basename(f.name)[0]

        for character in document:
            char_idx = self.characters.index(character)
            char_count[char_idx] += 1

        label_idx = self.labels.index(label)
        return np.concatenate(([label_idx], char_count))

    def fit(self, data):
        label_count = np.zeros(len(self.labels))
        char_count_by_label = np.zeros((len(self.labels), len(self.characters)))

        for d in data:
            label_idx = d[0]
            label_count[label_idx] += 1

            char_count_by_label[label_idx, :] += d[1:]

        # prior
        self.label_prob = label_count / np.sum(label_count)

        char_sum = np.sum(char_count_by_label, axis=1)
        self.char_prob = char_count_by_label / char_sum.reshape(-1, 1)

    # Define the prediction function
    def predict(self, X_test):
        probs = np.zeros(len(self.labels))

        for label_idx, label in enumerate(self.labels):
            # if self.use_log_prob:
            # probs[label_idx] = np.sum(
            #     X_test * np.log(self.char_prob[label_idx, :])
            #     + (1 - X_test) * np.log(1 - self.char_prob[label_idx, :])
            # ) + np.log(self.label_prob[label_idx])
            # else:
            probs[label_idx] = (
                np.prod(
                    X_test * self.char_prob[label_idx, :]
                    + (1 - X_test) * (1 - self.char_prob[label_idx, :])
                )
                * self.label_prob[label_idx]
            )

        pred = self.labels[np.argmax(probs)]
        print(f"probs: {probs}, pred: {pred}")
        return pred
