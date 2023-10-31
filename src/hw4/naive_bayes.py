import os
import numpy as np
from dataclasses import dataclass, field


@dataclass
class NaiveBayes:
    characters: list[str]
    labels: list[str]
    use_log_prob: bool = True
    use_smoothing: bool = True
    smooth_param: float = 1/2
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
        self.label_count = np.zeros(len(self.labels))
        self.char_count_by_label = np.zeros((len(self.labels), len(self.characters)))

        for d in data:
            label_idx = d[0]
            self.label_count[label_idx] += 1

            self.char_count_by_label[label_idx, :] += d[1:]

        if self.use_smoothing:
            self.label_count += self.smooth_param

        # prior
        self.label_prob = self.label_count / np.sum(self.label_count)

        if self.use_smoothing:
            for label_idx in range(len(self.labels)):
                self.char_count_by_label[label_idx, :] += self.smooth_param

        char_sum = np.sum(self.char_count_by_label, axis=1)
        self.char_prob = self.char_count_by_label / char_sum.reshape(-1, 1)

    def predict_by_file(self, file_path):
        X_test = self.file_to_characters(file_path)
        return self.predict(X_test=X_test[1:])

    def predict(self, X_test):
        probs = np.zeros(len(self.labels))

        for label_idx, label in enumerate(self.labels):
            probs[label_idx] = self.label_prob[label_idx]
            if self.use_log_prob:
                probs[label_idx] = np.log(probs[label_idx])

            for char_idx, char_num in enumerate(X_test):
                # multinomial
                if self.use_log_prob:
                    probs[label_idx] += char_num * np.log(
                        self.char_prob[label_idx, char_idx]
                    )
                else:
                    probs[label_idx] *= self.char_prob[label_idx, char_idx] ** char_num

        pred = self.labels[np.argmax(probs)]
        return pred
