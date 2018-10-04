import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, file, learning_rate=0.2, iterations=100):
        self.testing_set = None
        self.training_set = None
        self.weights = []
        self.class_labels = []
        self.learning_rate = learning_rate
        self.file = file
        self.MAX_ITER = iterations

    def read_dataset(self):
        data = read_csv(self.file)
        self.class_labels = list(set(data['class']))

        for i in range(len(self.class_labels)):
            data.loc[data['class'] == self.class_labels[i], 'class'] = i

        self.training_set, self.testing_set = train_test_split(data, train_size=0.8, test_size=0.2)

        self.weights = [1 / (data.shape[1] + 1) for _ in range(data.shape[1] - 1)]

    def train(self):
        print("Training on {} values".format(self.training_set.shape[0]))
        epochs = self.MAX_ITER
        values = self.training_set.loc[:, 'class']
        self.training_set = self.training_set.drop(columns='class')

        while epochs != 0:
            for i in range(self.training_set.shape[0]):
                current_row = self.training_set.iloc[i]
                y = values.iloc[i]
                cost = np.dot(current_row, self.weights)
                threshold = 0
                predicted_value = -1
                if cost > threshold:
                    predicted_value = 1
                else:
                    predicted_value = 0
                error = y - predicted_value
                if error != 0:
                    for j in range(len(self.weights)):
                        self.weights[j] += self.learning_rate * current_row[j] * error
            epochs -= 1

    def test(self):
        print("Testing on {} values".format(self.testing_set.shape[0]))
        values = self.testing_set.loc[:, 'class']
        self.testing_set = self.testing_set.drop(columns='class')
        count = 0
        tp = 0.01
        tn = 0.01
        fp = 0.01
        fn = 0.01

        for i in range(self.testing_set.shape[0]):
            current_row = self.testing_set.iloc[i]
            cost = np.dot(self.weights, current_row)
            predicted_value = -1
            if cost > 0:
                predicted_value = 1
            else:
                predicted_value = 0
            if predicted_value == values.iloc[i]:
                count += 1
                if predicted_value == self.class_labels[1]:
                    tp += 1
                else:
                    tn += 1
            else:
                if predicted_value == self.class_labels[1]:
                    fp += 1
                else:
                    fn += 1

        print("Accuracy: {}%".format(float(count / self.testing_set.shape[0]) * 100))
        print("Precision:\n\t(+): {}\n\t(-): {}".format(float(tp / (tp + fp)), float(tp / (tp + fn))))
        print("Recall:\n\t(+): {}\n\t(-): {}".format(float(tn / (tn + fn)), float(tn / (tn + fp))))


def main():
    print('DATASET: IRIS')
    perceptron = Perceptron('datasets/IRIS.csv')
    perceptron.read_dataset()
    perceptron.train()
    perceptron.test()

    print('\nDATASET: SPECT')
    perceptron = Perceptron('datasets/SPECT.csv')
    perceptron.read_dataset()
    perceptron.train()
    perceptron.test()


if __name__ == '__main__':
    main()
