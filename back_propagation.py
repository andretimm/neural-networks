import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split

learning_rate = 0.33
epochs = 100
class_labels = []
hidden_nodes = 5


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


def train(setX, setY, testX, testY):
    global learning_rate
    global epochs
    global hidden_nodes
    weights_i_to_h = [[1 / (setX.shape[1] * 5) for _ in range(hidden_nodes)] for i in range(setX.shape[1])]
    weights_h_to_o = [0.2 for _ in range(hidden_nodes)]
    bias1 = [1 / 6 for _ in range(hidden_nodes)]
    bias2 = 1 / 6

    while epochs != 0:
        for i in range(setX.shape[0]):
            current_row = np.array(list(setX.iloc[i]))
            current_y = setY.iloc[i]

            output1 = sigmoid(np.dot(current_row, weights_i_to_h) + np.array(bias1))
            output2 = sigmoid(np.dot(output1, weights_h_to_o) + np.array(bias2))

            if output2 >= 0.5:
                predicted_value = 1
            else:
                predicted_value = 0

            output_error = output2 * (1 - output2) * (current_y - predicted_value)
            hidden_error = np.dot(((output1 * output_error) * np.array(1 - np.array(output1))), weights_h_to_o)

            weights_h_to_o = np.array(weights_h_to_o) + learning_rate * output2 * np.array(output_error)
            weights_i_to_h = np.array(weights_i_to_h) + np.reshape(learning_rate * np.array(output1) * hidden_error,
                                                                   (-1, hidden_nodes))
            bias1 = np.array(bias1) + learning_rate * np.array(hidden_error)
            bias2 = np.array(bias2) + learning_rate * np.array(output_error)
        epochs -= 1

    return test(testX, testY, weights_i_to_h, weights_h_to_o, bias1, bias2)


def test(setX, setY, weights_i_to_h, weights_h_to_o, bias1, bias2):
    global learning_rate
    global hidden_nodes
    count = 0
    tp = 0.01
    tn = 0.01
    fp = 0.01
    fn = 0.01
    for i in range(setX.shape[0]):
        current_row = np.array(list(setX.iloc[i]))
        current_y = setY.iloc[i]

        output1 = sigmoid(np.dot(current_row, weights_i_to_h) + np.array(bias1))
        output2 = sigmoid(np.dot(output1, weights_h_to_o) + np.array(bias2))

        if output2 >= 0.5:
            predicted_value = 1
        else:
            predicted_value = 0
        error = current_y - predicted_value
        if error == 0:
            count = count + 1
        if predicted_value == current_y and predicted_value == 1:
            tp += 1
        elif predicted_value != current_y and predicted_value == 1:
            fp += 1
        elif predicted_value != current_y and predicted_value == 0:
            fn += 1
        else:
            tn += 1

    return float(count / setX.shape[0]) * 100, float(tp / (tp + fp)), float(tn / (tn + fn)), float(tp / (tp + fn)), \
        float(tn / (tn + fp))


def main():
    file = input("Enter the file name: ")
    data = read_csv("datasets/" + file)
    labels = list(set(data['class']))

    for i in range(len(labels)):
        data.loc[data['class'] == labels[i], 'class'] = i

    x = data.iloc[:, :data.shape[1] - 1]
    y = data.iloc[:, data.shape[1] - 1]
    accuracy_sum = 0
    positive_precision_sum = 0
    negative_precision_sum = 0
    positive_recall_sum = 0
    negative_recall_sum = 0

    for i in range(10):
        train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, test_size=0.2)

        acc, pos_pre, neg_pre, pos_rec, neg_rec = train(train_x, train_y, test_x, test_y)
        accuracy_sum += acc
        positive_precision_sum += pos_pre
        negative_precision_sum += neg_pre
        positive_recall_sum += pos_rec
        negative_recall_sum += neg_rec

    print("Accuracy: {}%".format(accuracy_sum / 10))
    print("Precision:\n\t(+): {}\n\t(-): {}".format(positive_precision_sum / 10, negative_precision_sum / 10))
    print("Recall:\n\t(+): {}\n\t(-): {}".format(positive_recall_sum / 10, negative_recall_sum / 10))


if __name__ == '__main__':
    main()
