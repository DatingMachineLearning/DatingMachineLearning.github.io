import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


def look_data(data, length=5):
    print(data.shape)
    print(data[:length])


def load_mnist(target='5'):
    memory = joblib.Memory('../view')
    fetch_openml_cached = memory.cache(fetch_openml)
    mnist_dataset = fetch_openml_cached('mnist_784', as_frame=True)
    X = mnist_dataset['data'].values
    y = np.where(mnist_dataset['target'].values == target, 1, 0).reshape(70000, 1)
    return X, y


def norm(data, training_ratio=0.8):
    # 归一化处理
    # maximum, minimum = np.max(training_data, axis=0, keepdims=True), np.min(training_data, axis=0, keepdims=True)
    # average = np.sum(training_data, axis=0, keepdims=True) / training_data.shape[0]

    # 利用训练集的最大最小归一化测试集
    # data = (data - minimum) / (maximum - minimum)
    mid = 127.5
    data = (data - mid) / mid
    return data


class Network:

    def __init__(self, num_of_weights: tuple):
        # np.random.seed(0)
        self.W = np.random.normal(0, 1, num_of_weights)
        self.Z = None
        self.A = None
        self.ac_name = None
        self.dZ = None
        self.dA = None

    def save_file(self, file_name):
        joblib.dump(self, file_name)

    @staticmethod
    def read_file(file_name):
        return joblib.load(file_name)

    def forward(self, A, ac_name='sigmoid'):
        self.Z = np.dot(self.W, A)
        self.ac_name = ac_name
        if ac_name == 'ReLU':
            self.A = Network.ReLU(self.Z)
        elif ac_name[:7] == 'sigmoid':
            self.A = Network.sigmoid(self.Z)
        else:
            print("None of the activation_name.")

    def backpropagation(self, input_layer, y, eta=0.001):
        if self.ac_name == "ReLU":
            self.dZ = self.dA * Network.ReLU(self.A)
        elif self.ac_name == 'sigmoid':
            p = Network.sigmoid_derivative(self.A)
            self.dZ = self.dA * p
        elif self.ac_name == 'sigmoid_CE':
            self.dZ = Network.sigmoid_CE_derivative(self.A, y)
        else:
            print("None of the activation_name.")
        dW = np.dot(self.dZ, input_layer.A.T) / y.shape[0]
        input_layer.dA = np.dot(self.W.T, self.dZ)
        self.W -= eta * dW

    def get_loss(self, y):
        assert len(self.A) == len(y)
        cross_entropy = Network.cross_entropy(self.A, y)
        return np.sum(cross_entropy, axis=0) / y.shape[0]

    @staticmethod
    def cross_entropy(y_hat, y):
        return np.where(y == 1, -np.log(y_hat), -np.log(1 - y_hat))

    @staticmethod
    def ReLU(z):
        return np.maximum(z, 0)

    @staticmethod
    def ReLU_derivative(z):
        return np.where(z >= 0, 1, 0)

    @staticmethod
    def sigmoid(z):
        return 1. / (1. + np.exp(-z))

    @staticmethod
    def sigmoid_CE_derivative(a, y):
        return a - y

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)

    @staticmethod
    def train(X_train, y_train, training_offset, num_epochs, batch_size, eta):
        hidden_layer1 = Network((10, 784))
        hidden_layer2 = Network((10, 10))
        out_layer = Network((1, 10))

        for epoch_id in range(num_epochs):
            X_train, y_train = shuffle_in_unison(X_train, y_train)
            mini_batches = [(k, k + batch_size) for k in range(0, training_offset, batch_size)]

            for iter_id, mini_batch in enumerate(mini_batches):
                start, end = mini_batch
                X_batch = X_train[start: end, :].T
                y_batch = np.int32(y_train[start: end, :]).reshape(X_batch.shape[1], 1).T

                hidden_layer1.forward(X_batch, ac_name="sigmoid")
                hidden_layer2.forward(hidden_layer1.A, ac_name="sigmoid")
                out_layer.forward(hidden_layer2.A, ac_name="sigmoid_CE")

                print("epoch_id: {0} , iter_id: {1}, {2}".format(
                    epoch_id, iter_id, out_layer.get_loss(y_batch)))

                init_net = Network((1, 1))
                init_net.A = X_batch
                out_layer.dA = 1
                out_layer.backpropagation(input_layer=hidden_layer2,
                                          y=y_batch, eta=eta)
                hidden_layer2.backpropagation(input_layer=hidden_layer1,
                                              y=y_batch, eta=eta)
                hidden_layer1.backpropagation(input_layer=init_net,
                                              y=y_batch, eta=eta)
        hidden_layer1.save_file("hidden_layer1")
        hidden_layer2.save_file("hidden_layer2")
        out_layer.save_file("out_layer")

        return hidden_layer1, hidden_layer2, out_layer

    @staticmethod
    def predict(X_test, y_test, hidden_layer1, hidden_layer2, out_layer):
        X_test = X_test.T
        result = []
        for i in range(X_test.shape[1]):
            x = X_test[:, i].reshape(X_test.shape[0], 1)
            hidden_layer1.forward(x, ac_name="sigmoid")
            hidden_layer2.forward(hidden_layer1.A, ac_name="sigmoid")
            out_layer.forward(hidden_layer2.A, ac_name="sigmoid")
            result.append(int(out_layer.A[0][0] > 0.5))
            # y_test[0, i]
        result = np.array(result).reshape(len(result), 1)
        result = np.where(result == y_test, 1, 0)
        accuracy = np.sum(result) / result.shape[0]
        print("accuracy = {:.2%}".format(accuracy))


def draw(row):
    digit = row[1:].reshape(28, 28)
    plt.imshow(digit, cmap="binary")
    plt.show()


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def main():
    X_src, y_src = load_mnist('4')

    # 划分训练集
    training_ratio = 0.8
    training_offset = int(y_src.shape[0] * training_ratio)
    X_src = norm(X_src)
    X_train, X_test = X_src[:training_offset], X_src[training_offset:]
    y_train, y_test = y_src[:training_offset], y_src[training_offset:]

    hidden_layer1, hidden_layer2, out_layer = Network.train(
        X_train=X_train, y_train=y_train,
        num_epochs=5, batch_size=200, eta=0.01,
        training_offset=training_offset)

    # hidden_layer1 = Network.read_file("hidden_layer1")
    # hidden_layer2 = Network.read_file("hidden_layer2")
    # out_layer = Network.read_file("out_layer")

    test_num = -2
    Network.predict(X_test[:test_num], y_test[:test_num], hidden_layer1, hidden_layer2, out_layer)


if __name__ == '__main__':
    main()
    #
    # test = np.array([[1, 2, 3]])
    # test1 = np.array([[1], [2], [3]])
    # print(test.dot(test1))
