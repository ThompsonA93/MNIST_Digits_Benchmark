from keras_neurnet import run_keras_neurnet
from sklearn_neurnet import run_sklearn_neurnet
from sklearn_svm import run_sklearn_svm

import matplotlib.pyplot as plt

from keras.datasets import mnist



# Entry point for mnist_digits_sklearn.__init__.py
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # x_train = training data as grayscale image data with shapes (60000, 28, 28)
        # y_train = Array of digit labels with shaoe (60000, ) for x_train
        # x_test = Test data as grayscale image data with shapes (10000, 28, 28)
        # y_test = Array of digit labels with shape (10000, ) for x_test

    # Visualize Samples
    num_classes = 10
    f, ax = plt.subplots(1, num_classes, figsize=(20,20))
    for i in range(0, num_classes):
        sample = x_train[y_train == i][0]
        ax[i].imshow(sample, cmap='gray')
        ax[i].set_title("Label: {}".format(i), fontsize=16)
        ax[i].axis('off')
    plt.show()

    # Reshape Data
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Run Tasks
    run_sklearn_svm(x_train, y_train, x_test, y_test)
    run_sklearn_neurnet(x_train, y_train, x_test, y_test)
    run_keras_neurnet(x_train, y_train, x_test, y_test)