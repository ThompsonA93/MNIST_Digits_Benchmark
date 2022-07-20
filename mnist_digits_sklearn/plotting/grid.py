
import matplotlib.pyplot as plt


def data_to_grid(_num_classes, _images, _label):
    print("Creating Plot")
    i = 0
    for j in range(_num_classes):
        plt.subplot(5,5,j+1)
        plt.imshow(_images[i+j], cmap='binary')
        plt.title(_label[i+j])
        plt.axis('off')
    plt.show()