import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from mnist_digits_sklearn.plotting.cmd import data_to_cmd

from .classification.svc_classifier_comparison import run_svc_classifier_comparison
from .plotting.grid import data_to_grid

c = 10.0
gamma = 0.001
kernel = 'rbf'

def start_app():
    num_classes = 10 # 0 .. 9
    digits = datasets.load_digits()
    data_to_grid(num_classes, digits.images, digits.target)

    X = digits.data
    y = digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    
    run_svc_classifier_comparison(X_train, y_train)






    # SVC Creation
    clf = svm.SVC(C=c, gamma=gamma, kernel=kernel)
    clf.fit(X_train, y_train)

    # Predict Test samples
    predicted = clf.predict(X_test)

    _, axes = plt.subplots(nrows=1, ncols=6, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8,8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    
    data_to_cmd(y_test, predicted)


    print(f"Classification report for classifier {clf}: \n", f"{metrics.classification_report(y_test, predicted)}") 
    plt.show()

