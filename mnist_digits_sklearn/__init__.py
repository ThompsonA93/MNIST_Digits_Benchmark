import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Import dataset from Keras
# SKLEARN implements smaller training size per default. ( datasets.load_digits() )
from keras.datasets import mnist

def start_app():
    # Load Data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Display Data
    num_classes = 10 # 0 .. 9
    f, ax = plt.subplots(1, num_classes, figsize=(20,20))
    for i in range(0, num_classes):
        sample = x_train[y_train == i][0]
        ax[i].imshow(sample, cmap='gray')
        ax[i].set_title("Label: {}".format(i), fontsize=16)
        ax[i].axis('off')
    plt.show()



    # PREPROCESSING DATA
    # Change from matrix to array --> dimension 28x28 to array of dimention 784
    x_train = x_train.reshape(60000, 784).astype('float32') / 255.0*100 - 50
    x_test = x_test.reshape(10000, 784).astype('float32') / 255.0*100 - 50
    print(x_train.shape[0], 'Train samples,', x_test.shape[0], 'Test samples')

    #I mod2 the label sets to have a result of 1 or 0 , for odds and evens respectively
    y_train = y_train % 2
    y_test = y_test % 2
        







    # PCA
    pca = PCA(n_components=50)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)











    #                         GRID SEARCH FOR PARAMETER OPTIMIZING
    svm = SVC()
    parameters = [
        {"kernel": ["linear"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        #{"kernel": ["poly"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        #{"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        #{"kernel": ["sigmoid"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        ]
    print("grid search")

    grid = GridSearchCV(estimator=svm, param_grid=parameters, verbose=3)
    print("grid.fit")
    #grid.fit(x_train[0:7000], y_train[0:7000]) #grid search learning the best parameters
    grid.fit(x_train, y_train) #grid search learning the best parameters
    #clf.fit(_X_train, _y_train)
    
    
    

    
    
    print("grid done")
    print (grid.best_params_)

    #Now we train the best estimator in the full dataset
    print("training svm")
    best_svm = grid.best_estimator_
    best_svm.fit(x_train , y_train)
    print("svm done")


    print("Testing")
    print("score: ", best_svm.score(x_test, y_test,))