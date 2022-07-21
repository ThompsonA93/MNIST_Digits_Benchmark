import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

# Import dataset from Keras
# SKLEARN implements smaller training size per default. ( datasets.load_digits() )

training_size = 1000    
num_classes = 10        # 0 .. 9 
txt_out_file_path = 'sklearn-svm-parameter-tuning-log.txt'

def print_to_txt_file(*s):
    with open(txt_out_file_path, 'a') as f:
        for arg in s:
            print(arg, file=f)
            print(arg)



def start_sklearn_app(x_train, y_train, x_test, y_test):
    # Display Data
    ax = plt.subplots(1, num_classes, figsize=(20,20))
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
        #{"kernel": ["linear"], "C": [1, 10, 100, 1000]}, # bugged; Does not start the search
        {"kernel": ["poly"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        {"kernel": ["sigmoid"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        ]

    scores = ["precision", "recall"]
    for score in scores:
        print_to_txt_file("--- [%s] Running Parameter-Tests ---" % datetime.now())
        print_to_txt_file("Tuning parameters for criteria [%s]" % score)
        
        grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring="%s_macro" % score, verbose=2)
        grid.fit(x_train[0:training_size], y_train[0:training_size]) #grid search learning the best parameters
    

        print_to_txt_file("Best parameters set found on following development set:")
        print_to_txt_file("\tSupport Vector: %s" % grid.best_estimator_)
        print_to_txt_file("\tSupport Vector Parametrization: %s" % grid.best_params_)
        print_to_txt_file("\tAsserted Score: %s" % grid.best_score_)
        print_to_txt_file("Total Score \t\t Configurations")

        means = grid.cv_results_["mean_test_score"]
        stds = grid.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, grid.cv_results_["params"]):
            print_to_txt_file("%0.3f (+/-%0.03f)\t%r" % (mean, std, params))
        print_to_txt_file("")
        print("Wrote classifier comparisons to file ", txt_out_file_path)

        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")

        y_true, y_pred = y_test, grid.predict(x_test)
        print(classification_report(y_true, y_pred))
        print()



    

    
    
    print("grid done")
    print (grid.best_params_)

    #Now we train the best estimator in the full dataset
    print("training svm")
    best_svm = grid.best_estimator_
    best_svm.fit(x_train , y_train)
    print("svm done")


    print("Testing")
    print("score: ", best_svm.score(x_test, y_test,))