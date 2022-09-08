### Packages
from datetime import datetime

import sys

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import seaborn as sns


### Configurations
# Training-Size
num_train = 15000                   # 60000 for full data set 
num_test  = 2500                    # 10000 for full data set

# Use GridSearchCV to look up optimal parameters (see below)
hyper_parameter_search = True       # True/False: Run hyper-parameter search via GridSearchCV. Takes a long time.


# Simple function to log information
txt_out_file_path = 'svm-parameter-tuning-log.txt'
def print_to_txt_file(*s):
    with open(txt_out_file_path, 'a') as f:
        for arg in s:
            print(arg, file=f)
            print(arg)

# Fetch MNIST-Data from Keras repository
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Display (Train) (Test) datasets
print("Data : Dataset Trainingset")
print(X_train.shape, X_test.shape)
print("Labels : Dataset Trainingset")
print(y_train.shape, y_test.shape)

# i.e.: We have 60000 images with a size of 28x28 pixels

# Visualize some examples
num_classes = 10 # 0 .. 9
f, ax = plt.subplots(1, num_classes, figsize=(20,20))
for i in range(0, num_classes):
  sample = X_train[y_train == i][0]
  ax[i].imshow(sample, cmap='gray')
  ax[i].set_title("Label: {}".format(i), fontsize=16)
  ax[i].axis('off')

# Reshape the data such that we have access to every pixel of the image
# The reason to access every pixel is that only then we can apply deep learning ideas and can assign color code to every pixel.
train_data = X_train.reshape((X_train.shape[0], 28*28)).astype('float32')
train_label = y_train.astype("float32")

test_data = X_test.reshape((X_test.shape[0], 28*28)).astype('float32')
test_label = y_test.astype("float32")

# We know the RGB color code where different values produce various colors. It is also difficult to remember every color combination. 
# We already know that each pixel has its unique color code and also we know that it has a maximum value of 255. 
# To perform Machine Learning, it is important to convert all the values from 0 to 255 for every pixel to a range of values from 0 to 1.
train_data = train_data / 255
test_data = test_data / 255

# As an optional step, we decrease the training and testing data size, such that the algorithms perform their execution in acceptable time
train_data = train_data[1:num_train,]
train_label = train_label[1:num_train]

test_data = test_data[1:num_test,]
test_label = test_label[1:num_test]

# Display (Train) (Test) datasets
print("Reshaped Data : Dataset Trainingset")
print(train_data.shape, test_data.shape)
print("Reshaped Labels : Dataset Trainingset")
print(train_label.shape, test_label.shape)

# As we can see: We now have X images with 784 pixels in total
# We now operate on this data

# The default layout of svm.svc() 
# @see https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
svm = SVC(
    C=1.0,                          # Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
    kernel='linear',                # Specifies the kernel type to be used in the algorithm. 
    degree=3,                       # Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
    gamma='scale',                  # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    coef0=0.0,                      # Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
    shrinking=True,                 # Whether to use the shrinking heuristic. 
    probability=False,              # Whether to enable probability estimates. 
    tol=0.001,                      # Tolerance for stopping criterion.
    cache_size=200,                 # Specify the size of the kernel cache (in MB).
    class_weight=None,              # Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. 
    verbose=False,                  # Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.
    max_iter=-1,                    # Hard limit on iterations within solver, or -1 for no limit.
    decision_function_shape='ovr',  # Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2).
    break_ties=False,               # If true, decision_function_shape='ovr', and number of classes > 2, predict will break ties according to the confidence values of decision_function;
    random_state=None               # Controls the pseudo random number generation for shuffling the data for probability estimates.
)

# Evalute SVM.SVC with parameters on data below
svm = SVC(
    C=1.0, 
    kernel='linear', 
    degree=3, 
    gamma='scale', 
    coef0=0.0, 
    shrinking=True, 
    probability=False, 
    tol=0.001, 
    cache_size=200, 
    class_weight=None,
    verbose=False, 
    max_iter=-1, 
    decision_function_shape='ovr', 
    break_ties=False, 
    random_state=None    
)
svm.fit(train_data, train_label)
print("Mean accuracy on train data: ", svm.score(train_data, train_label))   # Mean Accuracy on the given training data and labels
print("Mean accuracy on test data: ", svm.score(test_data, test_label))      # Mean Accuracy on the given test data and labels
# Hyperparameter search -- Takes up a long time.
if hyper_parameter_search:
    svm = SVC()
    parameters = {
            "kernel":["linear"], 
            "C":[1,10,100],                
            "shrinking":[True,False],      
            "probability":[True,False], 
            "tol":[0.01,0.001,0.0001],
    }
    scores = [
        'accuracy',
        #"precision",    # The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
        #"recall",       # The recall is intuitively the ability of the classifier to find all the positive samples.
        ]
    for score in scores:
        print_to_txt_file("--- [%s] Running Parameter-Tests [LINEAR-SVC] ---" % datetime.now())
        print_to_txt_file("Tuning parameters for criteria [%s]" % score)
        # FIXME: Doesn't take accuracy as score for some reason. Refer to line below for accuracy score
        #grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring="%s_macro" % score, verbose=3)
        grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring='accuracy', verbose=3)
        grid.fit(train_data, train_label)

        print_to_txt_file("Best parameters set found on following development set:")
        print_to_txt_file("\tSupport Vector: %s" % grid.best_estimator_)
        print_to_txt_file("\tSupport Vector Parametrization: %s" % grid.best_params_)
        print_to_txt_file("\tAsserted Score: %s" % grid.best_score_)
        print_to_txt_file("Total Score \t\t Configurations")

        means = grid.cv_results_["mean_test_score"]
        stds = grid.cv_results_["std_test_score"]
        params = grid.cv_results_["params"]
        for mean, std, params in zip(means, stds, params):
            print_to_txt_file("%0.3f (+/-%0.03f)\t%r" % (mean, std, params))
        print("Wrote classifier comparisons to file ", txt_out_file_path)

        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
    
        y_true, y_pred = test_label, grid.predict(test_data)
        print(classification_report(y_true, y_pred))
        print()



# Eval SVM on Training Data
svm = SVC(
    C=1.0, 
    kernel='poly', 
    degree=3, 
    gamma='scale', 
    coef0=0.0, 
    shrinking=True, 
    probability=False, 
    tol=0.001, 
    cache_size=200, 
    class_weight=None,
    verbose=False, 
    max_iter=-1, 
    decision_function_shape='ovr', 
    break_ties=False, 
    random_state=None    
)
svm.fit(train_data, train_label)
print("Mean accuracy on train data: ", svm.score(train_data, train_label))   # Mean Accuracy on the given training data and labels
print("Mean accuracy on test data: ", svm.score(test_data, test_label))      # Mean Accuracy on the given test data and labels

# Hyperparameter search -- Takes up a long time.
if hyper_parameter_search:
    svm = SVC()
    parameters = {
            "kernel":["poly"], 
            "C":[1,10,100],
            "gamma":["scale", "auto"],
            "coef0":[0.0, 0.5],
            "degree":[3,5,10],                
            "shrinking":[True,False],      
            "probability":[True,False], 
            "tol":[0.01,0.001,0.0001],
    }
    scores = [
        "accuracy",
        #"precision",    # The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
        #"recall",       # The recall is intuitively the ability of the classifier to find all the positive samples.
        ]
    for score in scores:
        print_to_txt_file("--- [%s] Running Parameter-Tests [POLY-SVC] ---" % datetime.now())
        print_to_txt_file("Tuning parameters for criteria [%s]" % score)
        # FIXME: Doesn't take accuracy as score for some reason. Refer to line below for accuracy score
        #grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring="%s_macro" % score, verbose=3)
        grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring='accuracy', verbose=3)
        grid.fit(train_data, train_label)

        print_to_txt_file("Best parameters set found on following development set:")
        print_to_txt_file("\tSupport Vector: %s" % grid.best_estimator_)
        print_to_txt_file("\tSupport Vector Parametrization: %s" % grid.best_params_)
        print_to_txt_file("\tAsserted Score: %s" % grid.best_score_)
        print_to_txt_file("Total Score \t\t Configurations")

        means = grid.cv_results_["mean_test_score"]
        stds = grid.cv_results_["std_test_score"]
        params = grid.cv_results_["params"]
        for mean, std, params in zip(means, stds, params):
            print_to_txt_file("%0.3f (+/-%0.03f)\t%r" % (mean, std, params))
        print("Wrote classifier comparisons to file ", txt_out_file_path)

        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
    
        y_true, y_pred = test_label, grid.predict(test_data)
        print(classification_report(y_true, y_pred))
        print()

# Eval SVM on Training Data
svm = SVC(
    C=1.0, 
    kernel='rbf', 
    degree=3, 
    gamma='scale', 
    coef0=0.0, 
    shrinking=True, 
    probability=False, 
    tol=0.001, 
    cache_size=200, 
    class_weight=None,
    verbose=False, 
    max_iter=-1, 
    decision_function_shape='ovr', 
    break_ties=False, 
    random_state=None    
)
svm.fit(train_data, train_label)
print("Mean accuracy on train data: ", svm.score(train_data, train_label))   # Mean Accuracy on the given training data and labels
print("Mean accuracy on test data: ", svm.score(test_data, test_label))      # Mean Accuracy on the given test data and labels

# Hyperparameter search -- Takes up a long time.
if hyper_parameter_search:
    svm = SVC()
    parameters = {
            "kernel":["rbf"], 
            "C":[1,10,100],           
            "gamma":["scale", "auto"],     
            "shrinking":[True,False],      
            "probability":[True,False], 
            "tol":[0.01,0.001,0.0001],
    }
    scores = [
        "accuracy",
        #"precision",    # The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
        #"recall",       # The recall is intuitively the ability of the classifier to find all the positive samples.
        ]
    for score in scores:
        print_to_txt_file("--- [%s] Running Parameter-Tests [RBF-SVC] ---" % datetime.now())
        print_to_txt_file("Tuning parameters for criteria [%s]" % score)
        # FIXME: Doesn't take accuracy as score for some reason. Refer to line below for accuracy score
        #grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring="%s_macro" % score, verbose=3)
        grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring='accuracy', verbose=3)
        grid.fit(train_data, train_label)

        print_to_txt_file("Best parameters set found on following development set:")
        print_to_txt_file("\tSupport Vector: %s" % grid.best_estimator_)
        print_to_txt_file("\tSupport Vector Parametrization: %s" % grid.best_params_)
        print_to_txt_file("\tAsserted Score: %s" % grid.best_score_)
        print_to_txt_file("Total Score \t\t Configurations")

        means = grid.cv_results_["mean_test_score"]
        stds = grid.cv_results_["std_test_score"]
        params = grid.cv_results_["params"]
        for mean, std, params in zip(means, stds, params):
            print_to_txt_file("%0.3f (+/-%0.03f)\t%r" % (mean, std, params))
        print("Wrote classifier comparisons to file ", txt_out_file_path)

        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
    
        y_true, y_pred = test_label, grid.predict(test_data)
        print(classification_report(y_true, y_pred))
        print()

# Eval SVM on Training Data
svm = SVC(
    C=1.0, 
    kernel='sigmoid', 
    degree=3, 
    gamma='scale', 
    coef0=0.0, 
    shrinking=True, 
    probability=False, 
    tol=0.001, 
    cache_size=200, 
    class_weight=None,
    verbose=False, 
    max_iter=-1, 
    decision_function_shape='ovr', 
    break_ties=False, 
    random_state=None    
)
svm.fit(train_data, train_label)
print("Mean accuracy on train data: ", svm.score(train_data, train_label))   # Mean Accuracy on the given training data and labels
print("Mean accuracy on test data: ", svm.score(test_data, test_label))      # Mean Accuracy on the given test data and labels

# Hyperparameter search -- Takes up a long time.
if hyper_parameter_search:
    svm = SVC()
    parameters = {
            "kernel":["sigmoid"], 
            "C":[1,10,100],            
            "gamma":["scale", "auto"],    
            "shrinking":[True,False],      
            "probability":[True,False], 
            "tol":[0.01,0.001,0.0001],
    }
    scores = [
        "accuracy",
        #"precision",    # The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
        #"recall",       # The recall is intuitively the ability of the classifier to find all the positive samples.
        ]
    for score in scores:
        print_to_txt_file("--- [%s] Running Parameter-Tests [RBF-SVC] ---" % datetime.now())
        print_to_txt_file("Tuning parameters for criteria [%s]" % score)
        # FIXME: Doesn't take accuracy as score for some reason. Refer to line below for accuracy score
        #grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring="%s_macro" % score, verbose=3)
        grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring='accuracy', verbose=3)
        grid.fit(train_data, train_label)

        print_to_txt_file("Best parameters set found on following development set:")
        print_to_txt_file("\tSupport Vector: %s" % grid.best_estimator_)
        print_to_txt_file("\tSupport Vector Parametrization: %s" % grid.best_params_)
        print_to_txt_file("\tAsserted Score: %s" % grid.best_score_)
        print_to_txt_file("Total Score \t\t Configurations")

        means = grid.cv_results_["mean_test_score"]
        stds = grid.cv_results_["std_test_score"]
        params = grid.cv_results_["params"]
        for mean, std, params in zip(means, stds, params):
            print_to_txt_file("%0.3f (+/-%0.03f)\t%r" % (mean, std, params))
        print("Wrote classifier comparisons to file ", txt_out_file_path)

        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
    
        y_true, y_pred = test_label, grid.predict(test_data)
        print(classification_report(y_true, y_pred))
        print()