#
# This file merely acts as an python-native substitue for the IPYNB (Jupyter) files, for the sake of automation and portability
#


### Packages
from datetime import datetime
import time
import os

import matplotlib.pyplot as plt
#%matplotlib inline

from keras.datasets import mnist

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import config

### Configurations
# Training-Size
num_train = config.num_train                   # 60000 for full data set 
num_test  = config.num_test                    # 10000 for full data set

# Simple function to log information
path = os.getcwd()+"/log"
logDir = os.path.exists(path)
if not logDir:
    os.makedirs(path)

training_results = path+"/svm-training-log.txt"
def log_training_results(*s):
    with open(training_results, 'a') as f:
        for arg in s:
            print(arg, file=f)
            print(arg)

hyperparameter_search_log = path+"/svm-hyperparameter-tuning-log.txt"
def log_hyperparameter_search(*s):
    with open(hyperparameter_search_log, 'a') as f:
        for arg in s:
            print(arg, file=f)
            print(arg)

print("Generated data will be located in ", training_results, hyperparameter_search_log)
log_training_results("[%s] on (%s, %s) using (Train: %s, Test: %s)" % (datetime.now(), config.os, config.cpu, config.num_train, config.num_test))
if config.hyper_parameter_search:
    log_hyperparameter_search("[%s] on (%s, %s) using (Train: %s, Test: %s)" % (datetime.now(), config.os, config.cpu, config.num_train, config.num_test))


# Fetch MNIST-Data from Keras repository
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Display (Train) (Test) datasets
print("Shape of training data:\t\t", X_train.shape)
print("Shape of training labels:\t", y_train.shape)
print("Shape of testing data:\t\t", X_test.shape)
print("Shape of testing labels:\t", y_test.shape)

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

print("Reshaped training data:\t\t", train_data.shape)
print("Reshaped training labels:\t", train_label.shape)
print("Reshaped testing data:\t\t", test_data.shape)
print("Reshaped testing labels:\t", test_label.shape)

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
svm = SVC(kernel='linear')
start_time = time.time()
svm.fit(train_data, train_label)
end_time = time.time() - start_time
log_training_results("Trained new model: {'Kernel':'%s'} in %s seconds" % (svm.get_params()["kernel"], end_time))

start_time = time.time()
score = svm.score(train_data, train_label)
end_time = time.time() - start_time
log_training_results("\t[%s]: mean accuracy on train-data: %s; execution time: %ss" % (svm.get_params()["kernel"], score, end_time))  

start_time = time.time()
score = svm.score(test_data, test_label)
end_time = time.time() - start_time
log_training_results("\t[%s]: mean accuracy on test-data: %s; execution time: %ss" % (svm.get_params()["kernel"], score, end_time))  

# Hyperparameter search -- Takes up a long time.
if config.hyper_parameter_search:
    svm = SVC()
    parameters = {
            "kernel":["linear"], 
            "C":[1,10,100],      
            "gamma":[0.01,0.005,0.001,0.0005,0.0001],
            #"shrinking":[True,False],      
            #"probability":[True,False], 
            #"tol":[0.01,0.001,0.0001],
    }
    scores = [
        'accuracy',
        #"precision",    # The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
        #"recall",       # The recall is intuitively the ability of the classifier to find all the positive samples.
        ]
    for score in scores:
        log_hyperparameter_search("\t--- [%s] Running Parameter-Tests [LINEAR-SVC] ---" % datetime.now())
        log_hyperparameter_search("\tTuning parameters for criteria [%s]" % score)
        # FIXME: Doesn't take accuracy as score for some reason. Refer to line below for accuracy score
        #grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring="%s_macro" % score, verbose=3)
        grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring='accuracy', verbose=3, n_jobs=-1)
        grid.fit(train_data, train_label)

        log_hyperparameter_search("\tBest parameters set found on following development set:")
        log_hyperparameter_search("\t\tSupport Vector: %s" % grid.best_estimator_)
        log_hyperparameter_search("\t\tSupport Vector Parametrization: %s" % grid.best_params_)
        log_hyperparameter_search("\t\tAsserted Score: %s" % grid.best_score_)
        log_hyperparameter_search("Total Score \t\t Configurations")

        means = grid.cv_results_["mean_test_score"]
        stds = grid.cv_results_["std_test_score"]
        params = grid.cv_results_["params"]
        for mean, std, params in zip(means, stds, params):
            log_hyperparameter_search("%0.3f (+/-%0.03f)\t%r" % (mean, std, params))
        print("Wrote classifier comparisons to file ", hyperparameter_search_log)

        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
    
        y_true, y_pred = test_label, grid.predict(test_data)
        log_hyperparameter_search(classification_report(y_true, y_pred))
        print()

# Evalute SVM.SVC with parameters on data below
svm = SVC(kernel='poly')
start_time = time.time()
svm.fit(train_data, train_label)
end_time = time.time() - start_time
log_training_results("Trained new model: {'Kernel':'%s'} in %s seconds" % (svm.get_params()["kernel"], end_time))

start_time = time.time()
score = svm.score(train_data, train_label)
end_time = time.time() - start_time
log_training_results("\t[%s]: mean accuracy on train-data: %s; execution time: %ss" % (svm.get_params()["kernel"], score, end_time))  

start_time = time.time()
score = svm.score(test_data, test_label)
end_time = time.time() - start_time
log_training_results("\t[%s]: mean accuracy on test-data: %s; execution time: %ss" % (svm.get_params()["kernel"], score, end_time))  

# Hyperparameter search -- Takes up a long time.
if config.hyper_parameter_search:
    svm = SVC()
    parameters = {
            "kernel":["poly"], 
            "C":[1,10,50,100],            
            "gamma":[0.01,0.005,0.001,0.0005,0.0001],        
            #"coef0":[0.0, 0.5],
            #"degree":[3,5,10],                
            #"shrinking":[True,False],      
            #"probability":[True,False], 
            #"tol":[0.01,0.001,0.0001],
    }
    scores = [
        "accuracy",
        #"precision",    # The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
        #"recall",       # The recall is intuitively the ability of the classifier to find all the positive samples.
        ]
    for score in scores:
        log_hyperparameter_search("--- [%s] Running Parameter-Tests [POLY-SVC] ---" % datetime.now())
        log_hyperparameter_search("\tTuning parameters for criteria [%s]" % score)
        # FIXME: Doesn't take accuracy as score for some reason. Refer to line below for accuracy score
        #grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring="%s_macro" % score, verbose=3)
        grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring='accuracy', verbose=3, n_jobs=-1)
        grid.fit(train_data, train_label)

        log_hyperparameter_search("\tBest parameters set found on following development set:")
        log_hyperparameter_search("\t\tSupport Vector: %s" % grid.best_estimator_)
        log_hyperparameter_search("\t\tSupport Vector Parametrization: %s" % grid.best_params_)
        log_hyperparameter_search("\t\tAsserted Score: %s" % grid.best_score_)
        log_hyperparameter_search("Total Score \t\t Configurations")

        means = grid.cv_results_["mean_test_score"]
        stds = grid.cv_results_["std_test_score"]
        params = grid.cv_results_["params"]
        for mean, std, params in zip(means, stds, params):
            log_hyperparameter_search("%0.3f (+/-%0.03f)\t%r" % (mean, std, params))
        print("Wrote classifier comparisons to file ", hyperparameter_search_log)

        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
    
        y_true, y_pred = test_label, grid.predict(test_data)
        log_hyperparameter_search(classification_report(y_true, y_pred))
        print()

# Evalute SVM.SVC with parameters on data below
svm = SVC(kernel='rbf')
start_time = time.time()
svm.fit(train_data, train_label)
end_time = time.time() - start_time
log_training_results("Trained new model: {'Kernel':'%s'} in %s seconds" % (svm.get_params()["kernel"], end_time))

start_time = time.time()
score = svm.score(train_data, train_label)
end_time = time.time() - start_time
log_training_results("\t[%s]: mean accuracy on train-data: %s; execution time: %ss" % (svm.get_params()["kernel"], score, end_time))  

start_time = time.time()
score = svm.score(test_data, test_label)
end_time = time.time() - start_time
log_training_results("\t[%s]: mean accuracy on test-data: %s; execution time: %ss" % (svm.get_params()["kernel"], score, end_time))  


# Hyperparameter search -- Takes up a long time.
if config.hyper_parameter_search:
    svm = SVC()
    parameters = {
            "kernel":["rbf"], 
            "C":[1,10,50,100],            
            "gamma":[0.01,0.005,0.001,0.0005,0.0001],        
            #"shrinking":[True,False],      
            #"probability":[True,False], 
            #"tol":[0.01,0.001,0.0001],
    }
    scores = [
        "accuracy",
        #"precision",    # The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
        #"recall",       # The recall is intuitively the ability of the classifier to find all the positive samples.
        ]
    for score in scores:
        log_hyperparameter_search("--- [%s] Running Parameter-Tests [RBF-SVC] ---" % datetime.now())
        log_hyperparameter_search("\tTuning parameters for criteria [%s]" % score)
        # FIXME: Doesn't take accuracy as score for some reason. Refer to line below for accuracy score
        #grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring="%s_macro" % score, verbose=3)
        grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring='accuracy', verbose=3, n_jobs=-1)
        grid.fit(train_data, train_label)

        log_hyperparameter_search("\tBest parameters set found on following development set:")
        log_hyperparameter_search("\t\tSupport Vector: %s" % grid.best_estimator_)
        log_hyperparameter_search("\t\tSupport Vector Parametrization: %s" % grid.best_params_)
        log_hyperparameter_search("\t\tAsserted Score: %s" % grid.best_score_)
        log_hyperparameter_search("Total Score \t\t Configurations")

        means = grid.cv_results_["mean_test_score"]
        stds = grid.cv_results_["std_test_score"]
        params = grid.cv_results_["params"]
        for mean, std, params in zip(means, stds, params):
            log_hyperparameter_search("%0.3f (+/-%0.03f)\t%r" % (mean, std, params))
        print("Wrote classifier comparisons to file ", hyperparameter_search_log)

        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
    
        y_true, y_pred = test_label, grid.predict(test_data)
        log_hyperparameter_search(classification_report(y_true, y_pred))
        print()

# Evalute SVM.SVC with parameters on data below
svm = SVC(kernel='sigmoid')
start_time = time.time()
svm.fit(train_data, train_label)
end_time = time.time() - start_time
log_training_results("Trained new model: {'Kernel':'%s'} in %s seconds" % (svm.get_params()["kernel"], end_time))

start_time = time.time()
score = svm.score(train_data, train_label)
end_time = time.time() - start_time
log_training_results("\t[%s]: mean accuracy on train-data: %s; execution time: %ss" % (svm.get_params()["kernel"], score, end_time))  

start_time = time.time()
score = svm.score(test_data, test_label)
end_time = time.time() - start_time
log_training_results("\t[%s]: mean accuracy on test-data: %s; execution time: %ss" % (svm.get_params()["kernel"], score, end_time))  

# Hyperparameter search -- Takes up a long time.
if config.hyper_parameter_search:
    svm = SVC()
    parameters = {
            "kernel":["sigmoid"], 
            "C":[1,10,50,100],            
            "gamma":[0.01,0.005,0.001,0.0005,0.0001],        
            #"shrinking":[True,False],      
            #"probability":[True,False], 
            #"tol":[0.01,0.001,0.0001],
    }
    scores = [
        "accuracy",
        #"precision",    # The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
        #"recall",       # The recall is intuitively the ability of the classifier to find all the positive samples.
        ]
    for score in scores:
        log_hyperparameter_search("--- [%s] Running Parameter-Tests [Sigmoid-SVC] ---" % datetime.now())
        log_hyperparameter_search("\tTuning parameters for criteria [%s]" % score)
        # FIXME: Doesn't take accuracy as score for some reason. Refer to line below for accuracy score
        #grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring="%s_macro" % score, verbose=3)
        grid = GridSearchCV(estimator=svm, param_grid=parameters, scoring='accuracy', verbose=3, n_jobs=-1)
        grid.fit(train_data, train_label)

        log_hyperparameter_search("\tBest parameters set found on following development set:")
        log_hyperparameter_search("\t\tSupport Vector: %s" % grid.best_estimator_)
        log_hyperparameter_search("\t\tSupport Vector Parametrization: %s" % grid.best_params_)
        log_hyperparameter_search("\t\tAsserted Score: %s" % grid.best_score_)
        log_hyperparameter_search("Total Score \t\t Configurations")

        means = grid.cv_results_["mean_test_score"]
        stds = grid.cv_results_["std_test_score"]
        params = grid.cv_results_["params"]
        for mean, std, params in zip(means, stds, params):
            log_hyperparameter_search("%0.3f (+/-%0.03f)\t%r" % (mean, std, params))
        print("Wrote classifier comparisons to file ", hyperparameter_search_log)

        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
    
        y_true, y_pred = test_label, grid.predict(test_data)
        log_hyperparameter_search(classification_report(y_true, y_pred))
        print()