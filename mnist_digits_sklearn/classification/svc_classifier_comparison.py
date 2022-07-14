from sklearn.svm import SVC
from datetime import datetime
from sklearn.model_selection import GridSearchCV

txt_out_file_path = 'sklearn-svm-parameter-tuning-log.txt'
csv_out_file_path = 'sklearn-svm-parameter-tuning-log.csv'

def print_to_txt_file(*s):
    with open(txt_out_file_path, 'a') as f:
        for arg in s:
            print(arg, file=f)

def run_svc_classifier_comparison(_X_train, _y_train):
    tuned_parameters = [
    {"kernel": ["linear"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    {"kernel": ["poly"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    {"kernel": ["sigmoid"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    #{"kernel": ["precomputed"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]}, # Dataset is not a square matrix !!
    ]
    scores = ["precision", "recall"]

    for score in scores:
        print_to_txt_file("--- [%s] Running Parameter-Tests ---" % datetime.now())
        print_to_txt_file("Tuning parameters for criteria [%s]" % score)

        clf = GridSearchCV(estimator=SVC(), param_grid=tuned_parameters, scoring="%s_macro" % score)
        clf.fit(_X_train, _y_train)

        print_to_txt_file("Best parameters set found on following development set:")
        print_to_txt_file("\tSupport Vector: %s" % clf.best_estimator_)
        print_to_txt_file("\tSupport Vector Parametrization: %s" % clf.best_params_)
        print_to_txt_file("\tAsserted Score: %s" % clf.best_score_)
        print_to_txt_file("\tTotal Score \t\t\t Configurations")

        means = clf.cv_results_["mean_test_score"]
        stds = clf.cv_results_["std_test_score"]
        for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
            print_to_txt_file("%0.3f (+/-%0.03f)\t%r" % (mean, std, params))
        print_to_txt_file("")
    print("Wrote classifier comparisons to file ", txt_out_file_path)