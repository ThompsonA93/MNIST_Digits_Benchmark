from sklearn import svm

def run_sklearn_svm(x_train, y_train, x_test, y_test):
    print("+-+-+-+-+ Executing SKLearn-SVM implementation +-+-+-+-+")

    clf = svm.SVC()
    clf.fit(x_train, y_train)