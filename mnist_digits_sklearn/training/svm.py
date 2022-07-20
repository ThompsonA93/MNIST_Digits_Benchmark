from sklearn import svm


def train_svm_model(c, gamma, kernel):
    return svm.SVC(C=c, gamma=gamma, kernel=kernel)
