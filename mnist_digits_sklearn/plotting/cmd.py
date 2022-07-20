from sklearn import metrics

def data_to_cmd(_y_test, _predictions):
    print("Creating Confusion Matrix")
    disp = metrics.ConfusionMatrixDisplay.from_predictions(_y_test,_predictions)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix: \n {disp.confusion_matrix}")
