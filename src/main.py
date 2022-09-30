import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

from preprocessing import *
from models import *

def draw_roc(x, y):
    fpr, tpr, thresholds = metrics.roc_curve(y, x)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    display.plot()
    plt.show()

#X stands for the features, class_c stands for the label class
def experiment(X, labels):
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33, random_state=1)
    y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
    X_train_fs, X_test_fs = feature_selection(X_train, X_test, y_train_enc, y_test_enc)
    #X_result = model_DT(X_train_fs, X_test_fs, y_train_enc, y_test_enc)

    models = [model_DT, model_RF, model_SVM, model_KNN]
    for m in models:
        X_result = m(X_train_fs, X_test_fs, y_train_enc, y_test_enc)
        print(confusion_matrix(y_test_enc, X_result))
        draw_roc(X_result, y_test_enc)

# I will split the data into data and labels
def main():
    file_name = './data/drug_consumption.data'
    print("Getting Data ...")
    data = get_data(file_name)
    print(data.shape)
    classes = [31, 30]
    for c in classes:
        convert_to_binary_class(data, c)

    X = data[:, 1:13]    # getting the features
    labels = data[:, 31] #getting class label
    experiment(X, labels)

if __name__ == "__main__":
    main()

#from sklearn import svm

# clf = svm.SVC()
# clf.fit(X_train, y_train)
# X_result = clf.predict(X_test)
# print(accuracy_score(y_test, X_result))
# print(confusion_matrix(y_test, X_result))
# x, y = prepare_targets(y_test, X_result)
# fpr, tpr, thresholds = metrics.roc_curve(x, y)
# roc_auc = metrics.auc(fpr, tpr)
# display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
# plt.show()
# display.plot()
