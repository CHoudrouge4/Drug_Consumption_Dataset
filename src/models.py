from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def model_exec(clf, X_train_fs, X_test_fs, y_train, y_test):
    clf = clf.fit(X_train_fs, y_train) #fitting the training data
    X_result = clf.predict(X_test_fs)  # testing
    print(accuracy_score(y_test, X_result))
    return X_result

def model_DT (X_train_fs, X_test_fs, y_train, y_test):
    clf = tree.DecisionTreeClassifier() #creating the model
    return model_exec(clf, X_train_fs, X_test_fs, y_train, y_test)

def model_RF(X_train_fs, X_test_fs, y_train, y_test):
    clf = RandomForestClassifier(random_state=0)
    return model_exec(clf, X_train_fs, X_test_fs, y_train, y_test)

def model_SVM(X_train_fs, X_test_fs, y_train, y_test):
    clf = svm.SVC()
    return model_exec(clf, X_train_fs, X_test_fs, y_train, y_test)

def model_KNN (X_train_fs, X_test_fs, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=5)
    return model_exec(clf, X_train_fs, X_test_fs, y_train, y_test)
