import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from matplotlib import pyplot
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt

def get_data():
    file_name = './data/drug_consumption.data'
    # Read the file in pandas data frame
    data = pd.read_csv(file_name, header=None)
    # store the datasfrom sklearn.metrics import accuracy_scoreet
    dataset = data.values
    return dataset


data = get_data()
print(data)
print(data.shape)

#Now, I want to change the last dimention to a binary type: user and nonuser
# In other words, I want to convert CL0 and CL1 -> nonuser, the other five to users
def convert_to_binary_class(data):
    m, n = data.shape
    for i in range(m):
        if data[i][31] == 'CL0' or data[i][31] == 'CL1':
            data[i][31] = 'user'
        else:
            data[i][31] = 'nonuser'

convert_to_binary_class(data)
print(type(data[0][0]))

# Now, it is the time to do feature selection
# Since our output data is chategorical and our input data is mixed chategorical and non chategorical
# We have two choices.
# one let s encode the chategorical ones.

# I will split the data into data and labels
X = data[:, :-1]
labels = data[:, -1]

def convert_to_number(data):
    m, n = data.shape
    for i in range(m):
        for j in range(n):
            if isinstance(data[i][j], str):
                if data[i][j] == 'CL0':
                    data[i][j] = 0
                elif data[i][j] == 'CL1':
                    data[i][j] = 1
                elif data[i][j] == 'CL2':
                    data[i][j] = 2
                elif data[i][j] == 'CL3':
                    data[i][j] = 3
                elif data[i][j] == 'CL4':
                    data[i][j] = 4
                elif data[i][j] == 'CL5':
                    data[i][j] = 5
                elif data[i][j] == 'CL6':
                    data[i][j] = 6
    return data

def prepare_targets(y_train, y_test):
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc


print(X[30, :])
X = convert_to_number(X)
print(X[30, :])
print(X.shape)
print(labels)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33, random_state=1)
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
from sklearn.metrics import accuracy_score
print(y_train)
print(y_train_enc)

fs = SelectKBest(score_func=f_classif, k=16)
fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)

for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

print(X_train_fs.shape)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

X_result = clf.predict(X_test)
#print(X_result)
print(accuracy_score(y_test, X_result))
print(confusion_matrix(y_test, X_result))
x, y = prepare_targets(y_test, X_result)
fpr, tpr, thresholds = metrics.roc_curve(x, y)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
display.plot()
plt.show()


# random forest
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
X_result = clf.predict(X_test)
print(accuracy_score(y_test, X_result))
print(confusion_matrix(y_test, X_result))

x, y = prepare_targets(y_test, X_result)
fpr, tpr, thresholds = metrics.roc_curve(x, y)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
display.plot()
plt.show()



clf = svm.SVC()
clf.fit(X_train, y_train)
X_result = clf.predict(X_test)
print(accuracy_score(y_test, X_result))
print(confusion_matrix(y_test, X_result))
x, y = prepare_targets(y_test, X_result)
fpr, tpr, thresholds = metrics.roc_curve(x, y)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
display.plot()
plt.show()




from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)
X_result = clf.predict(X_test)
print(accuracy_score(y_test, X_result))


print(confusion_matrix(y_test, X_result))
x, y = prepare_targets(y_test, X_result)
fpr, tpr, thresholds = metrics.roc_curve(x, y)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
display.plot()
plt.show()
