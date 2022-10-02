import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, SelectFpr, GenericUnivariateSelect

def get_data(file_name):
    # Read the file in pandas data frame
    data = pd.read_csv(file_name, header=None)
    # store the datasfrom sklearn.metrics import accuracy_scoreet
    dataset = data.values
    return dataset

#Now, I want to change the last dimention to a binary type: user and nonuser
# In other words, I want to convert CL0 and CL1 -> nonuser, the other five to users
def convert_to_binary_class(data, c):
    m, n = data.shape
    for i in range(m):
        if data[i][c] == 'CL0' or data[i][c] == 'CL1':
            data[i][c] = 'user'
        else:
            data[i][c] = 'nonuser'

def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    #print(le.classes_)
    return y_train_enc, y_test_enc

def feature_selection(X_train, X_test, y_train, y_test):
    fs = SelectKBest(score_func=f_classif, k=5)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)

    for i in range(len(fs.scores_)):
    	print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()
    return X_train_fs, X_test_fs
