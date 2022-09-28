import pandas as pd


def get_data():
    file_name = './data/drug_consumption.data'
    # Read the file in pandas data frame
    data = pd.read_csv(file_name, header=None)
    # store the dataset
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
label = data[:, -1]

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


print(X[30, :])
X = convert_to_number(X)
print(X[30, :])
print(X.shape)
print(label)
