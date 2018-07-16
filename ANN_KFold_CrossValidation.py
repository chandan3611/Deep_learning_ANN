#Problem:
#Data Set is of bank. Bank has measured few parameters about their coustomers and based on these data they want to find that is customer is going to stay with bank or not.
#I have build this model using K-Fold cross validation technique.

# Imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense


#kersaClassifier
#Here we have implemented model with keras and kfold corss validation function belongs to scikit learn. So in some way we to combine keras and scikit learn. That can be done by kersaClassifier.
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#Importing DataSet
dataset = pd.read_csv("C:\\Users\\Chandan.S\\Desktop\\DeepLearning\\ANN\\Churn_Modelling.csv")
x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


#Encoding the Categorical Independent Variable
#From Dataset we see that we have some categorical variables in matrix of features. Therefore we need to encode them.
#1) Here two independent variables(Geography and Gender) which have categories that are string and therefore we need to encode these categorical variables. One thing to note that we are encoding data before spliting data into tranning and test set.
#2) No need to encode Dependent variable.

#Here we have two categorical variable: Geography and Gender. So taking two object of LabelEncoder()
# 1) labelencoder_X_1 : Encoding Geography
labelencoder_X_1 = LabelEncoder()
x[:, 1] = labelencoder_X_1.fit_transform(x[:, 1])

# 2) labelencoder_X_2 : Encoding Gender
labelencoder_X_2 = LabelEncoder()
x[:, 2] = labelencoder_X_1.fit_transform(x[:, 2])


#Till now we have encoded all categorical variable. Since categorical variables are not ordinal that is there is no relational order between categories of categorical variables. So we need to create dummy variable for these categorical variables. We need to create dummy variable only for "Geography" categorical variable since it has 3 categories.
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()

# Remove one dummy variable to avoid falling in dummy variable trap
x = x[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling : There will be lots of computation, so we need to apply Feature Scaling to easy these calculation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Build model:
# 1) Create classifier object
# 2) Use add() method for adding hidden layers to ANN.
# units : Number of nodes in hidden layer, there is no tumb rule to find optimal number of nodes, that can be decided based on experiment.One way is to take average of number of nodes is input and number of node is output layer. In this case (11+1)/2 = 6. So we are taking units = 6. It is just a suggestion not and tumb rule. Otherwise we can use cross validation technique to figure out optimal vale of all parameters.
# input_dim : Important parameter, number of node in input layer that is number of independent variables.

def build_classifer():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = "binary_crossentropy",metrics = ['accuracy'])
    return classifier

# K-Fold Cross Validation
classifier = KerasClassifier(build_fn=build_classifer,batch_size=10, epochs =100)
accuracies = cross_val_score(estimator=classifier,X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()














