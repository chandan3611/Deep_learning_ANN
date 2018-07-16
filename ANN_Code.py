#Problem:
#Data Set is of bank. Bank has measured few parameters about their coustomers and based on these data they want to find that is customer is going to stay with bank or not.

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

# Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer:
#We have created classifer object in previous step, now we will use add() method of classifer to add different layer to the neural network. add() method will take only one argument as input and that is your hidden layer that we want to add to ANN. And this we can do using "Dense()" function.
#units : Number of nodes in hidden layer, there is no tumb rule to find optimal number of nodes, that can be decided based on experiment.One way is to take average of number of nodes is input and number of node is output layer. In this case (11+1)/2 = 6. So we are taking units = 6. It is just a suggestion not and tumb rule. Otherwise we can use cross validation technique to figure out optimal vale of all parameters.
#input_dim : Important parameter, number of node in input layer that is number of independent variables
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

#Dropout : Avoding Overfitting
#At each iteration of traning some neurals on ANN is randomaly disabled to prevent them from being to dependent while learning correlation. ANN will learn several independent correlation in the data. Since we have independent correlation, that prevents neruans from learning to much hence it will avoid overfitting. Dropout is applied at each layer.
classifier.add(Dropout(rate=0.1))

#Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.1))

#Adding the output layer
#Here units : will be 1 because dependent variable is categorical variable with binary outcome that is 0 or 1. units = 1 means one node in output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = "binary_crossentropy",metrics = ['accuracy'])


#Fit the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs =100)


#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Next step we will use confusion matrix for evaluation. predict() method here returns the probabilities and in order to use confusion matrix we dont need probabilites, instead we need predicated values
# threshold = 0.5
y_pred = (y_pred > 0.5)


#Evaluation : Confusion Matrix
from sklearn.metrics import confusion_matrix
ConfusionMatrix = confusion_matrix(y_test, y_pred)













