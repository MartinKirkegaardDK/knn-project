import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


#Here I open the data in a pandas dataframe
path = r'C:/Users/tjupp/Desktop/Projecter/MNIST project/' #Insert the path to your project
csv_file = 'train.csv'
open_file = path + csv_file
data = pd.read_csv(open_file) 


label = data["label"] #Here I store the correct labels which will be used for the test set.


#Here I remove the labels from the dataset
data = data.drop('label',axis = 1)


#Here I create the dataset and devide each element with 255 to get values between 0 and 1
data_set = []
for i in range(len(data.index)):
    k = data.iloc[i]
    data_set.append(np.true_divide(k,255))


data_set = np.array(data_set) #Here I convert the dataframe to a numpy array


#Here I create the test and train datasets
x_train, x_test, y_train, y_test =  train_test_split(data_set,label,test_size=0.2)


k1 = int(x_train.shape[1]**0.5) #As a rule of thumb k is the sqare root of number of elements in the dataset


model = KNeighborsClassifier(n_neighbors=k1) #Here I create the model
model.fit(x_train, y_train) #Here I train the model
prediction_list = model.predict(x_test) #Here I create a prediction based on the test material


#Here I print the accuracy
accuracy_score(y_test,prediction_list)


#Here I convert the lists to a matrix. This I can use for plotting
matrix = []
for i in range(len(data.index)):
    k = np.array(data.iloc[i].as_matrix().reshape(28,28)) #I use the reshape feature to create the matrix
    matrix.append(np.true_divide(k,255)) #Here I divide each element by 255 since I want all the data points to be between 0 and 1



n = 10 #N is number of predictions
for i in range(n,20):
    plt.imshow(matrix[y_test.index.values[i]],cmap = "gray") #Here I plot the matrix and make it gray scalled
    plt.title("Prediction: {}, actual {}".format(prediction_list[i],y_test.iloc[i])) #Here I print the title
    plt.show() #Here I display the plot


#Insert the file name and type
file_name = "0"
file_type = ".png"
#The picture has to have a transparent background 


from PIL import Image
img = Image.open(path + file_name + file_type).convert('LA').resize((28,28))#This converts it to gray scale and resizes the image

data = np.array(img) #I convert it to a numpy array

data = data[:,:,1] #I remove the transparent features

data = data.reshape(784) #I reshape it to a single list

data = np.true_divide(data,255) #I divide every element with 255 to get elements between 0 and 1

model.predict([data]) #Here I use the model to predict the image

print(model.predict_proba([data])) #Here you can see the probability of each prediction

