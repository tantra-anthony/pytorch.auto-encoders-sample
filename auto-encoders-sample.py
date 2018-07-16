# building an autoencoder

# we need to now import all the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# now we need to import the dataset
# commas are not used to separate the data
# engine needs to be python so that the data can be imported more efficiently
# classic encoding UTF8 cannot be imported as well, so need to change the encoding
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# import the users
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# import the ratings
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# now we prepare the training and test sets
# take only 1 train test split
# tab is better to user delimiter
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
# find patterns and features of the movies to predict the rating of the history of the user
# and previous movies

# then we need to convert this data frame into an array
# remember to specify type of array (int, etc.)
training_set = np.array(training_set, dtype = 'int')

# import and convert test_set
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# then we need to get the number of users and movies and the total number of users and movies
# so that we can create a matrix of features, we want to include all the movies, users, and rating
# u is user, i is movies, every panel is rating of user u and movie i
# need to get it programmatically as it's not very modular
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0]))) # first column of set
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1]))) # second column of set

# convert training and test set into array
# need to structure the data such that we can insert it into the network
# with users in rows and movies in columns (observations in lines and features in columns)
def convert(data):
    # create a list of lists
    new_data = []
    # loop for every user of the data
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users] # so that first column is the id_users
        id_ratings = data[:, 2][data[:, 0] == id_users]
        # initialize list of zeroes
        ratings = np.zeros(nb_movies)
        # access the indexes so we can insert the appropriate values
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings)) # make sure it's a list
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# convert all the training and test set into tensors, a matrix of vectors
# need to be list of lists
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# now we can build the autoencoders, the architecture of the Neural Network
# create a class for the autoencoder model
# this class is to define everything (activation, optimizer, etc)
class SAE(nn.Module): # stacked autoencoders with several hidden layers
    # inherit the nn.Module
    def __init__(self, ): # consider the nn.Module class for initializing variables
        # call super function to get inherited methods from nn.Module
        super(SAE, self).__init__()
        # define number of layers and hidden neurons
        # hidden layer shorter vector than the input layer
        # create object representing the connection between input and first encoded vector layer
        self.fc1 = nn.Linear(nb_movies, 20) # no of features, no of neurons in first hidden layer (tune this)
        # 20 is the features e.g. genre etc
        self.fc2 = nn.Linear(20, 10) # make connection with second hidden layer, everything must be linked
        self.fc3 = nn.Linear(10, 20) # make connection second to third hidden layer (first part of decoding)
        self.fc4 = nn.Linear(20, nb_movies) # make connection to output layer
        # we need to specify the activation function
        self.activation = nn.Sigmoid() # sigmoid is better than relu in this specific data
        
    # define a actions we are going to take to encode and decode
    # forward feed
    def forward(self, x):
        # need to do the first encoding
        x = self.activation(self.fc1(x)) # the self.fc1(x) returns the encoded vectors
        x = self.activation(self.fc2(x)) # do it for every full connection made
        x = self.activation(self.fc3(x))
        x = self.fc4(x) # this is the reconstructed input vector (output)
        # we do not need to put the activation in the decoded output function as it's the final step
        return x

# instantiate the class
sae = SAE()

# define the criteria (loss function) use MSE
criterion = nn.MSELoss()

# we need an optimizer, liek keras, we need to apply stochastic gradient descent
# there are classes for diff optimizers, for now use RMSprop
# lr is learning rate
# decay is used to decrease the lr every epoch to regulate the convergence
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5) # get all the params from the SAE

# now we need to train our stacked autoencoder
# need to create an optimized code
# train on several epochs
nb_epoch = 200
# loop over all epochs and loop over all values
for epoch in range(1, nb_epoch + 1):
    # initialize the loss
    train_loss = 0
    s = 0. # variable that will count the number of users that rate at least one movie
    for id_user in range(nb_users): # 0 to 943-1=942
        inp = Variable(training_set[id_user]).unsqueeze(0) # add additional fake dimension using torch
        target = inp.clone() # clone of the input since this is the "target"
        # save memory here to filter users who never rate anything
        if torch.sum(target.data > 0) > 0: # target.data is all the ratings of the user in the loop
            outp = sae(inp) # this will execute forward() function
            # gradient is computed against the input not the target
            target.require_grad = False
            # then we don't want to deal with the movies that are 0 or not rated
            outp[target == 0] = 0
            loss = criterion(outp, target) # vector of real ratings, vector of real ratings
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # only consider non-zero ratings
            # 1e-10 is required to prevent division by 0
            # mean_corrector represents the average of the error only that has ratings
            # now we call backward method for the loss (tweak the weights)
            loss.backward()
            # since loss is squared we need to get a first degree loss
            train_loss += np.sqrt(loss.data[0] * mean_corrector) # increment train_loss, get the part that contains the error
            s += 1. # increment s
            optimizer.step() # backward decides the direction the weight, optimizer.step() decides the weight/magnitude
    print('epoch: ', str(epoch) + ' loss: ' + str(train_loss/s))
        
# now we test the SAE using the model above
# initialize the loss
test_loss = 0
s = 0.
for id_user in range(nb_users): # 0 to 943-1=942
    # we need to keep the training set here because all the ratings are from the training_set
    # and we will target it to the test_set, this is like using the old model to get the result
    # then we compare the test_set with the predicted ratings in the training set
    inp = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0) # target is the real ratings
    # save memory here to filter users who never rate anything
    if torch.sum(target.data > 0) > 0: # target.data is all the ratings of the user in the loop
        outp = sae(inp) # this will execute forward() function
        # gradient is computed against the input not the target
        target.require_grad = False
        # then we don't want to deal with the movies that are 0 or not rated
        outp[target == 0] = 0
        loss = criterion(outp, target) # vector of real ratings, vector of real ratings
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # only consider non-zero ratings
        # 1e-10 is required to prevent division by 0
        # mean_corrector represents the average of the error only that has ratings
        # now we call backward method for the loss (tweak the weights)
        # loss.backward() don't need this as we don't need to backpropagate
        # since loss is squared we need to get a first degree loss
        test_loss += np.sqrt(loss.data[0] * mean_corrector) # increment train_loss, get the part that contains the error
        s += 1. # increment s
        # optimizer.step() no keep # backward decides the direction the weight, optimizer.step() decides the weight/magnitude
print('test loss: ' + str(test_loss/s))
        
    