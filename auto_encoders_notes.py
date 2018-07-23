'''

There are different types of autoencoders

What are Auto Encoders?
Auto Encoders encode itself
visible input layer >>encoding>> hidden nodes >>decoding>> visible output nodes
output is identical to inputs
they are self-supervised (not pure unsupervised) they are comparing to somthing    
Use case = feature detection, built recommender systems, used for encoding
usually we use the hyperbolic tangent activation function for this
in the end (output) we apply the softmax function

generally we add biases inside another node in input and output with +1 inside the node
z = f(Wx+b) biases are the b
bias affects the layer ahead

=== training autoencoders ===
- start with array where the lines (the obserations) correspond to the users and the columns
  (the features) correspond to the movies. Each cell (u, i) contains the rating (from 1 to 5, 0
  if no rating) of the movie i and the user u
- The first user goes into the network. INput vector x = (r1, r2, r3, ....) contains all
  the ratings for all the movies
- then input vector x is encoded into a vector z of lower dimensions by a mapping function f
  (usually either sigmoid or hyperbolic tangent) z = f(Wx+b) where W is the vector input
  weights and b the bias
- z is decoded into the output vector y of the same dimensions as x, aiming to replicate the
  input vector x
- The reconstructed error d(x,y) = ||x-y|| is computed. Goal is to minimize this error
- Back-propagation: from right to left, the error is back-propagated. The weights are updated
  according to how much they are responsible for the error. The learning rate decides by how much
  we update the weights. (gradient descent)
- Repeat steps 1 to 6 and update the weights after each observation (Reinforcement Learning). Or:
  repeat steps 1 to 6 but update the weights only after a batch of observations (Batch Learning)
- WHen the whole training set passed through the ANN then one epoch. repeat for each epoch

=== overcomplete hidden layers ===
more hidden layers than input layers
allow us to extract more features
problem is: autoencoder can cheat so can just pass the values for the output nodes
some hidden nodes are not even going to be used

=== sparse autoencoders ===
with overcomplete hidden layers
regularization technique is used in this network
it introduces a penalty on the loss function, so it doens't allow ALL the hidden layer
nodes to be used at the same time
end of the day: training the whole layer, but not using all of the nodes in the hidden layer
so the autoencoder cannot cheat

'''
