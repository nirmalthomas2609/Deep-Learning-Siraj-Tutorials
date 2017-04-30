from numpy import *
import matplotlib.pyplot as pyplot

class NeuralNetwork():
	def __init__(self):
		#seeds the random number generator
		random.seed(1)

		# assuming the weights of the neural natwork fall in the range of -1 to 1 with the mean being 0
		self.weight1=2*random.random((4,4))-1
		self.weight2=2*random.random((4,1))-1
	def predict(self,inputs):
		layer1=self.__sigmoid(dot(inputs,self.weight1))
		output=self.__sigmoid(dot(layer1,self.weight2))
		return (layer1,output)	

	def __sigmoid(self,x):
		return 1/(1+exp(-x))

	def __sigmoid_derivative(self,x):
		return x*(1-x)

	def train(self,inputs, outputs, no_of_iterations,no_of_training_examples,learning_rate):
		for i in range(no_of_iterations):
			layer1,output=self.predict(inputs)
			del3=output-outputs
			print "The value of the martrix del3 is ",del3
			print "The value of the first layer of the neural network is ",layer1
			print "The value of (dot(self.weight2,del3.T).T) is ",(dot(self.weight2,del3.T).T)
			print "The value of the matrix self.__sigmoid_derivative(layer1)) is ",self.__sigmoid_derivative(layer1)
			del2=((dot(self.weight2,del3.T).T)*(self.__sigmoid_derivative(layer1)))
			print "The value of the del2 matrix is ",del2
			Delta2=(1/no_of_training_examples)*dot(layer1.T,del3)
			Delta1=(1/no_of_training_examples)*dot(inputs.T,del2)
			print "The value of the matrix Delta2 is ",Delta2
			print "The value of the matrix Delta1 is ",Delta1
			self.weight1-=learning_rate*Delta1
			self.weight2-=learning_rate*Delta2


if __name__ == "__main__":
	#Feedforward neural network with a one input layer, one hidden layer and one output layer which has a single output class 

	nn=NeuralNetwork() # object of class Neural Network being created 
	
	print "----------------------------------------------------------------------"

	print "The initial weights of layer 1 of the neural network is -->:"
	print nn.weight1
	print "The shape of the weights of the layer 1 of the neural network is -->:"
	print nn.weight1.shape

	print "-----------------------------------------------------------------------"

	print "The initial weights of the layer 2 of the neural network is -->:"
	print nn.weight2
	print "The shape of the weights of the layer 2 of the neural network is -->:"
	print nn.weight2.shape

	print "-----------------------------------------------------------------------"

	#Assumes there are 4 input values for erach of the training  example, with the hidden layer also having 4 activation values
	#Assuming there are 10 training examples
	#Assuming that each of the input value is either 0 or 1
	# Randomly initialising the training set for convenience
	#initialising the learniubng

	inputs=random.randint(2,size=(10,4))
	print "The randomly initialised input training examples are -->:"
	print inputs

	outputs=random.randint(2,size=(10,1))
	print "The randomly initialised labels for the training examples are -->:"
	print outputs 

	# Performing the training iterations 100 times
	learning_rate=0.001
	nn.train(inputs,outputs,5,10,learning_rate)

	print "The weights of the layer 1 of the neural network after training is -->:"
	print nn.weight1

	print "The weight of the layer 2 of the neural network after training is -->:"
	print nn.weight2

	test=random.randint(2,size=(1,4))
	print "Predicting the output of the neural network when the input is -->:"
	print test

	print "Predicted probability that the output is 1 for the test training example is -->:"
	print nn.predict(test)[1]

	print "--------------------------------------------------------------------------"
