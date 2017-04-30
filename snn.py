from numpy import *
class NeuralNetwork():
	def __init__(self):
		#seeed the random number generator so that the random number generator produces
		#the same set of random numbers every time the program runs
		random.seed(1)

		#We model a single neural network with 3 inoput connections and 1 output value
		# We assign random synaptic weights to a 3x1 martrix, with values in the range -1 to 1
		# and having mean 0
		self.synaptic_weights=2*random.random((3,1))-1


	#The sigmoid function, which decscribes an s shaped curve 
	#we pass the weighted sums of the inputs through this function
	# to normalise the outputs between 0 and 1
	def __sigmoid(self,x):
		return 1/(1+exp(-x))
	
	def predict(self,inputs):
		# pass inputs through our neural network (our single neuron)
		return self.__sigmoid(dot(inputs,self.synaptic_weights))

	def train(self,training_set_inputs,training_set_outputs,number_of_training_iterations):
		for iteration in xrange(number_of_training_iterations):
			output=self.predict(training_set_inputs)
			print "The shape of the output vector is ",output.shape
			print "The shape of the training set output vector is ",training_set_outputs.shape
			#calculate the error between the predicted outputs and the labelled outputs
			error=training_set_outputs-output
#			print "Calling the sigmoid derivative function",self.__sigmoid_derivative(output)
			adjustment= dot(training_set_inputs.T,error*self.__sigmoid_derivative(output))
			print "The shape of the error vector is ",error.shape
			print "The shape of the training set transpose vector is ",training_set_inputs.T.shape
			print "The shape of the vector right dot vector is",error*self.__sigmoid_derivative(output).shape
			print "The shape of the adjustment vector  is ",adjustment.shape
			self.synaptic_weights+=adjustment

	def __sigmoid_derivative (self,x):
		return x*(1-x)



if __name__=="__main__":
	
	#Initialise a single neural network
	neural_network=NeuralNetwork()
	print "Random starting synaptic weights"
	print neural_network.synaptic_weights

	#The training set will have four inout examples, each consisting of 3 input values 
	# and one output value
	training_set_inputs=array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
	training_set_outputs=array([[0,1,1,0]]).T
	print "The shape of the training set output vecrtor is ",training_set_outputs.shape

	#train the neural network using the training set
	#1000 iterations during the training
	neural_network.train(training_set_inputs,training_set_outputs,1000)

	print "New synaptic weights after the training:"
	print neural_network.synaptic_weights

	print "Predicting the output for the given input [1,0,0] -->"
	print neural_network.predict(array([1,0,0]))
