from numpy import *
import matplotlib.pyplot as plt
from time import *

def compute_error_for_given_points(b,m,points):
	'''
	#manual method for the same function
	rows=points.shape[0]
	sum=0
	for i in range(rows):
		sum+=(points[i,1]-((m*points[i,0])+b))**2
	sum=float(sum)/rows
	return sum
	'''
	#Siraj Version
	totalError=0
	for i in range(0,len(points)):
		x=points[i,0]
		y=points[i,1]
		totalError+=(y-(m*x)-b)**2
	return totalError/float(len(points))

def step_gradient(b_current,m_current,points,learning_rate):
	#a single step in gradient descent
	b_gradient=0
	m_gradient=0
	N=float(len(points))
	'''
	# computes the gradients manually
	for i in range(len(points)):
		m_gradient+=points[i,0]*(points[i,1]-(m_current*points[i,0])-b_current)
		b_gradient+=(points[i,1]-(m_current*points[i,0])-b_current)
	b_gradient*=(-2/N)
	m_gradient*=(-2/N)
	'''
	for i in range(len(points)):
		x=points[i,0]
		y=points[i,1]
		m_gradient+=(-2/N)*(x*(y-m_current*x-b_current))
		b_gradient+=(-2/N)*(y-m_current*x-b_current)
	new_b=b_current-(learning_rate*b_gradient)
	new_m=m_current-(learning_rate*m_gradient)
	return(new_b,new_m)

def gradient_descent_runner(points,starting_b,starting_m,learning_rate,num_iterations):
	b=starting_b
	m=starting_m
	points=hstack((points,zeros((len(points),1))))
	print (points)
	plt.axis([20,80,0,150])
#	plt.plot(points[:,0],points[:,2])
	plt.ion()
	for i in range(num_iterations):
		print "Iteration Number :- ",i+1
		print "\n\n\n"
		b,m=step_gradient(b,m,points,learning_rate)
		for j in range(len(points)):
			points[j,2]=m*points[j,0]+b
#		del plt.lines[0]
		sc=plt.scatter(points[:,0],points[:,1])
		ln,=plt.plot(points[:,0],points[:,2])
		plt.pause(0.00005)
		ln.remove()
		sc.remove()
		print points
	return [b,m]
def run():
	points=genfromtxt('data.csv',delimiter=',')
	print ("Running....")
	print ("Initial b-->",0)
	print ("Initial m-->",0)
	print ("Initial Error-->",compute_error_for_given_points(0,0,array(points)))

	#hyper parameter --> tuning knobs in machine learning (determines how fast our model learns)
	learning_rate=0.00001
	#b--> the y intercept
	#m--> slope of the line
	initial_b=0
	initial_m=0
	num_iterations=200
	[b,m]=gradient_descent_runner(array(points),initial_b,initial_m,learning_rate,num_iterations)
	print ("Optimized value of b ",b)
	print ("Optimized value of m ",m)
	print ("Error after performing gradient descent ",compute_error_for_given_points(b,m,array(points)))

if __name__ == '__main__':
	run()