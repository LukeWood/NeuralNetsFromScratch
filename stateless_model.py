import numpy as np
import random
class stateless_model:

	def __init__(self,input_size,out_size,epsilon=0.01,reg_lambda=0.01,hidden_size=5, hidden_layers=1):	
		self.input_layer = np.random.random((input_size,hidden_size))
		#first hidden
		self.hidden_layers = [np.random.random((hidden_size,hidden_size)) for x in range(hidden_layers)]
		#end hidden_layers
		self.output_layer= np.random.random((hidden_size,out_size))
		
	def activate(self,inp):
		xn = np.dot(inp,self.input_layer)
		for hidden_layer in self.hidden_layers:
			xn = np.dot(xn,hidden_layer)
		out  = np.dot(xn,self.output_layer)
		return out
	
	#activates the temporary network to test fitness
	def __activate_temp(self,inp,temp_inp,hidden,out):
		xn = np.dot(inp,temp_inp)
		for hidden_layer in hidden:
			xn = np.dot(xn,hidden_layer)
		out2  = np.dot(xn,out)
		return out2
		
	#randomly adjusts layers for training, highly inefficient
	def training_iter_random(self,inp,out,adjustment_rate=.01):
		temp_inp = self.input_layer
		temp_hidden = self.hidden_layers
		temp_output = self.output_layer
		
		#add a random number in the range of adjustment_rate to each element in our np array
		for i in np.nditer(temp_inp,op_flags=["readwrite"]):
			i+=(adjustment_rate * random.uniform(-1, 1))
		for j in temp_hidden:
			for i in np.nditer(j,op_flags=["readwrite"]):
				i+=(adjustment_rate * random.uniform(-1, 1))
		for i in np.nditer(temp_output,op_flags=["readwrite"]):
			i+=(adjustment_rate * random.uniform(-1, 1))
		temp_out = self.__activate_temp(inp,temp_inp,temp_hidden,temp_output)
		self_out = self.activate(inp)
		
		temp_dist = np.linalg.norm(out-temp_out)
		self_dist = np.linalg.norm(out-self_out)
		
		if abs(temp_dist) < abs(self_dist):
			self.input_layer = temp_inp
			self.hidden_layers = temp_hidden
			self.output_layer = temp_output
		return
		
				
	def train(self,inp,out, training_iterations=10):
		for i in range(training_iterations):
			for i0, o0 in zip(inp,out):
				self.training_iter_random(i0,o0)
	
a = stateless_model(2,2)

inpu = np.array([[2,2]])
target = np.array([[2,2]])
print("TARGET: ")
print(target)
for i in range(1000):
	print("Iteration: %i"%(i*10))
	a.train(inpu,target,training_iterations=10)
	print(a.activate(inpu))
