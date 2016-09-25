#by Luke Wood 2016

#external libraries
import numpy as np
import sys
#these are modules I wrote
import data_prep
from stateless_model import stateless_model

if __name__ =="__main__":
	sys.dont_write_bytecode = True
	first_it = True
	all_data = []
	for line in open("data/student-mat.csv"):
		#exclude first iteration
		if first_it:
			first_it= False
			continue
		all_data.append(data_prep.prepare_data(line))

	#create network model
