import numpy as np
def isint(x):
	try:
		a=int(x)
		return True
	except:
		return False

#this should be able to convert any non integers to integers and keep each row separate.
def convert_to_int(inp,index):
	if index not in convert_to_int.indexes.keys():
		convert_to_int.indexes[index] = dict()
		convert_to_int.indexes[index]["counter"] = 0
	if inp not in convert_to_int.indexes[index].keys():
		convert_to_int.indexes[index]["counter"]+=1
		convert_to_int.indexes[index][inp] = convert_to_int.indexes[index]["counter"]
	return convert_to_int.indexes[index][inp]
convert_to_int.indexes = dict()

def clean(x):
	return x.replace(" ","").replace("\n","").replace("\"","").strip()

#returns each row of data in an array of ints
def prepare_data(line):
	split = line.split(";")
	data = [int(clean(x)) if isint(clean(x)) else convert_to_int(x,y) for x,y in zip(split,range(len(split)))]
	targets = data[-1:]
	data = data[:-3]
	return tuple([np.array(data),np.array(targets)])

