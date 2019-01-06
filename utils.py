import numpy as np

def one_hot(value, classes):
	res = np.zeros(classes, dtype=np.int)
	res[value] = 1
	return res