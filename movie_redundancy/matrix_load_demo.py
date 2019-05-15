import numpy as np

matrix = np.zeros((59370,59370),dtype=np.float32) #Hard code here
with open('matrix.txt') as f:
	index = 0
	while True:
		print(index)
		line = f.readline()
		if not line:
			break
		temp_list = line.split()
		for i,value in enumerate(temp_list):
			matrix[index][i] = value
		index += 1