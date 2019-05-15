import numpy as np
import math

neighbourhood_number = 10
recommend_number = 10

def cosSimilarity(a,b):
	return np.vdot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


def generateResult(missing_number,neighbourhood_number,recommend_number,top250_index,rate_matrix,result_matrix):
	print "*****missing number: " + str(missing_number) + " recommend_number: " + str(recommend_number) + " *****"
	final_matrix = np.zeros((top250_index.shape[0],recommend_number))
	for n,target in enumerate(top250_index):
		temp = dict()
		for i in range(rate_matrix.shape[1]):
			if rate_matrix[target][i] == 0:
				temp[i] = result_matrix[target][i]
			else:
				temp[i] = 0
		result = sorted(temp.items(),key= lambda x:x[1],reverse=True)[:recommend_number]
		count = 0
		for index,value in result:
			final_matrix[n][count] = index
			count += 1
		
	np.savetxt("IBCF/"+str(missing_number)+"_"+str(recommend_number)+"_ib_matrix.txt",final_matrix,fmt="%d")

if __name__ == '__main__':
	print "Loading Data"
	rate_matrix = np.loadtxt('100_missing_matrix.txt')
	top250_index = np.load('top250_movie_pos.npy')
	result_matrix = np.zeros((rate_matrix.shape[1],rate_matrix.shape[0]))

	rate_matrix = np.transpose(rate_matrix)

	similar_matrix = np.zeros((rate_matrix.shape[0],rate_matrix.shape[0]))

	print "Caculating Similar_matrix"
	for i in range(rate_matrix.shape[0]):
		print i
		for j in range(i+1,rate_matrix.shape[0]):
			result = cosSimilarity(rate_matrix[i],rate_matrix[j])
			similar_matrix[i][j] = result
			similar_matrix[j][i] = result

	for target in range(rate_matrix.shape[0]):
		print target
		similar_dict = dict()
		for i in range(rate_matrix.shape[0]):
			if i == target:
				continue
			else:
				similar_dict[i] = similar_matrix[target][i]
		result = sorted(similar_dict.items(),key= lambda x:x[1],reverse=True)[:neighbourhood_number]
		current = [0 for i in range(rate_matrix.shape[1])]
		total_weight = 0
		for index,sim in result:
			current += sim*rate_matrix[index]
			total_weight += sim
		current = current/total_weight
		for i,value in enumerate(current):
			result_matrix[target][i] = value

	result_matrix = np.transpose(result_matrix)
	rate_matrix = np.transpose(rate_matrix)
	for i in range(1,101):
		generateResult(100,neighbourhood_number,i,top250_index,rate_matrix,result_matrix)
