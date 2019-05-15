import numpy as np
import os

recommend_number = 10

def cosSimilarity(a,b):
	return np.vdot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def genereateResult(missing_number,recommend_number,top250_index,rate_matrix):
	print "*****missing number: " + str(missing_number) + " recommend_number: " + str(recommend_number) + " *****"
	final_matrix = np.zeros((top250_index.shape[0],recommend_number))
	for n,target in enumerate(top250_index):
		print target
		sort_dict = dict()
		for i,value in enumerate(rate_matrix[target]):
			sort_dict[i] = value
		sort_result = sorted(sort_dict.items(),key= lambda x:x[1],reverse=True)
		count = 0
		for i,value in sort_result:
			if rate_matrix[target][i] != 1:
				final_matrix[n][count] = i
				count += 1
			if count == recommend_number:
				break
	np.savetxt('TagRank/'+str(missing_number)+'_'+str(recommend_number)+'_redundancy_matrix.txt',final_matrix,fmt="%d")


if __name__ == '__main__':
	print "Loading rating matrix"
	rate_matrix = np.loadtxt('100_missing_matrix.txt')
	top250_index = np.load('top250_movie_pos.npy')
	similar_matrix = np.zeros((rate_matrix.shape[0],rate_matrix.shape[0]))
	
	print "Caculating Similar_matrix"
	for i in range(rate_matrix.shape[0]):
		print i
		for j in range(i+1,rate_matrix.shape[0]):
			result = cosSimilarity(rate_matrix[i],rate_matrix[j])
			if result > 0.3:
				similar_matrix[i][j] = result
				similar_matrix[j][i] = result

	print "Calculating result"
	iteration = 1

	for n in range(iteration):
		for j in range(rate_matrix.shape[1]):
			print j
			rate_matrix[:,j] = np.dot(similar_matrix,rate_matrix[:,j].reshape(-1,1)).reshape(-1)
	
	for i in range(1,101):
		genereateResult(100,i,top250_index,rate_matrix)
	
