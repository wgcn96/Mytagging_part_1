#-*-encoding:utf-8-*-
#以余弦相似度为基础的user-based协同过滤
import numpy as np
import math

neighbourhood_number = 5
#使用numpy的函数加快运行速度
def cosSimilarity(a,b):
	return np.vdot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def generate(similar_matrix,top250_index,rate_matrix,missing_number,recommend_number):
	print "*****missing number: " + str(missing_number) + " recommend_number: " + str(recommend_number) + " *****"
	final_matrix = np.zeros((top250_index.shape[0],recommend_number))
	for n,target in enumerate(top250_index):
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
			current += sim*(rate_matrix[index]-1)
			total_weight += sim
		current = 1+(current/total_weight)
		sort_dict = dict()
		for i,value in enumerate(current):
			sort_dict[i] = value
		sort_result = sorted(sort_dict.items(),key= lambda x:x[1],reverse=True)
		count = 0
		for i,value in sort_result:
			if rate_matrix[target][i] != 1:
				final_matrix[n][count] = i
				count += 1
			if count == recommend_number:
				break
	np.savetxt('UBCF/'+str(missing_number)+'_'+str(recommend_number)+"_ub_matrix.txt",final_matrix,fmt="%d")

if __name__ == '__main__':
	print "Loading Rating Matrix"
	rate_matrix = np.loadtxt('100_missing_matrix.txt')
	
	similar_matrix = np.zeros((rate_matrix.shape[0],rate_matrix.shape[0]))
	top250_index = np.load('top250_movie_pos.npy')
	top250_index_set = set()
	
	for index in top250_index:
		top250_index_set.add(index)
	print "Caculating Similar_matrix for preparation"
	for i in range(rate_matrix.shape[0]):		
		if i not in top250_index_set:
			continue
		print i
		for j in range(rate_matrix.shape[0]):
			if i == j:
				continue
			result = cosSimilarity(rate_matrix[i],rate_matrix[j])
			similar_matrix[i][j] = result
			similar_matrix[j][i] = result

	print "Collecting result"
	for i in range(1,101):
		generate(similar_matrix,top250_index,rate_matrix,100,i)



