#-*- encoding:utf-8 -*-
import numpy as np
import sys
import json
reload(sys)
sys.setdefaultencoding('utf-8')

def cosSimilarity(a,b):
	return np.vdot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def generate(rate_matrix,top250_vector,top250_index,tag_vector_map,missing_number,recommend_number):
	print "*****missing number: " + str(missing_number) + " recommend_number: " + str(recommend_number) + " *****"
	final_matrix = np.zeros((top250_index.shape[0],recommend_number))
	for n,target in enumerate(top250_index):
		sort_dict = dict()
		for i in range(rate_matrix[target].shape[0]):
			if rate_matrix[target][i] == 0:
				try:
					sort_dict[i] = cosSimilarity(top250_vector[n],tag_vector_map[tag_list[i].decode()])
				except:
					continue
		count = 0
		for i,value in sorted(sort_dict.items(),key= lambda x:x[1],reverse=True)[:recommend_number]:
			final_matrix[n][count] = i
			count += 1
	np.savetxt('Rocchio/'+str(missing_number)+'_'+str(recommend_number)+"_r_matrix.txt",final_matrix,fmt="%d")


if __name__ == '__main__':
	rate_matrix = np.loadtxt('100_missing_matrix.txt')
	top250_index = np.load('top250_movie_pos.npy')
	with open('tag_vector.json') as f:
		tag_vector_map = json.load(f)
	tag_list = []
	movie_list = []
	top250_vector = []
	for i in top250_index:
		top250_vector.append(np.zeros(100))
	with open('tag.txt') as f:
		for i,line in enumerate(f.readlines()):
			tag_list.append(line[:-2])

	with open('movie.txt') as f:
		for i,line in enumerate(f.readlines()):
			movie_list.append(line[:-2])		

	final_result = dict()
	for n,target in enumerate(top250_index):
		final_result[movie_list[target]] = []
		count = 0
		for i in range(rate_matrix[target].shape[0]):
			if rate_matrix[target][i] == 1:
				count += 1
				top250_vector[n] += tag_vector_map[tag_list[i].decode()]
		top250_vector[n] /= count


	for i in range(1,101):
		generate(rate_matrix,top250_vector,top250_index,tag_vector_map,100,i)
		