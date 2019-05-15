workdir = 'D:\\workProject\\pythonProject\\Mytagging'
datadir = 'D:\\workData\\movieTagging'
redundancydir = 'D:\\workData\\movieTagging\\video redundancy'
tensorflow_data_dir = "D:\\workProject\\pythonProject\\Mytagging_part_2\\data_2"
tensorflow_data_3_dir = "D:\\workProject\\pythonProject\\Mytagging_part_2\\data_3"
baseline_dir = "D:\\workProject\\pythonProject\\Mytagging\\baselines"
svdpp_data_dir = tensorflow_data_dir + "\\SVD++"
own_dir = tensorflow_data_dir + "\\own"
server_dir = tensorflow_data_dir + "\\server"
own3_dir = tensorflow_data_3_dir + "\\own"
server3_dir = tensorflow_data_3_dir + "\\server"

all_tags_file = datadir + '\\all_tag_search.json'
all_movies_dir = datadir + '\\电影列表'
# movie_list_dir = datadir + '\\电影列表\\'
movie_review_dir = datadir + '\\长评'
movie_detail_dir = datadir + '\\电影详细属性'
movie_tags_extension_dir = datadir + '\\标签扩展'

redundancy_matrix = redundancydir + '\\matrix.txt'
redundancymovie_id = redundancydir + '\\movie_id_duplicate.txt'
redundancymovie_id_re = redundancydir + '\\movie_id.txt'

redundancy_output_dir = workdir + '\\movie_redundancy\\output'
redundancy2_output_dir = workdir + '\\movie_redundancy_2\\output'
SVD_output_dir = workdir + '\\SVD\\output'
SVD2_output_dir = workdir + '\\SVD_2\\output'
SVD2_own_output_dir = SVD2_output_dir + '\\own'
SVD3_output_dir = workdir + '\\SVD_3\\output'
SVD3_own_output_dir = SVD3_output_dir + '\\own'
baseline_output_dir = baseline_dir + "\\output"