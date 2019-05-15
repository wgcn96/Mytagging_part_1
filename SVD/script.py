
from SVD.tag_extension import *

matrix = np.load(os.path.join(workdir, 'wt_bias.npy'))
matrix, extension_matrix = tag_extension(matrix)
all_movie_pos_file_Path = os.path.join(workdir, "all_movie_matrix_dict.json")
all_movie_pos_file = open(all_movie_pos_file_Path, encoding='utf-8', mode='r')
all_movie_pos_dict = json.load(all_movie_pos_file)

tagFile = os.path.join(redundancy_output_dir, 'tag.txt')
tag_list = loadTagList(tagFile)
SVD_movie_result = os.path.join(SVD_output_dir, "SVD_movie_wt_bias_result.json")
matrix2json(extension_matrix, all_movie_pos_dict, tag_list, SVD_movie_result)
