"""
提取电影类型数据
"""

from static import *
from count import *
from review_extension.data1 import *


movie_geners_dict = {}
geners_dict = collections.OrderedDict()
geners_set = set()
movie_detail_files = get_files(movie_detail_dir)
geners_count = 0        # 统计没有类型数据的影片的数量
count = 0
for file in movie_detail_files:
    file_path = os.path.join(movie_detail_dir, file)
    f = open(file_path, encoding='utf-8')
    content = json.load(f)
    data = content['data']
    for movie_detail_dict in data:
        count += 1
        name = movie_detail_dict['id']
        geners = movie_detail_dict['genres']
        movie_geners_dict[name] = geners
        # print(name, geners)       # for debug
        if geners:
            geners_set.update(geners)
        else:
            geners_count += 1
    f.close()

print("有电影类型属性的电影数，电影")
print(count, geners_count)
print()

item_pos = 0
for item in geners_set:
    geners_dict[item] = item_pos
    item_pos += 1

none_genres_count = 0
for movie_name in uiarray_movie_name:
    y_genres = []
    if movie_name in movie_geners_dict.keys():
        current_genres = movie_geners_dict[movie_name]
        for item in current_genres:
            y_genres.append(geners_dict[item])
    else:
        none_genres_count += 1
        y_genres.append([])
print(none_genres_count)
