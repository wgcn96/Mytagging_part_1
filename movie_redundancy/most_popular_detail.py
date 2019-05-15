
from static import *
from count import *


print("hello world")


def get_most_popular_movie_dict():
    most_popular_movie_dict = {}
    most_popular_movie_rate_dict = {}

    files = get_files(movie_detail_dir)
    for file in files:
        # print("current file", file)
        file_root = os.path.join(movie_detail_dir, file)
        f = open(file_root, encoding='utf-8')
        content = json.load(f)
        data = content["data"]  # data是一个list，每条记录为一个 dict
        for one_movie_detail_dict in data:
            movie_id = one_movie_detail_dict["id"]
            if movie_id == "":
                continue
            movie_collection_count = one_movie_detail_dict["collect_count"]
            movie_rate = one_movie_detail_dict["rating"]["average"]
            if movie_collection_count == "":
                continue
            try:
                most_popular_movie_dict[movie_id] = int(movie_collection_count)
                most_popular_movie_rate_dict[movie_id] = float(movie_rate)
            except Exception:
                print(movie_id)
        f.close()

    most_popular_movie_turple_list = sorted(most_popular_movie_dict.items(), key=lambda item:item[1], reverse=True)

    return most_popular_movie_dict, most_popular_movie_rate_dict, most_popular_movie_turple_list


most_popular_movie_dict, most_popular_movie_rate_dict, most_popular_movie_turple_list = get_most_popular_movie_dict()


def get_most_popular_movie_id_list():
    result = []
    for item in most_popular_movie_turple_list:
        movie_id = item[0]
        result.append(movie_id)
    return result

most_popular_movie_list = get_most_popular_movie_id_list()


def tag2ListNFile(filePath):
    f = open(filePath, 'w')
    tag_list = []
    for tag, pos in tag_order_dict.items():
        tag_list.append(tag)
        line = tag + ' ' + str(pos) + '\n'
        f.write(line)
    f.close()
    print(tag_list.__len__())
    return tag_list


tag_file = os.path.join(redundancy_output_dir, 'tag.txt')
tag_list = tag2ListNFile(tag_file)


filePath = os.path.join(redundancy_output_dir, "most_popular_movie.json")
f = open(filePath, encoding='utf-8', mode='w')

most_popular_movie_dict = {}


for i in range(1000):
    movie_tag_list = []
    cur_movie = most_popular_movie_list[i]
    cur_rate = most_popular_movie_rate_dict[cur_movie]
    tag_pos = movie_rate_pos_dict[cur_movie]

    cur_tags = movie_rates_matrix[tag_pos]
    tag_indexes = cur_tags.tolist()
    for pos, indicator in enumerate(tag_indexes):
        if indicator:
            movie_tag_list.append(tag_list[pos])
        else:
            pass
    print(tag_pos, len(movie_tag_list))
    most_popular_movie_dict[cur_movie] = {}
    most_popular_movie_dict[cur_movie]["rate"] = cur_rate
    most_popular_movie_dict[cur_movie]["tags"] = movie_tag_list
    print(most_popular_movie_dict[cur_movie])
json.dump(most_popular_movie_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
f.close()

print("finish")
