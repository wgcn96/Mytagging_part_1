"""
根据影评记录获取UI矩阵， 保存数据进行分类或回归
"""

import collections
import jieba
import jieba.analyse
import time

from count import *

uiarray_movie_dict = {}     # 根据名字取下标
uiarray_movie_name = []     # 根据下标取名字
y = np.zeros([count_all_movies, 1], dtype=np.float32)       # 评分预测值，根据下标取得分
uiarray = np.zeros([count_all_movies, tags_set.__len__()], dtype=np.int32)
review_dir = walk_into_dir(movie_review_dir)
file_count = check_count = complementary_count = none_in_set_count = 0
start_time = time.time()

for root_dir in review_dir:
    files = get_files(root_dir)
    for file in files:
        print("current file", file)
        file_root = os.path.join(root_dir,  file)
        f = open(file_root, encoding='utf-8')
        content = json.load(f)
        data = content["data"]  # data是一个list，每条记录为一个 dict
        if len(data):
            current_com_count = 0   # 当前电影被补的标签数
            file_name = os.path.splitext(file)[0]
            uiarray_movie_dict[file_name] = file_count      # 正反两取
            uiarray_movie_name.append(file_name)

            if file_name in all_tags_content.keys():        # 原有的标签
                tags = all_tags_content[file_name]
                for tag in tags:
                    tag_pos = tag_order_dict[tag]
                    uiarray[file_count, tag_pos] += 1
                    check_count += 1

            text = ''       # 评论文本拼接
            tag_file_path = os.path.join(movie_tags_extension_dir, file_name+'.txt')
            tag_file = open(tag_file_path, 'w', encoding='utf-8')
            for onereview in data:
                onereview = onereview.strip()
                text += onereview

            mytags = jieba.analyse.extract_tags(text)       # 提取关键词
            extraction_tags = ' '.join(mytags)
            tag_file.write(extraction_tags + '\n')
            complementary_tags_list = []
            for tag in mytags:
                if tag in tags_set:
                    tag_pos = tag_order_dict[tag]
                    if uiarray[file_count, tag_pos] != 1:       # 补标签
                        complementary_tags_list.append(tag)
                        uiarray[file_count, tag_pos] = 1
                        current_com_count += 1
                else:
                    none_in_set_count += 1
            complementary_count += current_com_count
            complementary_tags = ' '.join(complementary_tags_list)
            tag_file.write(complementary_tags + '\n')
            tag_file.write(str(current_com_count))
            tag_file.close()

            rate = root_dir.split('\\')[-1]
            y[file_count] = float(rate)
            file_count += 1

        f.close()

print("有影评的电影数，有影评的电影打标签次数，补标签次数，提取的标签不在集合中的数目：", end='')
print(file_count, check_count, complementary_count, none_in_set_count)
print()
print("total time:", time.time() - start_time)
uiarray = uiarray[:file_count, ]     # 切片
y = y[:file_count, ]

# np.save(os.path.join(workdir, "after_review_tags.npy"), uiarray)
# np.save(os.path.join(workdir, "after_review_y.npy"), y)

