"""
读电影详细属性文件，划分出最热门的1000部电影，并准备libsvm数据格式
"""

from count import *
from movie_redundancy.tagRank import *
from movie_redundancy.prepare_data import array2file


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


def prepare_data(round_matrix):

    test_matrix = np.zeros((1000, 2015), dtype=np.float32)
    test_y = np.zeros((1000, 1), dtype=np.float32)

    remove_pos_list = []

    most_popular_movie_list = get_most_popular_movie_id_list()

    count = 0
    for item in most_popular_movie_list:

        # print(count)

        if count >= 1000:
            break

        try:
            cur_row_pos = movie_order_dict.get(item, -1)
            if cur_row_pos > -1:
                cur_row = user_item_matrix[cur_row_pos]
                test_matrix[count] = cur_row

            cur_y_pos = redundancy_movie_dict[item]
            cur_y = movie_redundancy_y[cur_y_pos]
            test_y[count] = cur_y

            remove_pos_list.append(cur_y_pos)
            count += 1
        except Exception:
            print(item)

    train_matrix = np.delete(round_matrix, remove_pos_list, 0)
    train_y = np.delete(movie_redundancy_y, remove_pos_list, 0)

    return train_matrix, test_matrix, train_y, test_y


def prepare_ori_data(round_matrix, round_y):

    test_matrix = np.zeros((1000, 2015), dtype=np.float32)
    test_y = np.zeros((1000, 1), dtype=np.float32)

    remove_pos_list = []

    most_popular_movie_list = get_most_popular_movie_id_list()

    count = 0
    for item in most_popular_movie_list:

        # print(count)

        if count >= 1000:
            break

        try:
            cur_row_pos = movie_order_dict.get(item, -1)
            if cur_row_pos > -1:
                cur_row = user_item_matrix[cur_row_pos]
                test_matrix[count] = cur_row
                test_y[count] = movie_rates_y[cur_row_pos]

                remove_pos_list.append(movie_rate_pos_dict[item])
                count += 1
        except Exception:
            print(item)

    train_matrix = np.delete(round_matrix, remove_pos_list, 0)
    train_y = np.delete(round_y, remove_pos_list, 0)

    return train_matrix, test_matrix, train_y, test_y


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


if __name__ == "__main__":
    # print(1)
    X_train, X_test, y_train, y_test = prepare_data(m1matrix_2)
    X_train, y_train = shuffle_in_unison(X_train, y_train)
    array2file(X_train, y_train, redundancy_output_dir + '\\svm\\m1_train_2.data')
    array2file(X_test, y_test, redundancy_output_dir + '\\svm\\m1_test_2.data')

    X_train, X_test, y_train, y_test = prepare_data(m2matrix_2)
    X_train, y_train = shuffle_in_unison(X_train, y_train)
    array2file(X_train, y_train, redundancy_output_dir + '\\svm\\m2_train_2.data')
    array2file(X_test, y_test, redundancy_output_dir + '\\svm\\m2_test_2.data')

    X_train, X_test, y_train, y_test = prepare_data(m4matrix_2)
    X_train, y_train = shuffle_in_unison(X_train, y_train)
    array2file(X_train, y_train, redundancy_output_dir + '\\svm\\m4_train_2.data')
    array2file(X_test, y_test, redundancy_output_dir + '\\svm\\m4_test_2.data')

    X_train, X_test, y_train, y_test = prepare_data(m8matrix_2)
    X_train, y_train = shuffle_in_unison(X_train, y_train)
    array2file(X_train, y_train, redundancy_output_dir + '\\svm\\m8_train_2.data')
    array2file(X_test, y_test, redundancy_output_dir + '\\svm\\m8_test_2.data')

    X_train, X_test, y_train, y_test = prepare_ori_data(movie_rates_matrix, movie_rates_y)
    X_train, y_train = shuffle_in_unison(X_train, y_train)
    array2file(X_train, y_train, redundancy_output_dir + '\\svm\\ori_train.data')
    array2file(X_test, y_test, redundancy_output_dir + '\\svm\\ori_test.data')
