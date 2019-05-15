
import json
import re

import jieba
import jieba.analyse

from static import *

'''
text = "故宫的著名景点包括乾清宫、太和殿和午门等。其中乾清宫非常精美，午门是紫禁城的正门，午门居中向阳。"
seg_list = jieba.cut(text, cut_all=False)
print("分词结果:")

print("/".join(seg_list))


# 获取关键词
tags = jieba.analyse.extract_tags(text, topK=3)
print("关键词:")
print(" ".join(tags))

'''

'''
def segment():
    file_nums = 0
    count = 0
    url = base_url + 'processed_data/demo/'
    fileNames = os.listdir(url)
    for file in fileNames:
        logging.info('starting ' + str(file_nums) + 'file word Segmentation')
        segment_file = open(url + file + '_segment', 'a', encoding='utf8')
        with open(url + file, encoding='utf8') as f:
            text = f.readlines()
            for sentence in text:
                sentence = list(jieba.cut(sentence))
                sentence_segment = []
                for word in sentence:
                    if word not in stopwords:
                        sentence_segment.append(word)
                segment_file.write(" ".join(sentence_segment))
            del text
            f.close()
        segment_file.close()
        logging.info('finished ' + str(file_nums) + 'file word Segmentation')
        file_nums += 1
'''

url = datadir + '\\长评\\2.4\\25727236.json'
with open(url, encoding='utf8') as f:
    content = json.load(f)
    text = content['data']
    result = []
    r1 = u'[a-zA-Z0-9’!"#$%&\'()（）~·*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    r2 = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    for sentence in text:
        sentence = sentence.strip()
        sentence = re.sub(r1, '', sentence)
        seg_list = list(jieba.cut(sentence))
        result.extend(seg_list)
    print("/".join(result))
