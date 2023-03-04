
from glob import glob
import os
import random
import pandas as pd
from pytorch.bilstm.config import *


# 根据标注文件生成对应关系
def get_annotation(ann_path):
    with open(ann_path, encoding='utf-8') as file:
        anns = {}
        for line in file.readlines():
            # print(line.split(' '))
            # exit()
            arr = line.split('\t')[1].split()
            # print(arr)
            # exit()
            name = arr[0]
            start = int(arr[1])
            end = int(arr[-1])

            # 标注太长，可能有问题
            if end - start > 50:
                continue
            anns[start] = 'B-' + name
            for i in range(start + 1, end):
                anns[i] = 'I-' + name
        return anns

def get_text(txt_path):
    with open(txt_path, encoding='utf-8') as file:
        return file.read()

# 建立文字和标签对应关系
def generate_annotation():
    for txt_path in glob(ORIGIN_DIR + '*.txt'):
        ann_path = txt_path[:-3] + 'ann' # 去掉后三位的txt+上ann可以得到ann的文件
        anns = get_annotation(ann_path)
        text = get_text(txt_path)

        # 建立文字和标注对应
        df = pd.DataFrame({'word': list(text), 'label': ['O'] * len(text)})  #  做好对应  现将文本转为为list， 然后统一都变成o，到后期在对特殊标签的位置在进行替换
        df.loc[anns.keys(), 'label'] = list(anns.values())
        # print(list(df.head(100)['label']))  # 一种比较好的查看方法
        # exit()

        # 导出文件
        file_name = os.path.split(txt_path)[1]
        print(os.path.split(txt_path))
        exit()
        df.to_csv(ANNOTATION_DIR + file_name, header=None, index=None)

# 拆分训练集和测试集
def split_sample(test_size=0.3):
    files = glob(ANNOTATION_DIR + '*.txt')
    random.seed(0)
    random.shuffle(files)
    n = int(len(files) * test_size)
    test_files = files[:n]
    train_files = files[n:]
    # 合并文件
    merge_file(train_files, TRAIN_SAMPLE_PATH)
    merge_file(test_files, TEST_SAMPLE_PATH)


def merge_file(files, target_path):
    with open(target_path, 'a', encoding='utf-8-sig', errors='ignore') as file:
        for f in files:
            text = open(f, encoding='utf-8').read()
            file.write(text)

# 生成词表
def generate_vocab():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[0], names=['word'])
    # print(df['word'].value_counts().keys().tolist())
    # exit()
    vocab_list = [WORD_PAD, WORD_UNK] + df['word'].value_counts().keys().tolist()
    # print(len(vocab_list))
    # exit()
    vocab_list = vocab_list[:VOCAB_SIZE]
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab = pd.DataFrame(list(vocab_dict.items()))
    # print(vocab)
    # exit()
    vocab.to_csv(VOCAB_PATH, header=None, index=None)

# 生成标签表
def generate_label():
    df = pd.read_csv(TRAIN_SAMPLE_PATH, usecols=[1], names=['label'])
    label_list = df['label'].value_counts().keys().tolist()
    label_dict = {v: k for k, v in enumerate(label_list)}
    label = pd.DataFrame(list(label_dict.items()))
    label.to_csv(LABEL_PATH, header=None, index=None)


if __name__ == '__main__':
    # anns = get_annotation('./input/origin/0.ann')
    # print(anns)

    # get_annotation('./input/origin/0.ann')
    # 建立文字和标签对应关系
    # generate_annotation()

    # 拆分训练集和测试集
    split_sample()

    # 生成词表
    # generate_vocab()

    # 生成标签表
    # generate_label()

















