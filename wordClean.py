import jieba
import jieba.analyse as ana
import re
import pandas as pd
from configparser import ConfigParser

conf = ConfigParser()
conf.read("config.ini", encoding='UTF-8')

ana.set_stop_words(conf.get("data", "stopword2_path"))
myfont = r'C:\Windows\Fonts\msyh.ttc'
jieba.load_userdict(r"data/wordClean/附加词库.txt")

stop_table = pd.read_table(conf.get("data", "stopword_path"), names=['txt'], encoding='UTF-8')
stop_list = []

# stop_list = [word for word in stop_list['txt']]
for i in range(len(stop_table)):
    stop_list.append(stop_table['txt'][i])


def m_cut(text):
    return [word for word in jieba.lcut(text) if word not in stop_list]


cuttxt = lambda x: " ".join(m_cut(x))

brand_name = []
with open(conf.get("data", "brand_name"), 'r', encoding='utf-8') as f:
    for line in f:
        brand_name.append(line.strip())

spokesman = []
with open(conf.get("data", "spokesman_name"), 'r', encoding='utf-8') as f:
    for line in f:
        spokesman.append(line.strip())

features_words = []
with open(conf.get("data", "feature_words"), 'r', encoding='utf-8') as f:
    for line in f:
        features_words.append(line.strip())


def replaceSpokesman(df):
    for i in spokesman:
        df = df.replace(i, "代言人")
    for i in brand_name:
        df = df.replace(i, "品牌")
    return df


def cleanSpokesman(df):
    for i in spokesman:
        df = df.replace(i, "")
    return df


def getFeaturesWords(df):
    temp = []
    for i in df:
        if i in features_words: temp.append(i)
    return temp