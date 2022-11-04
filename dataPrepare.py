import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from wordClean import replaceSpokesman, m_cut, getFeaturesWords
from configparser import ConfigParser

conf = ConfigParser()
conf.read("config.ini", encoding='UTF-8')

def get_vectors(words, pretrained_model, w2v_word_list):
    maxSeqLength = conf.getint("model", "maxSeqLength")
    vectorLength = conf.getint("model", "vectorLength")

    vectors = np.zeros([maxSeqLength, vectorLength])
    for i in range(min(maxSeqLength, len(words))):
        if words[i] in w2v_word_list:
            vectors[i] = pretrained_model.get_vector(words[i])
    return vectors

def getData(dataPath):


    marked_data = pd.read_excel(dataPath)

    marked_data['text'] = marked_data['text'].apply(replaceSpokesman)
    marked_data['text_cut'] = marked_data['text'].apply(m_cut)
    marked_data['text_cut'] = marked_data['text_cut'].apply(getFeaturesWords)

    data = marked_data[['text_cut', '品牌风格']]

    # 加载第三方预训练模型：
    pretrained_model_pat = conf.get("model", "pretrained_model_path")
    pretrained_model = KeyedVectors.load(pretrained_model_pat)
    w2v_word_list = list(pretrained_model.index_to_key)

    data['vectors'] = data.apply(lambda x: get_vectors(x['text_cut']
                                                       , pretrained_model
                                                       , w2v_word_list
                                                       )
                                 , axis=1
                                 )

    x = [i for i in data['vectors']]
    y = [i for i in data['品牌风格']]
    return x, y






