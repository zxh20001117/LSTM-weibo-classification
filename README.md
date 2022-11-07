# LSTM-weibo-classification

## data source
从微博社交平台上，精准搜索之后，通过API进行的爬取
然后进行人工标记，针对标签种类进行0-1分类打标签，有标签的数据为8000+条

## word2vec 模型
直接使用预训练的300维的word2vec向量

## 分词与预处理
jieba分词库，需要一个通用的停用词词库与定制化的停用词词库
对文本预处理的时候，需要去除html标签，去除标点符号和特殊表情

## LSTM输入数据维度
将一句话中最多10个词的word2vec向量按照顺序组合成10*300的输入维度，缺词或者长度不足10，都用0向量替代，超过10个词的句子直接截断。