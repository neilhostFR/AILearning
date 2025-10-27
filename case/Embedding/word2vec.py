from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

sentences = LineSentence('./segment/segment_0.txt')
# print(sentences);

model = Word2Vec(
    sentences,
    vector_size=300,
    window=3,
    min_count=1
)

print(model.wv['曹操'])
print(model.wv.similarity('曹操', '关羽'))
print(model.wv.similarity('司马懿', '诸葛亮'))
print(model.wv.most_similar(positive=['曹操', '刘备'], negative=['张飞']))