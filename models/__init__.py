import gensim


W2V_PATH = '/Users/andreyrumyantsev/Downloads/all.norm-sz100-w10-cb0-it1-min100.w2v'
W2V_SIZE = 100
modelw2v = gensim.models.KeyedVectors.load_word2vec_format(W2V_PATH, binary=True, unicode_errors='ignore')