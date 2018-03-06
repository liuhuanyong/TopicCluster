#coding=utf-8
import os,sys
reload(sys)
sys.setdefaultencoding('utf-8')
from gensim.models import LdaModel,TfidfModel,LsiModel
from gensim import similarities
from gensim import corpora

def create_data(corpus_path):#构建数据，先后使用doc2bow和tfidf model对文本进行向量表示
    sentences = []
    sentence_dict={}
    count=0
    for line in open(corpus_path):
       # print line
        line = line.strip().split('\t')
        if len(line) == 2:
            sentence_dict[count]=line[1]
            count+=1
            sentences.append(line[1].split(' '))
        else:
            break
    #对文本进行处理，得到文本集合中的词表
    dictionary = corpora.Dictionary(sentences)
    #利用词表，对文本进行cbow表示
    corpus = [dictionary.doc2bow(text) for text in sentences]
    #利用cbow，对文本进行tfidf表示
    tfidf=TfidfModel(corpus)
    corpus_tfidf=tfidf[corpus]
    return sentence_dict,dictionary,corpus,corpus_tfidf

def lda_model(sentence_dict,dictionary,corpus,corpus_tfidf,cluster_keyword_lda):#使用lda模型，获取主题分布   
    lda = LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=11)
    f_keyword = open(cluster_keyword_lda, 'w+')
    for topic in lda.print_topics(11,53):
        print '****'*5
        words=[]
        for word in topic[1].split('+'):
            word=word.split('*')[1].replace(' ','')
            words.append(word)
        f_keyword.write(str(topic[0])+'\t'+','.join(words)+'\n')
    #利用lsi模型，对文本进行向量表示，这相当于与tfidf文档向量表示进行了降维，维度大小是设定的主题数目  
    corpus_lda = lda[corpus_tfidf]
    for doc in corpus_lda:
        print len(doc),doc   
    return lda

def lsi_model(sentence_dict,dictionary,corpus,corpus_tfidf,cluster_keyword_lsi):#使用lsi模型，获取主题分布
    lsi = LsiModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=11)
    f_keyword = open(cluster_keyword_lsi, 'w+')
    for topic in lsi.print_topics(11,50):
        print topic[0]
        words=[]
        for word in topic[1].split('+'):
            word=word.split('*')[1].replace(' ','')
            words.append(word)
        f_keyword.write(str(topic[0])+'\t'+','.join(words)+'\n')
   
    return lsi


if __name__=="__main__":
    corpus_path = "./corpus_train.txt"
    cluster_keyword_lda = './cluster_keywords_lda.txt'
    cluster_keyword_lsi = './cluster_keywords_lsi.txt'
    sentence_dict,dictionary,corpus,corpus_tfidf=create_data(corpus_path)
    lsi_model(sentence_dict,dictionary,corpus,corpus_tfidf,cluster_keyword_lsi)
    lda_model(sentence_dict, dictionary, corpus, corpus_tfidf,cluster_keyword_lda)

