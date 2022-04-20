from gensim import models,corpora,similarities
from gensim.models import LdaModel
import os
from nltk.corpus import stopwords
import nltk
from gensim.parsing.preprocessing import STOPWORDS
import re
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def is_stop_words(word):
    """
    判断是否为停用词
    :param word:
    :return:
    """
    word = str(word).strip()
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '+', '/-', 'and/or', 'their', 'they', 'They', 'Their',
                            ' ', "''", "±", "vs.", 'p', 'v', 'n', 'vs', 'we', 'We', '>', '--', 'c', 'IV', 'i/r', '≥', 'µg/ml', 'kg/m', 'S', 'H.', 'III/IV',
                            "'", '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '<', "`", "''", '=', 'I', 'II', '–']
    regex = re.compile(r'[\d]*[\.]?[\d]+')
    stoplist = list(STOPWORDS) + stopwords.words('english') + english_punctuations
    retex_w = re.compile(r'[\w]+')
    if str(word) in stoplist:
        return True
    elif str(word).isdigit():
        return True
    elif len(regex.findall(str(word)))>0:
        return True
    elif len(str(word)) == 1 or len(retex_w.findall(str(word)))<1:
        return True
    else:
        return False
    return False

def process_word_type(word):
    """
    将非缩写形式的单词变为小写形式
    :param word:
    :return:
    """
    if str(word).isupper():
        return str(word)
    else:
        return str(word).lower()

def token_abstract(abstract):
    """
    将摘要数据进行分词处理，并返回分词结果
    :param abstract:
    :return:
    """
    abstract = str(abstract).replace("\n", "")
    abstract = str(abstract).replace(".,", ".")
    abstract = str(abstract).replace(" 's", "_s")
    word_token = nltk.word_tokenize(abstract)
    word_token = [process_word_type(word) for word in word_token]  ##将非大写单词替换为小写形式

    return  word_token

def process_doc(inputpath, tokenfile):
    """
    将目录下的所有文件进行分词处理，并将结果去除停用词后list形式返回，用于lda模型，同时将分词结果保存在文档中，用于word2vec模型训练
    :param inputpath:
    :param tokenfile:
    :return:
    """
    doc = [] ##一年的文档集合
    files = os.listdir(inputpath)
    for file in files:
        filepath = os.path.join(inputpath, file)
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            data = f.readlines()
        abstract = " ".join(data)
        word_tokens = token_abstract(abstract)

        with open(tokenfile, 'a', encoding='utf-8-sig') as fw:
            fw.write(' '.join(word_tokens) + ' ')

        word_tokens = [word for word in word_tokens if not is_stop_words(word)]  ##去除停用词
        doc.append(word_tokens)
    return doc

def tfidf_lda(doc, outfile, topicnums, alpha, gamma, passes, iteration):
    """
    使用tf-idf进行lda训练
    :param doc:
    :param outfile:
    :param topicnums:
    :param alpha:
    :param gamma:
    :param passes:
    :param iteration:
    :return:
    """
    dictionary = corpora.Dictionary(doc)
    corpus = [dictionary.doc2bow(text) for text in doc]

    ## 初始化tf-idf
    tfidf = models.TfidfModel(corpus)

    ## Transforming vectors
    corpus_tfidf = tfidf[corpus]

    ## train lda
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=topicnums, alpha=alpha, eta=gamma, passes=passes, iterations=iteration)
    with open(outfile, 'w', encoding='utf-8-sig') as fw:
        for i in range(topicnums):
            topic = lda.get_topic_terms(i, topn=30)
            topic = [str(dictionary[x[0]])+";" + str(x[1]) for x in topic]
            fw.write("topic " + str(i) + "\t" + "\t".join(topic) + "\n")


def lda_doc(doc, modelfile, outfile, topicnums, alpha, gamma, passes, iteration):
    """
    训练LDA
    :param doc:
    :param modelfile:
    :param outfile:
    :param topicnums:
    :param alpha:
    :param iteration:
    :return:
    """

    dictionary = corpora.Dictionary(doc)
    corpus = [dictionary.doc2bow(text) for text in doc]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=topicnums, alpha=alpha, eta=gamma, passes=passes, iterations=iteration)
    ##保存模型
    #lda.save(modelfile)
    ##将结果写入文件

    # for i in range(topicnums):
    #     #print(lda.get_topic_terms(i, topn=10))
    #     topic = lda.get_topic_terms(i, topn = 10)
    #     topic = [(dictionary[x[0]], x[1]) for x in topic ]
    #     print(topic)

    with open(outfile, 'w', encoding='utf-8-sig') as fw:
        for i in range(topicnums):
            topic = lda.get_topic_terms(i, topn=30)
            topic = [str(dictionary[x[0]])+";" + str(x[1]) for x in topic]
            fw.write("topic " + str(i) + "\t" + "\t".join(topic) + "\n")

    # corpus_words = sum(cnt for document in corpus for _, cnt in document)
    # Per_wordPerplexity = lda.log_perplexity(corpus)
    # print(Per_wordPerplexity, corpus_words)
    # return Per_wordPerplexity, corpus_words

def train_word2vec(tokenfile, word_size, outfile):
    """
    训练词向量
    :param tokenfile:
    :param word_size:
    :param outfile:
    :return:
    """
    sg=1
    window = 10
    min_count = 0
    model = Word2Vec(LineSentence(tokenfile), sg=sg, hs=1, min_count=min_count, window=window, size=word_size)
    model.save(outfile)


if __name__ == '__main__':
    inputpath = './data/abstracts-lemmatize'
    outpath = './data/twords-freq'
    token_path = './data/tokens'
    modle_path = './data/word2vec-models'
    tfidf_path = './data/twords-tfidf'
    passes = 5
    iteration = 500
    years = os.listdir(inputpath)
    topicnum = 10
    alpha = 1.25
    tf_alpha = 1.25
    eta = 0.01
    word_size=300


    for year in years:
        print(year)
        yearpath = os.path.join(inputpath, year)
        tokenfile = os.path.join(token_path, year+'_tokens')

        doc = process_doc(yearpath, tokenfile)
        print('doc length: ',len(doc))
        twordsfile = os.path.join(outpath, year + "_" + str(topicnum) + "_" + str(alpha) + "_" + str(eta) + "_twords")
        lda_doc(doc, '', twordsfile, topicnum, alpha, eta, passes, iteration)
        print("lda: ", twordsfile)
        modelfile = os.path.join(modle_path, year+"_"+str(word_size)+"_model")
        train_word2vec(tokenfile, word_size, modelfile)
        print("train wordvec")
        tfidf_twords = os.path.join(tfidf_path, year + "_" + str(topicnum) + "_" + str(tf_alpha) + "_" + str(eta) + "_twords")
        tfidf_lda(doc, tfidf_twords, topicnum, tf_alpha, eta, passes, iteration)
        print("tfidf: ", tfidf_twords)



