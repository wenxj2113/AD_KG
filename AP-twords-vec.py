from gensim.models.word2vec import Word2Vec
from gensim import models,corpora,similarities
import numpy as np
import os
import sklearn.metrics as metrics
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity

def lda_vector(inputfile, wordvecfile, ap_path, topics_num, size_word):
    """
    将lda的结果中的每个单词转换为word2vec的结果，并将结果保存，用于聚类使用
    :param inputfile:
    :param wordvecfile:
    :param vecfile:
    :return:
    """
    ##读取lda的结果，并将结果保存在list中
    twords = []
    with open(inputfile, 'r', encoding='utf-8-sig') as f:
        data = f.readlines()
    for line in data:
        line = str(line).replace("\n", "")
        topic_words = line.split("\t")
        topics = []
        ##从第二个数据开始为topic中的单词以及对应的频率
        for i in range(1, len(topic_words)):
            word_freqs = topic_words[i]
            word = word_freqs.split(";")[0]
            freq = (float)(word_freqs.split(";")[1])
            topics.append((word, freq))
        twords.append(topics)

   # print("映射主题到word2vec空间")
    the_id = [] ## 每个主题的的单词
    the_vl = [] ##每个主题的单词的value
    the_w = [] ##每个主题的单词的占权重

  #  print("计算主题内每个词的权重")
    for i in range(len(twords)):
        the_id.append([xx[0] for xx in twords[i]])
        the_sum = sum([xx[1] for xx in twords[i]])
        the_w.append([xx[1]/the_sum for xx in twords[i]])

   # print("开始映射到坐标")
    m = 0

    ##加载word2vec模型
    model_wv = Word2Vec.load(wordvecfile)
    the_wv = np.zeros([topics_num, size_word]) ## 每个主题映射到word2vec,主题数，word2vec
    ##主题下每个单词在word2vec下坐标加权求和
    # try:
    #     for words in the_id:
    #         n = 0
    #         for word in words:
    #             # print(word)
    #             the_wv[m] += [x_word * the_w[m][n] for x_word in model_wv[word]]
    #             n += 1
    #         m += 1
    # except BaseException as e:
    #     with open(error_file, 'a', encoding='utf-8-sig') as fe:
    #         fe.write(modelname + "\t" + str(e)+ "\n")
    for words in the_id:
        n = 0
        for word in words:
           # print(word)
            the_wv[m] += [x_word*the_w[m][n] for x_word in model_wv[word]]
            n += 1
        m += 1

    ##进行聚类分析

    #ap = AffinityPropagation(preference=preference, affinity='precomputed').fit(the_wv)
    sim = cosine_similarity(the_wv)
    # print("max：", np.max(sim))
    # print("min: ", np.min(sim))
    # print("mean: ", np.mean(sim))
    # print("median: ", np.median(sim))
    preference = np.median(sim)
    for i in range(len(sim)):
        sim[i][i] = preference

    # print("相似矩阵：")
    # print(sim)
    ap = AffinityPropagation(affinity='precomputed').fit(sim)
    cluster_centers_indices = ap.cluster_centers_indices_
    #print("center: ", cluster_centers_indices)
    labels = ap.labels_
    n_clusters = len(cluster_centers_indices)

    # score = metrics.silhouette_score(the_wv, labels, metric='cosine')
    # score2 = metrics.silhouette_score(the_wv, labels, metric='euclidean')
    # score3 = metrics.calinski_harabaz_score(the_wv, labels)
    # print("score: ", score)
    # with open(vecfile, 'a', encoding='utf-8-sig') as f:
    #     f.write(str(modelname) + "\t" + str(n_clusters) + "\t" + str(score) + "\t" + str(score2) + "\t" + str(score3) + "\n")
    #print("n_clusters: ", n_clusters)
    #print("labels: ", labels)
    # for i in range(n_clusters):
    #     center_topic = twords[cluster_centers_indices[i]]
    #     print("center topic ",cluster_centers_indices[i],center_topic)

    cluster_dict = {}
    for i in range(len(labels)):
        center_indice = labels[i]
        topic = twords[i]
        center_words = cluster_dict.get(center_indice)
        if center_words == None:
            center_words = {}
        ##将topic中的单词写入该簇类中，同样的单词频率进行累加
        for word, freq in topic:
            center_word_freq = center_words.get(word)
            if center_word_freq == None:
                center_word_freq = (freq, 1)
            else:

                center_word_freq = (center_word_freq[0] + freq, center_word_freq[1] + 1)
            center_words[word] = center_word_freq
       # print("center words: ", center_words)
        cluster_dict[center_indice] = center_words

    result_dict = {}
    for center in cluster_dict:
        center_word = cluster_dict.get(center)
       # print("center word: ",center_word)
        center_freq = {}
        for item in center_word:
            freq = center_word.get(item)[0]/center_word.get(item)[1]
          #  print("freq: ", freq)
            center_freq[item] = freq
        result_dict[center] = center_freq
    print("result: ", result_dict)
    #cluster_dict = sorted(cluster_dict.items(), key=lambda d: d[1], reverse=True)



    # for i in range(n_clusters):
    #     cluster_center = the_wv[cluster_centers_indices[i]]
    #     print("i:",i, "cluster_center: ",cluster_center)

    #将每个主题转换为word2vec坐标的结果保存
    with open(ap_path+"_"+str(n_clusters), 'w', encoding='utf-8-sig')as fw:
        for item in result_dict:
            topic_cluster = sorted(result_dict.get(item).items(), key=lambda d: d[1], reverse=True)
            print(topic_cluster)
            for word, freq in topic_cluster:
                fw.write(word + "," + str(freq) + "\t")
            fw.write("\n")
        # for topic in the_wv:
        #     fw.write(str(topic) + "\n")

def compute_cosine_similarity(the_wv, preference):
    """
    计算余弦相似度矩阵
    :param the_wv:
    :return:
    """
    sim = cosine_similarity(the_wv)
    for i in range(len(sim)):
        sim[i][i] = preference
    return sim

if __name__ == '__main__':

    twords_path = './data/twords-freq'
    word2vec_path = './data/word2vec-models'
    ap_result_path = './data/ap-results-freq'
    topics_num = 100
    size = 300
    for year in range(2008, 2020, 1):
        tword_file = os.path.join(twords_path, str(year) + '_100_0.125_0.01_twords')
        model_file = os.path.join(word2vec_path, str(year) + '_300_model')

        ap_file_path = os.path.join(ap_result_path, str(year)+"_ap")

        lda_vector(tword_file, model_file, ap_file_path, topics_num, size)



    ap_path = './data/ap-split-freq'
    ap_files = os.listdir(ap_result_path)
    for file in ap_files:
        ap_file = os.path.join(ap_result_path, file)
        with open(ap_file, 'r', encoding='utf-8-sig') as f:
            data = f.readlines()
        index = 0
        for line in data:
            line = str(line).replace("\n", "")
            ap_out_file = os.path.join(ap_path, file + "_" + str(index))
            words_freq = line.split("\t")
            with open(ap_out_file, 'w', encoding='utf-8-sig') as fw:
                for word_freq in words_freq:
                    fw.write(word_freq.replace(",", "\t") + "\n")

            index += 1