# AD_KG
代码执行步骤：
1.xml-parse.py用于从解析PubMed上下载的XML文件，将摘要数据解析；
2.preprocessing-data.py 用于将摘要数据分词处理；
3.LDA-word2vec-AP-tfidf.py 使用LDA模型处理分词结果；
4.AP-twords-vec.py 将LDA结果聚类；
5.recognize_mesh.py将AP聚类结果中的医学实体识别出来。
