import os
import nltk
import spacy
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag

def load_spacy_nlp(spacy_path):
    nlp = spacy.load(spacy_path)
    return nlp

def spacy_abstract(abstract, nlp):
    '''
    使用依存分析进行词形还原，是否加入停用词判断
    :param abstract: 还原的摘要数据
    :param nlp: 依存分析变量
    :return: 依存分析后的结果
    '''
    abstract_doc = nlp(str(abstract))
    abstract_lemma = []
    for token in abstract_doc:
        if token.lemma_ != '-PRON-':
            abstract_lemma.append(token.lemma_)
        else:
            abstract_lemma.append(str(token))
    abstract_string = " ".join(abstract_lemma)
    abstract_string = abstract_string.replace(" / ", "/")
    abstract_string = abstract_string.replace(" ,", ",")
    abstract_string = abstract_string.replace(" .", ".")
    abstract_string = abstract_string.replace(" %", "%")
    abstract_string = abstract_string.replace(" '", "'")
    abstract_string = abstract_string.replace(" - ", "-")
    abstract_string = abstract_string.replace("  ", " ")
    return abstract_string

def tuple_phrase(inputpath, tuplefile, outpath):
    """
    将输入路径下的所有摘要数据按照词组文件中的词组进行分词处理，并将结果保存在输出目录下
    :param inputpath:
    :param tuplefile:
    :param outpath:
    :return:
    """
    tuples = pickle.load(open(tuplefile, 'rb')) ##词组数据
    mwe_tokenizer = nltk.tokenize.MWETokenizer(tuples) ##分词工具，将词组合并为一个单词，用下划线连接

    years = os.listdir(inputpath)
    for year in years:
        yeardir = os.path.join(inputpath, year)
        files = os.listdir(yeardir)
        ##输出目录
        outdir = os.path.join(outpath, year)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for file in files:
            filepath = os.path.join(yeardir, file)
            ##获取数据
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                data = f.readlines()
            abstract = " ".join(data)
            abstract = str(abstract).replace("\n", "")

            ##将词组合并为一个单词
            word_token = nltk.word_tokenize(abstract)
            wordlist = mwe_tokenizer.tokenize(word_token)
            result_abstract = " ".join(wordlist)
            ##结果文件
            outfile = os.path.join(outdir, file)
            with open(outfile, 'w', encoding='utf-8-sig') as fw:
                fw.write(str(result_abstract))

def tuple_test_file():
    """
    测试合并词组
    :return:
    """
    tuplefile = './category_dicts/mesh_tuples.pkl'
    test_abstract = '''
        Deficiency of transcriptional regulator p8 induces autophagy and causes impaired cardiac function. 
        Through autophagy cells adapt to nutrient availability, recycle cellular material and eliminate toxic 
        proteins and damaged cellular organelles. Dysregulation of autophagy is implicated in the pathogenesis 
        of various diseases, including cancer, neurodegeneration and cardiomyopathies. The transcription factor 
        FoxO3 activates autophagy by enhancing the expression of several genes. We find a role for the transcriptional 
        regulator p8 in controling autophagy by repressing FoxO3 transcriptional activity. p8 silencing increases 
        the association of FoxO3 with the bnip3 promoter, a known pro-autophagic FoxO3 target, and results in
         increasead basal autophagy and decreased cellular viability. Likewise, p8 overexpression inhibits Bnip3 
         upregulation after autophagy activation. Thus, p8 appears to antagonize the promotion of autophagy mediated by 
         the FoxO3-Bnip3 axis. Consistent with this, bnip3 knockdown restores viability in p8-deficient cells. 
         In vivo, hearts from p8-/- mice have higher basal autophagy and bnip3 levels. These mice develop left 
         ventricular wall thinning and chamber dilation, with consequent impaired cardiac function.
        '''
    tuples = pickle.load(open(tuplefile, 'rb'))  ##词组数据
    mwe_tokenizer = nltk.tokenize.MWETokenizer(tuples)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')  ##分句
    word_token = nltk.word_tokenize(test_abstract)
    wordlist = mwe_tokenizer.tokenize(word_token)
    result_abstract = " ".join(wordlist)
    print(result_abstract)

    sentences = tokenizer.tokenize(test_abstract)
    result = []
    for sentence in sentences:
        word_token = nltk.word_tokenize(sentence)
        wordlist = mwe_tokenizer.tokenize(word_token)
        sentence = " ".join(wordlist)
        result.append(sentence)

    print(" ".join(result))

# 获取单词词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def preprocessing_abstracts(inputpath, outpath, spacy_nlp, tuplefile):
    lemma = WordNetLemmatizer()

    ## tuple file
    tuples = pickle.load(open(tuplefile, 'rb'))  ##词组数据
    tuples.append(['alzheimer', "'s", 'disease'])
    tuples.append(['Alzheimer', "'s", 'disease'])
    tuples.append(['Alzheimer', 'disease'])
    mwe_tokenizer = nltk.tokenize.MWETokenizer(tuples)  ##分词工具，将词组合并为一个单词，用下划线连接

    years = os.listdir(inputpath)
    for year in years:
        print(year)
        yeardir = os.path.join(inputpath, year)
        files = os.listdir(yeardir)
        ##输出目录
        outdir = os.path.join(outpath, year)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for file in files:
            filepath = os.path.join(yeardir, file)
            ##获取数据
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                data = f.readlines()
            abstract = " ".join(data)
            abstract = str(abstract).replace("\n", "")
            abstract = str(abstract).replace(".,", ".")
            abstract = str(abstract).replace(",.", ".")

            ##将词组合并为一个单词
            word_token = nltk.word_tokenize(abstract)
            wordlist = mwe_tokenizer.tokenize(word_token)
            abstract_tuples = " ".join(wordlist)

            ## lemmatize
            abstract_lemma = spacy_abstract(abstract_tuples, spacy_nlp)

            ## nltk lemmatize
            word_token = nltk.tokenize.word_tokenize(abstract_lemma)  # 分词
            tagged_sent = pos_tag(word_token)  # 获取单词词性
            lemma_sent = []
            for tag in tagged_sent:
                wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
                lemma_sent.append(lemma.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原

            abstract_nltk = ' '.join(lemma_sent)

            ## 将结果写入文件
            outfile = os.path.join(outdir, file)
            with open(outfile, 'w', encoding='utf-8-sig') as fw:
                fw.write(str(abstract_nltk))


if __name__ == '__main__':
    tuplefile = './data/category_dicts/mesh_tuples.pkl'
    inputpath = './data/abstracts'
    outpath = './data/abstracts-lemmatize'

    spacy_path = "C:/workspace-python/models/en_core_web_sm-2.1.0/en_core_web_sm/en_core_web_sm-2.1.0"

    spacy_nlp = load_spacy_nlp(spacy_path)

    preprocessing_abstracts(inputpath, outpath, spacy_nlp, tuplefile)

