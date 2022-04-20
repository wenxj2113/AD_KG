import xml.etree.ElementTree as ET
import os


def traverseXml(element):
    print(len(element))
    if len(element)>0:
        for child in element:
            print(child.tag, "----", child.text)
            traverseXml(child)
def process_keyword(keywordlist):
    keywords = []
    if isinstance(keywordlist, ET.Element):
        for keyword in keywordlist:
            keywords.append(keyword.text)
    return keywords
def process_title(article):
    article_title = article.find("ArticleTitle")
    title = ""
    if isinstance(article_title, ET.Element):
        title = article_title.text
    title = str(title)
    return title
def process_abstract(abstract):
    abstracts = []
    if isinstance(abstract, ET.Element):
        for child in abstract:
            if isinstance(child, ET.Element):
                abstracts.append(str(child.text))
    if len(abstracts) > 0:
        return " ".join(abstracts)
    else:
        return ""

def process_pubmedarticle(element):
    for child in element:
        child_tag = child.tag
        if child_tag == 'MedlineCitation':
            pmid = child.find("PMID").text
            #print("pmid: ", pmid)
            article = child.find("Article")
            article_title = process_title(article)
            #print("article title: ", article_title)
            abstract = article.find("Abstract")
            abstract_text = process_abstract(abstract)
            #("abstract text: ", abstract_text)
            keywordlist = child.find("KeywordList")
            keywords = process_keyword(keywordlist)
          #  print("keywords: ", keywords)
        elif child_tag == "BookDocument":
            pmid = child.find("PMID").text
            #print("pmid: ", pmid)
            article = child.find("Book")
            article_title = process_title(article)
            #print("article title: ", article_title)
            abstract = child.find("Abstract")
            abstract_text = process_abstract(abstract)
            #print("abstract text: ", abstract_text)
            keywordlist = child.find("KeywordList")
            keywords = process_keyword(keywordlist)
           # print("keywords: ", keywords)

    return pmid, article_title, abstract_text, keywords



if __name__ == "__main__":
    inputpath = "./data/pubmed-xml/"
    outpath = "./data/abstracts/"
    years = os.listdir(inputpath)
    for year in years:
        print(year)
        inputfiles = os.listdir(inputpath + year)
        if not os.path.exists(outpath+year):
            os.mkdir(outpath+year) ##创建输出文件夹
        pmids = []
        article_titles = []
        abstract_texts = []
        keyword_lists = []
        #print(inputfiles)
        for file in inputfiles:
            filepath = os.path.join(inputpath+year, file)
            print(filepath)
            try:

                tree = ET.parse(filepath)
                print("tree type: ", type(tree))
                root = tree.getroot()

                for child in root:
                    child_tag = child.tag
                    pmid, article_title, abstract_text, keywords = process_pubmedarticle(child)
                    pmids.append(pmid)
                    article_titles.append(article_title)
                    abstract_texts.append(abstract_texts)
                    keyword_lists.append(keywords)
                    ##将结果保存
                    outfile = os.path.join(outpath + year, pmid)
                    # print("outfile: ", outfile)
                    if len(article_title + " " + abstract_text)>50:
                        with open(outfile, 'w+', encoding='utf-8') as f:
                            f.write(article_title + " " + abstract_text)
            except Exception as e:
                print("parse file fail!")

