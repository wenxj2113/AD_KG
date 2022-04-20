import os
import spacy
import time
import pymysql

class MysqlHelper:
    def __init__(self, host='39.98.161.93', port=3306, db='medai-rawdata-pubmed', user='jida', passwd='Jida123456.',
                 charset='utf8'):
        self.conn = pymysql.connect(host=host, port=port, db=db, user=user, passwd=passwd, charset=charset)

    def insert(self, sql, params):
        return self.__cud(sql, params)

    def update(self, sql, params):
        return self.__cud(sql, params)

    def delete(self, sql, params):
        return self.__cud(sql, params)

    def __cud(self, sql, params=[]):
        try:
            # 用来获得python执行Mysql命令的方法,也就是我们所说的操作游标
            # cursor 方法将我们引入另外一个主题：游标对象。通过游标扫行SQL 查询并检查结果。
            # 游标连接支持更多的方法，而且可能在程序中更好用。
            cs1 = self.conn.cursor()
            # 真正执行MySQL语句
            rows = cs1.execute(sql, params)
            self.conn.commit()
            # 完成插入并且做出某些更改后确保已经进行了提交，这样才可以将这些修改真正地保存到文件中。
            cs1.close()
            self.conn.close()
            return rows  # 影响到了哪行
        except Exception as e:
            print(e)
            self.conn.rollback()

    def fetchone(self, sql, params=[]):
        # 一次只返回一行查询到的数据
        try:
            cs1 = self.conn.cursor()
            cs1.execute(sql, params)
            row = cs1.fetchone()
            # 把查询的结果集中的下一行保存为序列
            cs1.close()
            self.conn.close()
            return row
        except Exception as e:
            return None

    def fetchall(self, sql, params):
        # 接收全部的返回结果行
        # 返回查询到的全部结果值
        try:
            cs1 = self.conn.cursor()
            cs1.execute(sql, params)
            rows = cs1.fetchall()
            cs1.close()
            self.conn.close()
            return rows
        except Exception as e:
            return None


def get_mesh_level(dbname, tablename):
    """
    从数据库中下载mesh的层级数据
    :param dbname:
    :param tablename:
    :return:
    """

    helper = MysqlHelper(db=dbname)
    sql = "SELECT meshname, treenum1, treenum2 FROM " + str(tablename)
    try:
        cs1 = helper.conn.cursor()
        cs1.execute(sql)
        rows = cs1.fetchall()
        cs1.close()
        helper.conn.close()
        return rows
    except Exception as e:
        return None


def print_run_time(func):
    """
    装饰器：计算时间函数
    :param func:
    :return:
    """

    def wrapper(*args, **kw):
        local_time = time.time()
        f = func(*args, **kw)
        print('Function [%s] run time is %.2f' % (func.__name__, time.time() - local_time))
        return f

    return wrapper

def load_nlp(spacy_path):
    '''
    加载依存分析模型
    :param spacy_path: 依存分析模型路径
    :return:
    '''
    nlp = spacy.load(spacy_path) ##加载模型
    return nlp

def recognize_ap_entity(inputpath, mesh_dict, nlp, outpath):
    """
    将ap结果中的实体进行识别，并将识别结果保存
    :param inputpath:
    :param outpath:
    :return:
    """

    inputfiles = os.listdir(inputpath)

    for file in inputfiles:
        apfile = os.path.join(inputpath, file)

        ## 读取AP结果，判断每个单词或词组是否是实体
        with open(apfile, 'r', encoding='utf-8-sig') as f:
            data = f.readlines()
        outfile = os.path.join(outpath, file+"_entity")
        for line in data:
            line = str(line).replace("\n", "")
            if len(line)>1:
                word_freqs = line.split("\t")
                for word_freq in word_freqs:
                    word = str(word_freq).split(",")[0]
                    level = recognize_entity(word, mesh_dict, nlp)

                    if len(level)>0:
                        ## 是实体，保存
                        with open(outfile, 'a', encoding='utf-8-sig') as fw:
                            fw.write(word + "\t" + "\t".join(level) + "\n")


def recognize_entity(word, mesh_list, nlp):
    """
    识别单词word在mesh中的分类，其中mesh_list包含的是mesh的所有分类，格式为：mesh_entity, treenum2, treenum1
    用字典形式保存
    :param word:
    :param mesh_list:
    :return: 如果是mesh分类下的实体，返回两级分类数据，否则返回为空
    """
    level = []

    ##判断是否是词组，如果是词组则含有下划线
    if str(word).find("_")>0:
        ## 是词组，将下划线用空格替换
        word = str(word).replace("_", " ")

    ## 1.先对word进行直接匹配，即直接在mesh_list中进行查找
    if mesh_list.get(word) != None:
        ##匹配成功
        level = mesh_list.get(word)
        return level

    ## 2. 如果没有找到，对word进行变形，进行词形还原处理
    doc = nlp(word)  ##依存分析结果
    words = [str(token.lemma_).strip() for token in doc]
    word_l = " ".join(words)
    if  mesh_list.get(word_l) != None:
        level = mesh_list.get(word_l)
        return level

    ## 3. 如果没有找到，对word进行大小写转换处理
    if str(word).islower():
        ## 将每个单词的首字母大写
        word_t = " ".join([str(item).capitalize() for item in word.split(" ")])
    elif str(word).isupper():
        ## 如果是大写，不进行转换，认为是缩写词
        word_t = word
    else:
        ## 将单词转换为小写
        word_t = str(word).lower()

    if mesh_list.get(word) != None:
        level = mesh_list.get(word_t)
        return level

    return level

def trans_file_2_dict(inputfile, outfile):
    """
    将输入的mesh list文件的数据转换为字典格式输出，其中字典格式为：entity：[level1, level2]
    将实体转换为其他形式，包括小写形式
    :param inputfile:
    :return:
    """

    mesh_dict = {} ## 结果字典

    with open(inputfile, 'r', encoding='utf-8-sig') as f:
        line = f.readline()
        while line != None and len(line)>1:
            line = str(line).replace("\n", "")
            words = line.split("\t")
            if len(words)>2:
                entity = words[0] ## 实体
                level1 = words[1] ## 第一分类（大类）
                level2 = words[2] ## 第二分类（次级分类）
                level = []
                level.append(level1)
                level.append(level2)

                ## 将实体及对应的级别添加到字典中
                if mesh_dict.get(entity) == None:
                    mesh_dict[entity] = level
                    # with open(outfile, 'a', encoding='utf-8-sig') as fw:
                    #     fw.write(entity + "\t" + str(level) + "\n")
                ## 对实体进行变形处理，变成小写形式
                entity_l = str(entity).lower()
                if entity != entity_l:
                    ## 将实体及对应的级别添加到字典中
                    if mesh_dict.get(entity_l) == None:
                        mesh_dict[entity_l] = level
                        # with open(outfile, 'a', encoding='utf-8-sig') as fw:
                        #     fw.write(entity_l + "\t" + str(level) + "\n")
           # print(line)
            line = f.readline()
    print(len(mesh_dict))
    return mesh_dict

def get_mesh_level_file(dbname, tablename, mesh_level_file):
    """
    从数据库中下载mesh level数据，并将结果保存在文件中
    :param dbname:
    :param tablename:
    :param mesh_level_file:
    :return:
    """
    mesh_levels = get_mesh_level(dbname, tablename)
    if mesh_levels != None:
        ##查找到数据，保存
        for row in mesh_levels:
            with open(mesh_level_file, 'a', encoding='utf-8-sig') as fw:
                fw.write(str(row[0]) + "\t" + str(row[1]) + "\t" + str(row[2]) + "\n")

def sorted_entity_dict(inputpath, outpath):
    """
    将实体的结果按照类别进行排序
    :param inputpath:
    :param outpath:
    :return:
    """

    inputfiles = os.listdir(inputpath)

    for file in inputfiles:
        inputfile = os.path.join(inputpath, file)
        outfile = os.path.join(outpath, file)
        with open(inputfile, 'r', encoding='utf-8-sig') as f:
            data = f.readlines()

        entity_level = []
        entity_set = []
        for line in data:
            line = str(line).replace("\n", "")
            if len(line)>0:
                words = line.split("\t")
                entity = words[0]
                level1 = words[1]
                level2 = words[2]
                if entity not in entity_set:
                    entity_set.append(entity)
                    entity_level.append([entity, level2])

        ##排序
        entity_level_sorted = sorted(entity_level, key=lambda x:x[1])
        with open(outfile, 'w', encoding='utf-8-sig') as fw:
            for item in entity_level_sorted:
                fw.write(str(item[0]) + "\t" + str(item[1]) + "\n")

def count_level_entity(inputpath, outfile):
    """
    统计每个类别下的实体数量，并将结果保存在文件中
    :param inputpath:
    :param outfile:
    :return:
    """

    inputfiles = os.listdir(inputpath)
    for file in inputfiles:
        inputfile = os.path.join(inputpath, file)
        with open(inputfile, 'r', encoding='utf-8-sig') as f:
            data = f.readlines()

        entity_level_dict = {}
        for line in data:
            line = str(line).replace("\n", "")
            words = line.split("\t")
            if len(words)>0:
                entity = words[0]
                level = str(words[1])[0]

                entity_set = entity_level_dict.get(level)
                if entity_set == None:
                    entity_set = []

                if entity not in entity_set:
                    entity_set.append(entity)

                entity_level_dict[level] = entity_set

        with open(outfile, 'a', encoding='utf-8-sig') as fw:
            fw.write(str(file) + "\n")
            for item in entity_level_dict:
                fw.write(str(item) + "\t" + str(len(entity_level_dict.get(item))) + "\n")
                entity_set = entity_level_dict.get(item)
                fw.write(" | ".join(entity_set) + "\n\n")


if __name__ == '__main__':
    # dbname = "medai-tred"
    # tablename = "mesh_treenumber_lev2"
    mesh_level_file = './data/mesh_level_data.txt'

    #get_mesh_level_file(dbname, tablename, mesh_level_file)
    mesh_level_dict = './data/mesh_level_dict.txt'
    mesh_dict = trans_file_2_dict(mesh_level_file, mesh_level_dict)
    print(len(mesh_dict))

    appath = './data/ap-results-tfidf'
    ap_entity_path = './data/ap-results-tfidf-entity'

    spacy_path = 'C:\\workspace-python\\models\\en_core_web_sm-2.1.0\\en_core_web_sm\\en_core_web_sm-2.1.0'
    # spacy_path = 'D:/en_core_web_lg/en_core_web_lg/en_core_web_lg-2.0.0'
    nlp = load_nlp(spacy_path)
    print("load nlp success")
    recognize_ap_entity(appath, mesh_dict, nlp, ap_entity_path)

    sorted_path = './data/ap-results-tfidf-entity-sorted'
    sorted_entity_dict(ap_entity_path, sorted_path)

    entity_level_count = './data/tfidf_entity_level_count.txt'
    count_level_entity(sorted_path, entity_level_count)
