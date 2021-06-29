import jieba
from urllib.request import urlopen
import json
import os
import pandas as pd
import gensim
from gensim.models import LdaModel, LdaMulticore
from gensim.corpora import Dictionary


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def openJsonFile(filename):
    '''
    输入文件路径，读取Json文件，返回一个字典
    '''
    with open(filename, 'r') as f:
        data = json.load(f)
    return data
    # resp = json.loads(u.read().decode('utf-8'))
    
def FindAllFile(base):
    '''
    遍历目录下所有文件，返回一个生成器
    '''
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f
            
def stopwordslist():
    '''
    创建停用词列表
    '''
    stopwords = [line.strip() for line in open('./topics/stopwords.txt', 'r', encoding='UTF-8').readlines()]
    return stopwords

def seg_depart(sentence):
    '''
    定义停词函数，对句子进行中文分词
    '''
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(sentence.strip())
    # 创建一个停用词列表
    stopwords = stopwordslist()
    # 去停用词
    output = filter(lambda x: x not in stopwords and x != '\xa0' and not x.isnumeric() and len(x.strip())>1, sentence_depart)
    return output

    
def predict(text, model, dictionary):
    '''
    字典函数，输入文本和模型，输出一个二元组，(id, topic)，id为主题编号，topic为主题分布
    text: 文本
    model: 主题模型
    dictionray: 字典文件
    '''
    text_list = list(seg_depart(text))
    text_corpus = dictionary.doc2bow(text_list)
    vector = model[text_corpus]
    id = vector[0][0]
    return model.print_topics(num_words=20)[id]
    
def visualization(corpus, model, dictionary, save_path):
    '''
    可视化函数，输入语料、模型和字典文件，输出html文件
    corpus: 输入语料
    model: 模型
    dictionary: 字典
    save_path: html保存路径
    '''
    import pyLDAvis.gensim_models as gensimvis
    import pyLDAvis
    vis = gensimvis.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(vis, save_path)
    
    
if __name__=="__main__":
    # 定义目录路径
    model_path = "topics/model.lda"
    dict_path = "topics/doc2bow.dict"
    copurs_path = "topics/data_corpus.mm"
    save_path = "topics/vis.html"
    model = LdaModel.load(model_path)
    from pprint import pprint
    pprint(model.print_topics(num_words=20)) # 输出主题
    
    # 加载字典文件、语料文件、模型文件
    dictionary = gensim.corpora.Dictionary.load(dict_path)
    corpus = gensim.corpora.MmCorpus(copurs_path)
    lda = gensim.models.ldamodel.LdaModel.load(model_path)
    
    # 可视化
    visualization(corpus, model, dictionary, save_path)
    
    # 预测接口
    text = '''王安石的诗十首 　　王安石，是北宋时期著名的思想家、政治家、文学家、改革家。下面是小编分享的王安石的诗十首，希望大家喜欢！ 　　1、《梅花》 　　年代：宋 作者：王安石 　　墙角数枝梅，凌寒独自开。 　　遥知不是雪，为有暗香来。 　　2、《登飞来峰》 　　年代：宋 作者：王安石 　　飞来山上千寻塔，闻说鸡鸣见日升。 　　不畏浮云遮望眼，自缘身在最高层。 　　3、《泊船瓜洲》 　　年代：宋 作者：王安石 　　京口瓜洲一水间，钟山只隔数重山。 　　春风又绿江南岸，明月何时照我还？ 　　4、《元日》 　　年代：宋 作者：王安石 　　爆竹声中一岁除，春风送暖入屠苏。 　　千门万户曈曈日，总把新桃换旧符。 　　5、《明妃曲》 　　年代：宋 作者：王安石 　　明妃初出汉宫时，泪湿春风鬓脚垂。 　　低徊顾影无颜色，尚得君王不自持。 　　归来却怪丹青手，入眼平生几曾有； 　　意态由来画不成，当时枉杀毛延寿。 　　一去心知更不归，可怜着尽汉宫衣； 　　寄声欲问塞南事，只有年年鸿雁飞。 　　家人万里传消息，好在毡城莫相忆； 　　君不见咫尺长门闭阿娇，人生失意无南北。 　　6、《春日》 　　年代：宋 作者：王安石 　　冉冉春行暮，菲菲物竞华。 　　莺犹求旧友，雁不背贫家。 　　室有贤人酒，门无长者车。 　　醉眠聊自适，归梦到天涯。 　　7、《梅花》 　　年代：宋 作者：王安石 　　白玉堂前一树梅，为谁零落为谁开。 　　唯有春风最相惜，一年一度一归来。 　　8、《孟子》 　　年代：宋 作者：王安石 　　沉魄浮魂不可招，遗编一读想风标。 　　何妨举世嫌迂阔，故有斯人慰寂寥。 　　9、《即事》 　　年代：宋 作者：王安石 　　径暖草如积，山晴花更繁。 　　纵横一川水，高下数家村。 　　静憩鸡鸣午，荒寻犬吠昏。 　　归来向人说，疑是武陵源。 　　10、《江上》 　　年代：宋 作者：王安石 　　江北秋阴一半开，晚云含雨却低回。 　　青山缭绕疑无路，忽见千帆隐映来。 【王安石的诗十首】相关文章： 1.王安石的古诗三十首 2.王安石写的诗 3.关于王安石的诗 4.王安石的诗有 5.王安石所写的诗 6.王安石有名的诗 7.王安石的诗精选 8.王安石的诗'''
    predict(text, model, dictionary)