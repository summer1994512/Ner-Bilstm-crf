# -*- coding: utf-8 -*-
import os,time
import fasttext.FastText as fasttext
from test1 import evaluate_line
from pprint import pprint
from intention import get_intention
from collections import OrderedDict
import tensorflow as tf

dir_path = os.getcwd()

cla_model = os.path.join('model','data_dim100_lr05_iter5.model')

cws_model_path = os.path.join(dir_path,'model','cws.model')  # 分词模型路径，模型名称为`cws.model`

pos_model_path = os.path.join(dir_path,'model','pos.model')  # 词性标注模型路径，模型名称为`pos.model`

ner_model_path = os.path.join(dir_path,'model','ckpt')  # 实体识别模型路径，模型名称为`ner.model`
Ner = evaluate_line(ner_model_path)                    #加载实体识别模型
classifier = fasttext.load_model(cla_model)            #加载文本分类模型

from pyltp import Segmentor, Postagger


class PyltpTool:
    def __init__(self,sentence):
        self.sentence = sentence

        self.segmentor = Segmentor()  # 初始化实例
        self.segmentor.load(cws_model_path)  # 加载模型
        self.postagger = Postagger()
        self.postagger.load(pos_model_path)
        self.seg_sentences = []
        self.pos_tags = []
        self.ner_sentence = []
        self.id = 0
        self.a = []

    def get_seg_sentences(self):
        seg_sentence = self.segmentor.segment(self.sentence)
        self.seg_sentences.extend(list(seg_sentence))
        self.a.extend([i[0] for i in classifier.predict(' '.join(seg_sentence))])

    def get_pos_tags(self):
        # for seg_sentence in self.seg_sentences:
        pos_tag = self.postagger.postag(list(self.seg_sentences))
        self.pos_tags.extend(list(pos_tag))
        # print(self.pos_tags)



    def ner(self):
        ner_content = Ner.pridict(self.sentence)
        for i in range(len(self.seg_sentences)):
            s=OrderedDict()
            s['cont'] = self.seg_sentences[i]
            s['pos'] = self.pos_tags[i]
            if i!=0:
                self.id+=len(self.seg_sentences[i-1])
            if ner_content[0][self.id]=='O':
                s['ner'] = ner_content[0][self.id]
            else:
                s['ner'] = ner_content[0][self.id].split('-')[1]
            s['id'] = self.id
            self.a.append(s)
        return self.a


    def realease(self):
        self.segmentor.release()
        self.postagger.release()

if __name__=='__main__':
    # 输入存放一行一行句子的文件，分别输出分词、词性标注和依存分析三个文件。
    while True:
        text = input('请输入：')
        # text = '刘伟，帮我打开厨房空调。'
        t1 = time.time()
        ctb6_pyltp_tool = PyltpTool(text)
        ctb6_pyltp_tool.get_seg_sentences()
        ctb6_pyltp_tool.get_pos_tags()
        ctb6_pyltp_tool.realease()
        ltp_result = ctb6_pyltp_tool.ner()

        # pprint(ltp_result)
        Intention = get_intention()
        intention_result, summary = Intention.extract_intention(ltp_result)
        print('意图识别的结果：\n')
        pprint(intention_result)
        print('具体的分词信息\n')
        pprint(summary)
        print('{}秒'.format(time.time()-t1))


        # if not text:
        #     break
