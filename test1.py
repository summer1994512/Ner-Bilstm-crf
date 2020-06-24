# encoding=utf8

import os,time
import tensorflow as tf
from model import Model
from utils import load_config, load_model, save_model
from data_utils import input_from_line
from vocab import Vocab

flags = tf.app.flags


flags.DEFINE_string("label_file", os.path.join("data", "labell.txt"), "File for lobels")
# flags.DEFINE_string("ckpt_path", os.path.join("model", "ckpt"), "Path to save model")
flags.DEFINE_string("config_file", os.path.join("data", "config_file"), "File for config")
flags.DEFINE_string("emb_file", os.path.join("data", "zhwiki-20190501-t2s_100_w5_mc5_iter10"),
                    "Path for pre_trained embedding")

FLAGS = tf.app.flags.FLAGS


class evaluate_line(object):
    def __init__(self,model_file):
# def evaluate_line(sentence_predict):
        self.config = load_config(FLAGS.config_file)
        self.model_file = model_file
        # limit GPU memory
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.label_vocab = Vocab.load(FLAGS.label_file)
        self.word_vocab, self.embedding = Vocab.load_word2vec(FLAGS.emb_file)
        self.id_to_tag = {}
        for i, v in self.label_vocab.w2i.items():
            self.id_to_tag[v] = i
        self.sess = tf.Session(config=self.tf_config)
        self.sess.as_default()

        self.model = load_model(self.sess, Model, self.model_file, self.config)
    def pridict(self,line):
        result = self.model.predict(self.sess, input_from_line(line, self.word_vocab.w2i), self.id_to_tag)
        return result

if __name__=='__main__':
    pp = os.path.join("model", "ckpt")
    a = evaluate_line(pp)
    while True:
        line = input('请输入句子：')
        t1 = time.time()
        w = a.pridict(line)
        print(time.time()-t1)
        print(w)
