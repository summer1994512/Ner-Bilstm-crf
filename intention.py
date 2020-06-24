from collections import OrderedDict
from pprint import pprint


class get_intention(object):
    def __init__(self):
        p = {
            'intention':None,
            'confidence':None,
            'Action': None,
            'Time':None,
            'Place':[],
            'Person':None,
            'Home_Appliances':[],
            'Weather':None,
            'Song':None,
            'Illness':None,
            'MedTerm':None,
            'Body':None,
            'location':None
        }
        self.intention_result = OrderedDict(p)

    def extract_intention(self,content):
        aa = []
        self.intention_result['intention'] = content[0]
        self.intention_result['confidence'] = content[1]
        for i,ii in enumerate(content):
            if i>1:
                for j,k in ii.items():
                    if k == 'Time':
                        self.intention_result[k] = ii['cont']
                    if k == 'Place':
                        self.intention_result[k].append(ii['cont'])
                    if k == 'Action':
                        self.intention_result[k] = ii['cont']
                    if k == 'Person' or ii['pos'] == 'nh':
                        self.intention_result['Person'] = ii['cont']
                    if k == 'Home_Appliances':
                        self.intention_result[k].append(ii['cont'])
                    if k == 'Weather':
                        self.intention_result[k] = ii['cont']
                    if k == 'Song':
                        self.intention_result[k] = ii['cont']
                    if k == 'Illness':
                        self.intention_result[k] = ii['cont']
                    if k == 'MedTerm':
                        self.intention_result[k] = ii['cont']
                    if k == 'Body':
                        self.intention_result[k] = ii['cont']
                    if ii['pos'] =='ns':
                        self.intention_result['location'] = ii['cont']
        for i,j in enumerate(list(self.intention_result.values())):
            if j==None or j==[] or j=='欧拉蜜':
                aa.append(list(self.intention_result.keys())[i])
        for i in aa:
            del self.intention_result[i]
        return self.intention_result,content[2:]



if __name__=='__main__':
    ltp = ['__label__irconditioner-',
           0.9997875094413757,
           {'cont': '欧拉蜜', 'pos': 'nh', 'ner': 'Person', 'id': 0},
           {'cont': '，', 'pos': 'wp', 'ner': 'O', 'id': 3},
           {'cont': '帮', 'pos': 'v', 'ner': 'O', 'id': 4},
           {'cont': '我', 'pos': 'r', 'ner': 'O', 'id': 5},
           {'cont': '打开', 'pos': 'v', 'ner': 'Action', 'id': 6},
           {'cont': '厨房', 'pos': 'n', 'ner': 'Place', 'id': 8},
           {'cont': '的', 'pos': 'u', 'ner': 'O', 'id': 10},
           {'cont': '空调', 'pos': 'n', 'ner': 'Home_Appliances', 'id': 11}]
    Intention = get_intention()
    p = Intention.extract_intention(ltp)
    pprint((p))