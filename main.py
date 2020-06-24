from data_split import split_train_test,TransformData
from train import train_model,cal_precision_and_recall
import jieba


def main(source):
    basename = source.rsplit('.', 1)[0]
    csv_file = basename + '.csv'

    td = TransformData()
    handler = open(source,encoding='utf-8')
    td.to_csv(handler, csv_file)
    handler.close()

    train_file, test_file = split_train_test(csv_file)

    dim = 100
    lr = 5
    epoch = 5
    model = f'model/data_dim{str(dim)}_lr0{str(lr)}_iter{str(epoch)}.model'

    classifier = train_model(ipt=train_file,
                             opt=model,
                             model=model,
                             dim=dim, epoch=epoch, lr=0.5
                             )

    result = classifier.test(test_file)
    print(result)

    cal_precision_and_recall(classifier)
    while True:
        text = input('请输入句子:')
        seg_text = ' '.join(jieba.lcut(text))
        if text == '-1':
            break
        print(classifier.predict(seg_text))


if __name__ == '__main__':
    main('data.txt')
