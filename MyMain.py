from MyOption import args
from MyTrain import Mytrain as train
from MyTest import Mytest as test
import datetime
import pytz


def main():
    print('Device:\t', args.DEVICE)
    print('Task:\t', args.temp_dir.split('/')[-1])
    print('Batch Size:\t', args.batch_size)
    print('Start Time:\t', datetime.datetime.now(pytz.timezone('PRC')).strftime("%Y-%m-%d %H:%M:%S"))

    print('###########################################################################')
    print('################################## Train ##################################')
    print('###########################################################################')

    train()

    print('###########################################################################')
    print('################################## Test ###################################')
    print('###########################################################################')

    test()


if __name__ == '__main__':
    main()
