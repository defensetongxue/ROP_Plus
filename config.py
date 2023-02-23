import argparse
def paser_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--PATH', type=str, default="../autodl-tmp/", help='Where the data is')
    parser.add_argument('--TEST_DATA', type=int, default=1e10, help='How many data will be used')
    parser.add_argument('--train_proportion', type=float, default=0.6, help='What proportion of the training data is')
    parser.add_argument('--val_proportion', type=float, default=0.2, help='What proportion of the validation data is')
    
    parser.add_argument('--batch_size', type=int, default=128 ,help='batch size')
    parser.add_argument('--epoch', type=int, default=15 ,help='epoch when triaing')
    parser.add_argument('--lr', type=float, default=1e-3 ,help='learning rate')
    parser.add_argument('--GEN_DATA', type=bool, default=False, help='if generate data again')
    parser.add_argument('--pretrain', type=bool, default=True, help='if load the pretrain model')
    
    args = parser.parse_args()
    return args