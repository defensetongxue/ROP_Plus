import argparse
def paser_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--PATH', type=str, default="../autodl-tmp/", help='Where the data is')
    parser.add_argument('--save_name', type=str, default="best.pth", help='where the model will be save')
    parser.add_argument('--gen_vessel', type=bool, default=False, help='if generate vessel')

    parser.add_argument('--GEN_DATA', type=bool, default=False, help='if generate data again')
    parser.add_argument('--TEST_DATA', type=int, default=1e10, help='How many data will be used')
    parser.add_argument('--train_proportion', type=float, default=0.6, help='What proportion of the training data is')
    parser.add_argument('--val_proportion', type=float, default=0.2, help='What proportion of the validation data is')

    parser.add_argument('--batch_size', type=int, default=128 ,help='batch size')
    parser.add_argument('--epoch', type=int, default=30 ,help='epoch when triaing')
    parser.add_argument('--lr', type=float, default=1e-3 ,help='learning rate')
    parser.add_argument('--pretrain', type=bool, default=True, help='if load the pretrain model')
    parser.add_argument('--early_stop', type=int, default=5, help='patient')

    # vessel_seg pars
    parser.add_argument('--ves_patch_height', default=96)
    parser.add_argument('--ves_patch_width', default=96)
    parser.add_argument('--ves_stride_height', default=16)
    parser.add_argument('--ves_stride_width', default=16)

    args = parser.parse_args()
    return args