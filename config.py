import argparse
'''
datafile_list
orignal
replace_blue
'''
def get_config():
    
    parser = argparse.ArgumentParser()
    # cleansing
    parser.add_argument('--path_src', type=str, default="../autodl-tmp/data_original", help='Where the data is')
    parser.add_argument('--path_tar', type=str, default="../autodl-tmp/dataset_ROP", help='Where the data generate')
    parser.add_argument('--train_split', type=float, default=0.7, help='training data proportion')
    parser.add_argument('--val_split', type=float, default=0.1, help='valid data proportion')

    parser.add_argument('--cleansing', type=bool, default=True, help='if parse orginal data')
    parser.add_argument('--vessel', type=bool, default=True, help='if generate vessel segmentation result')
    parser.add_argument('--optic_disc', type=bool, default=True, help='if doing optic disc detection')

    parser.add_argument('--save_name', type=str, default="best.pth", help='where the model will be save')

    parser.add_argument('--data_file', type=str, default='original', help='which file of data')

    parser.add_argument('--train_proportion', type=float, default=0.6, help='What proportion of the training data is')
    parser.add_argument('--val_proportion', type=float, default=0.2, help='What proportion of the validation data is')

    parser.add_argument('--batch_size', type=int, default=128 ,help='batch size')
    parser.add_argument('--epoch', type=int, default=30 ,help='epoch when triaing')
    parser.add_argument('--lr', type=float, default=1e-3 ,help='learning rate')
    parser.add_argument('--early_stop', type=int, default=10, help='patient')

    args = parser.parse_args()
    return args