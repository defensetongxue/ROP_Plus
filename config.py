import argparse
import yaml

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

    # train and test
    parser.add_argument('--config_file', type=str, default='./YAML/default.yaml', help='load config file')

    with open('./', 'r') as file:
        config = yaml.safe_load(file)
    args = parser.parse_args()
    return args,config