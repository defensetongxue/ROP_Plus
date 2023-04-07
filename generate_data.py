from utils import generate_data_processer
from config import paser_args
import os

args = paser_args()
data_processer = generate_data_processer(PATH=args.PATH,data_file=args.data_file,
                                         TEST_DATA=args.TEST_DATA)
data_processer.generate_test_data()
data_processer.get_data_condition()
