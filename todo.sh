python -u generate_data.py --data_file original
python -u train.py --data_file original
python -u generate_data.py --data_file replace_channel
python -u train.py --data_file replace_channel
python -u generate_data.py --data_file vessel
python -u train.py --data_file vessel