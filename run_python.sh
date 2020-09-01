python3 main.py --train_data_path ./THUCNews/train.txt \
				--valid_data_path ./THUCNews/valid.txt \
				--test_data_path  ./THUCNews/test.txt \
				--word2vec_data_path ./word2vec_data_path \
				--mode train \
				--model rand \
				--save_model \
				--early_stopping \
				--epoch 200 \
				--learning_rate 0.1


