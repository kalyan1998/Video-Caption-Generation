wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mARNBxX3SCoCj1YkdeQa84KV_iR41A45' -O Shiva_HW2_model0.h5
python3 train_and_test.py $1 $2
# 1) If you want to train the model you can run the same script providing input as shown below
# sh hw2_seq2seq.sh /MLDS_hw2_1_data/training_data training_label.json

# 2) If you want to test the model you can run the same script providing input as shown below
# sh hw2_seq2seq.sh /MLDS_hw2_1_data/testing_data output.txt

# 3) Please Download all the files including model file, obj files and testing_label.json to test the model 
# If not you can train whole model again which generates the above files as shown in statement number (1) and then test the model as shown in statement number (2).
