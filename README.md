Instructions to Execute code:
1) Training the Model
If you want to train the model from scratch, you need to provide the path to the training data directory and the training labels JSON file as arguments to the shell script. For example:
sh hw2_seq2seq.sh /path/to/training_data training_label.json
Replace /path/to/training_data with the actual path to your training data directory. The script will train the model using the specified data and labels, and it will generate the necessary files such as the model file and object files (*.obj) for later use.
2) Testing the Model
To test the model, you should specify the path to the testing data directory, which should include a feat subdirectory containing the video features. Additionally, you need to provide the name of the output file where the test results will be stored. For instance:
sh hw2_seq2seq.sh /path/to/testing_data output.txt
Replace /path/to/testing_data with the actual path to your testing data directory which contains a feat directory containing video features. The script will use the pre-trained model to generate captions for the test data and save the results in output.txt.

3) Downloading Necessary Files
Before testing the model, ensure that you have downloaded all the necessary files, including the model file (*.h5), object files (*.obj), and the testing_label.json file. If these files are not available, you can train the model again as described in step 1, which will generate these files.
Note:
The model file is saved as Shiva_HW2_model0.h5 by default. If you have a different model file, you may need to modify the script to load the correct file.

If you encounter an error with the model downloaded using wget from hw2_seq2seq.sh, it could be due to extraction issues. In such cases, please use the direct download link from Google Drive(https://docs.google.com/uc?export=download&id=1mARNBxX3SCoCj1YkdeQa84KV_iR41A45) provided, Additionally, ensure that you run the model on a device with CUDA support, as it was trained using CUDA technology.
