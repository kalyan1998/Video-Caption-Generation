# Seq2Seq Video Captioning with Attention Mechanism

## Project Description

This project implements a Seq2Seq (Sequence-to-Sequence) model with an attention mechanism for video captioning. The goal is to generate descriptive captions for video clips using deep learning techniques. The model is trained on video features and their corresponding captions, leveraging the attention mechanism to improve caption quality.

## Workflow Overview

1. **Data Collection and Preparation**
   - **Dataset**: Video features and captions from a predefined dataset.
   - **Preprocessing Steps**:
     - **Vocabulary Building**: Constructs a dictionary of words from captions, mapping words to indices and vice versa.
     - **Data Cleaning**: Removes punctuation and non-alphanumeric characters.
     - **Data Loading**: Associates video features with corresponding captions and saves mappings using pickle.

2. **Model Architecture**
   - **Attention Mechanism**:
     - Computes context vectors using the hidden state of the decoder and the encoder outputs.
   - **Encoder**:
     - Processes input video features through linear layers, dropout, and GRU to produce output vectors and a hidden state.
   - **Decoder**:
     - Generates output sequences (captions) using an embedding layer, attention mechanism, GRU, and a linear layer.
   - **Seq2Seq Model**:
     - Combines encoder and decoder for training, inference, and beam search.

3. **Model Training**
   - **Training Setup**:
     - Learning rate: 0.001
     - Batch size: 128
     - Number of epochs: 200
     - Beam size: 5
     - Maximum decoder steps: 15
   - **Training Process**:
     - Converts captions to numerical indices.
     - Trains the model, calculates loss, and performs backpropagation.

4. **Evaluation**
   - **Metrics Used**:
     - BLEU Score: Measures the quality of generated captions.
   - **Evaluation Process**:
     - Tests the model on a test dataset.
     - Compares generated captions with ground truth captions.

## File Descriptions

### `vocab_build.py`
- **build_vocab_dict**: Constructs a vocabulary dictionary from captions.
- **filter_token**: Cleans a string by removing specified punctuation.
- **create_data_objects**: Loads video captions and associates them with video IDs.

### `train_and_test.py`
- **Config Class**: Defines configuration parameters for the model.
- **captions_to_indices**: Converts captions to numerical indices.
- **train_and_calculate_loss**: Trains the Seq2Seq model and calculates loss.
- **test_and_evaluate**: Evaluates the model on test data.
- **main**: Entry point for training and testing the model.

### `seq2seq_model.py`
- **attention Class**: Implements the attention mechanism.
- **Encoder Class**: Processes input video features.
- **Decoder Class**: Generates output sequences (captions).
- **Seq2Seq Class**: Combines encoder and decoder for complete model.

## Instructions to Execute Code

### 1) Training the Model

To train the model from scratch, provide the path to the training data directory and the training labels JSON file as arguments to the shell script. For example:
```bash
sh hw2_seq2seq.sh /path/to/training_data training_label.json
```
Replace /path/to/training_data with the actual path to your training data directory. The script will train the model using the specified data and labels, generating necessary files such as the model file and object files (*.obj) for later use.

### 2) Testing the Model

To test the model, specify the path to the testing data directory, which should include a feat subdirectory containing the video features. Additionally, provide the name of the output file where the test results will be stored. For instance:
```bash
sh hw2_seq2seq.sh /path/to/testing_data output.txt
```
Replace /path/to/testing_data with the actual path to your testing data directory containing a feat directory with video features. The script will use the pre-trained model to generate captions for the test data and save the results in output.txt.

### 3) Downloading Necessary Files
Before testing the model, ensure you have downloaded all necessary files, including the model file (.h5), object files (.obj), and the testing_label.json file. If these files are not available, you can train the model again as described in step 1, which will generate these files.

Note:

The model file is saved as Shiva_HW2_model0.h5 by default. If you have a different model file, you may need to modify the script to load the correct file.
If you encounter an error with the model downloaded using wget from hw2_seq2seq.sh, it could be due to extraction issues. In such cases, use the direct download link from Google Drive provided here. Additionally, ensure that you run the model on a device with CUDA support, as it was trained using CUDA technology.
