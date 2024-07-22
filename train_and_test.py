import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import pickle
import random
import os
from torch.utils.data import Dataset, DataLoader
import seq2seq_model 
from torch.autograd import Variable
from bleu_eval import BLEU
import sys
import time
from vocab_build import create_data_objects
import re

class Config:
    lambda_r = 0.001
    batch_size = 128
    num_epochs = 200
    beam_size = 5
    max_decoder_steps = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model(model, path):
    torch.save(model, path)
    
def preprocess_data(files_directory, label_json_path, word_dict, word_to_index):
    annotated_captions = []
    with open(label_json_path, 'r') as f:
        labels = json.load(f)
    for data in labels:
        for caption in data['caption']:
            caption = re.sub(r'[.!,;?]', ' ', caption).split()
            tokenized_caption = [word_to_index.get(word, 3) for word in caption] 
            tokenized_caption.insert(0, 1) 
            tokenized_caption.append(2)
            annotated_captions.append((data['id'], tokenized_caption))

    avi_features = {}
    files = os.listdir(files_directory)
    for file in files:
        feature = np.load(os.path.join(files_directory, file))
        avi_features[file.split('.npy')[0]] = feature

    return annotated_captions, avi_features

class training_data(Dataset):
    def __init__(self, label_file, files_dir, word_dict, w2i):
        self.label_file = label_file
        self.files_dir = files_dir
        self.word_dict = word_dict
        self.w2i = w2i
        self.data_pair, self.avi = preprocess_data(files_dir, label_file, word_dict, w2i)
        
    def __len__(self):
        return len(self.data_pair)
    
    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000)/10000.
        return torch.Tensor(data), torch.Tensor(sentence)

class test_data(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])
    def __len__(self):
        return len(self.avi)
    def __getitem__(self, idx):
        return self.avi[idx]

def captions_to_indices(captions, word_to_index, max_length):
    indices = []
    for caption in captions:
        words = caption.lower().split()
        caption_indices = [word_to_index.get(word, word_to_index['<unk>']) for word in words]
        caption_indices = caption_indices[:max_length - 1]
        caption_indices += [word_to_index['<eos>']]
        caption_indices += [word_to_index['<pad>']] * (max_length - len(caption_indices))
        indices.append(caption_indices)
    return torch.tensor(indices, dtype=torch.long)

def minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths

def train_and_calculate_loss(model, epoch, train_loader, loss_fn):
    model.train()
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    total_loss = 0

    for batch_idx, (avi_feats, ground_truths, lengths) in enumerate(train_loader):
        avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()
        optimizer.zero_grad()
        seq_logProb, _ = model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)

        ground_truths = ground_truths[:, 1:]
        loss = 0

        for i in range(len(seq_logProb)):
            seq_len = lengths[i] - 1  # Exclude the start token
            loss += loss_fn(seq_logProb[i, :seq_len], ground_truths[i, :seq_len])

        loss /= len(seq_logProb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch: {epoch}: Average Loss: {np.round(avg_loss, 3)}")
    return avg_loss

def test_and_evaluate(test_dataloader, model, index_to_word, test_label_json, output_file, config, use_beam_search=False):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch_idx, (video_ids, avi_features) in enumerate(test_dataloader):
            avi_features = avi_features.cuda()
            avi_features = Variable(avi_features).float()
            if use_beam_search:
                _, seq_predictions = model(avi_features, mode='beam_search', beam_width=config.beam_size)
            else:
                _, seq_predictions = model(avi_features, mode='inference')

            for idx, sequence in enumerate(seq_predictions):
                words = [index_to_word[token.item()] for token in sequence if index_to_word[token.item()] not in ['<pad>', '<bos>', '<eos>', '<unk>']]
                caption = ' '.join(words).split('<eos>')[0].strip()
                predictions.append((video_ids[idx], caption))

    with open(output_file, 'w') as f:
        for video_id, caption in predictions:
            f.write(f'{video_id},{caption}\n')

    test_data = json.load(open(test_label_json, 'r'))
    results = {}
    with open(output_file, 'r') as f:
        for line in f:
            line = line.rstrip()
            video_id, caption = line.split(',', 1)
            results[video_id] = caption

    bleu_scores = [BLEU(results[item['id']], [x.rstrip('.') for x in item['caption']], True) for item in test_data]
    average_bleu_score = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU score is {average_bleu_score}")


def main():
    config = Config()

    model_path = 'Shiva_HW2_model0.h5'
    if os.path.exists(model_path):
        print("Model file exists. Loading the model...")
        print("If you get an error, try providing training data folder input as '/MLDS_hw2_1_data/testing_data' do not provide / after testing_data")
        testing_feat_folder = sys.argv[1]+'/feat/'
        outputDir = sys.argv[2]
        testing_label_json = 'testing_label.json'
        test_dataset = test_data(testing_feat_folder)
        test_dataloader = DataLoader(dataset = test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
        if torch.cuda.is_available():
            model = torch.load(model_path)
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))
        print("Model loaded successfully.")
        with open('train_keytowordtranslation.obj', 'rb') as f:
            train_index_to_word = pickle.load(f)
        test_dataset = test_data(testing_feat_folder)
        test_dataloader = DataLoader(dataset = test_dataset, batch_size=1, shuffle=True, num_workers=8)
        test_and_evaluate(test_dataloader, model.cuda(),train_index_to_word, testing_label_json, outputDir, config)     
    else:
        training_feat_folder = sys.argv[1]+'/feat/'
        training_label_json = sys.argv[2]
        print("Model file is not downloaded, Hence creating model")
        print("Provide training data containing folder and training_label.json file to create a model")
        train_vocab_dict = create_data_objects(training_feat_folder, training_label_json,"train")
        with open('train_wordtokeytranslation.obj', 'rb') as f:
            train_word_to_index = pickle.load(f)
        with open('train_keytowordtranslation.obj', 'rb') as f:
            train_index_to_word = pickle.load(f)
        train_dataset = training_data(training_label_json, training_feat_folder, train_vocab_dict, train_word_to_index)
        train_dataloader = DataLoader(dataset = train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, collate_fn=minibatch)
        loss_arr = list()
        x = len(train_index_to_word)+4
        encoder = seq2seq_model.Encoder()
        decoder = seq2seq_model.Decoder(512, x, x, 1024, 0.3)
        model = seq2seq_model.Seq2Seq(encoder = encoder,decoder = decoder).to(config.device)
        optimizer = optim.Adam(model.parameters(), lr=config.lambda_r)
        criterion = nn.CrossEntropyLoss(ignore_index=train_word_to_index['<pad>'])
        start = time.time()
        for epoch in range(config.num_epochs):
            loss_arr.append(train_and_calculate_loss(model,epoch+1, train_dataloader, criterion))
        end = time.time()
        save_model(model,model_path)
        with open('loss_arr.txt', 'w') as f:
            for loss in loss_arr:
                f.write(f"{loss}\n")
        print("Training finished {}  elapsed time: {: .3f} seconds. \n".format('test', end-start))
    
if __name__ == '__main__':
        main()
