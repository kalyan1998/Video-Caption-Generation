import sys
import os
from os import listdir
import numpy as np
import json
import pickle as pk

def build_vocab_dict(sentences, min_word_count):
    word_counts = {}
    total_sequences = 0
    for sentence in sentences:
        total_sequences += 1
        for word in sentence.lower().split(' '):
            word_counts[word] = word_counts.get(word, 0) + 1

    filtered_vocab = [word for word, count in word_counts.items() if count >= min_word_count]
    print(f'From {len(word_counts)} words, filtered {len(filtered_vocab)} words to dictionary with minimum count [{min_word_count}]\n')

    id_to_word = {0: '<pad>', 1: '<bos>', 2: '<eos>', 3: '<unk>'}
    word_to_id = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}

    for index, word in enumerate(filtered_vocab, start=4):
        word_to_id[word] = index
        id_to_word[index] = word

    for special_word in ['<pad>', '<bos>', '<eos>', '<unk>']:
        word_counts[special_word] = total_sequences

    return word_to_id, id_to_word, filtered_vocab

def filter_token(string):
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translation_table = str.maketrans('', '', filters)
    return string.translate(translation_table)

def create_data_objects(training_feat_folder, training_label_json, mode,min_word_count=4):
    training_feat_filenames = listdir(training_feat_folder)
    training_feat_filepaths = [(training_feat_folder + filename) for filename in training_feat_filenames]

    video_id = [filename[:-4] for filename in training_feat_filenames]

    ID_feat_map = {os.path.basename(filepath)[:-4]: np.load(filepath) for filepath in training_feat_filepaths}

    video_captions = json.load(open(training_label_json, 'r'))

    captions_by_video_id = {video["id"]: [filter_token(sentence) for sentence in video["caption"]] for video in video_captions}

    captions_corpus = [caption for video_captions in captions_by_video_id.values() for caption in video_captions]

    wordtokeytranslation, keytowordtranslation, vocab_dict = build_vocab_dict(captions_corpus, min_word_count=min_word_count)
    
    wkfilename = './'+mode+'_wordtokeytranslation.obj'
    kwfilename = './'+mode+'_keytowordtranslation.obj'
    pk.dump(wordtokeytranslation, open(wkfilename, 'wb'))
    pk.dump(keytowordtranslation, open(kwfilename, 'wb'))

    video_features_and_captions = [(ID_feat_map[video], caption) for video in video_id for caption in captions_by_video_id[video]]
    captions_tokenized = [caption.split() for captions in captions_by_video_id.values() for caption in captions]
    all_caption_words = [word for words in captions_tokenized for word in words]

    unique_caption_words, word_counts = np.unique(all_caption_words, return_counts=True)
    max_caption_length = max(len(words) for words in captions_tokenized)
    avg_caption_length = np.mean([len(words) for words in captions_tokenized])
    num_unique_caption_words = len(unique_caption_words)

    print(f"Caption dimension: {np.shape(video_features_and_captions[0][0])}")
    print(f"Caption's max length: {max_caption_length}")
    print(f"Average length of captions: {avg_caption_length}")
    print(f"Unique words in captions: {num_unique_caption_words}")

    return vocab_dict
