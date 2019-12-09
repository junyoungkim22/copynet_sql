import os
import random
from torch.utils.data import Dataset
from utils import tokens_to_seq, contains_digit
from operator import itemgetter


class Language(object):
    def __init__(self, vocab_limit, data_path, parser):
        self.data_path = data_path
        self.parser = parser

        '''
        if files:
            self.files = files
        else:
            self.files = os.listdir(self.data_path)
        '''

        vocab = self.create_vocab()

        #truncated_vocab = sorted(vocab.items(), key=itemgetter(1), reverse=True)[:vocab_limit]
        truncated_vocab = sorted(vocab.items(), key=itemgetter(1), reverse=True)[:]

        del vocab

        self.tok_to_idx = dict()
        self.tok_to_idx['<MSK>'] = 0
        self.tok_to_idx['<SOS>'] = 1
        self.tok_to_idx['<EOS>'] = 2
        self.tok_to_idx['<UNK>'] = 3
        for idx, (tok, _) in enumerate(truncated_vocab):
            self.tok_to_idx[tok] = idx + 4
        self.idx_to_tok = {idx: tok for tok, idx in self.tok_to_idx.items()}

    def create_vocab(self):
        if self.parser is None:
            from spacy.lang.en import English
            self.parser = English()
        vocab = dict()
        word2count = dict()
        '''
        with open('cleaned_first_names.txt', 'r') as f:
            lines = f.readlines()
            names = [line.lower().split()[0] for line in lines]  # unambiguous name tokens
        '''

        '''
        for file_idx, file in enumerate(self.files):
            if file_idx % 1000 == 0:
                print("reading file %i/%i" % (file_idx, len(self.files)), flush=True)

            with open(self.data_path + file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 2
                tokens = list(lines)[0].split() + list(lines)[1].split()
                for token in tokens:
                    # do not add name tokens to vocab
                    if token not in names and not contains_digit(token) and '@' not in token and 'http' not in token and 'www' not in token:
                        vocab[token] = vocab.get(token, 0) + 1
                        
        '''
        '''
        vocab = dict()
        with open(self.data_path + "copynet_train.txt", "r") as f:
            lines = f.readlines()
            tokens = list(lines)[0].split() + list(lines)[1].split()
            print(tokens)
            for token in tokens:
                # do not add name tokens to vocab
                if not contains_digit(token) and '@' not in token and 'http' not in token and 'www' not in token:
                    vocab[token] = vocab.get(token, 0) + 1
        return vocab
        '''

        with open(self.data_path + "copynet_train.txt", "r", encoding='utf-8') as f:
            for line in f:
                nl = line.split('\t')[0]
                sql = line.split('\t')[1]
                tokens = nl.split() + sql.split()
                for token in tokens:
                    self.add_word(vocab, word2count, token)
        print("Size of vocab before trimming: " + str(len(vocab)))
        vocab = self.trim(vocab, word2count, 3)
        print("Size of vocab after trimming: " + str(len(vocab)))
        return vocab

    def add_word(self, vocab, word2count, word):
        if word not in vocab:
            vocab[word] = len(vocab)
            word2count[word] = 1
        else:
            word2count[word] += 1

    def trim(self, vocab, word2count, min_count):
        keep_words = []
        for k, v in word2count.items():
            if v >= min_count:
                keep_words.append(k)
        del vocab
        del word2count
        vocab = dict()
        word2count = dict()
        for word in keep_words:
            self.add_word(vocab, word2count, word)
        return vocab

class SequencePairDataset(Dataset):
    def __init__(self,
                 data_path='./data/',
                 maxlen=200,
                 lang=None,
                 vocab_limit=None,
                 val_size=0.1,
                 seed=42,
                 use_cuda=False,
                 use_extended_vocab=True,
                 data_type='test'):

        self.data_path = data_path
        self.maxlen = maxlen
        self.use_cuda = use_cuda
        self.parser = None
        self.val_size = val_size
        self.seed = seed
        self.use_extended_vocab = use_extended_vocab

        '''
        if os.path.isdir(self.data_path):
            self.files = [f for f in os.listdir(self.data_path) if not f.startswith('.')]
            idxs = list(range(len(self.files)))
            random.seed(self.seed)
            random.shuffle(idxs)
            num_val = int(len(idxs) * self.val_size)

            if self.is_val:
                idxs = idxs[:num_val]
            else:
                idxs = idxs[num_val:]

            self.files = [self.files[idx] for idx in idxs]
        else:
            self.files = []
        '''

        if lang is None:
            lang = Language(vocab_limit, self.data_path, parser=self.parser)

        self.lang = lang
        self.pairs = []

        with open(self.data_path + "copynet_" + data_type + ".txt", "r", encoding='utf-8') as pair_file:
            for line in pair_file:
                nl = line.split('\t')[0]
                sql = line.split('\t')[1]
                input_token_list = sql.split()
                output_token_list = nl.split()
                self.pairs.append((input_token_list, output_token_list))
        random.seed(self.seed)
        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)
        #return len(self.files)

    def __getitem__(self, idx):
        """
        :arg
        idx: int

        :returns
        input_token_list: list[int]
        output_token_list: list[int]
        token_mapping: binary array"""

        '''
        with open(self.data_path + self.files[idx], "r", encoding='utf-8') as pair_file:
            input_token_list = pair_file.readline().split()
            output_token_list = pair_file.readline().split()
        '''
        input_token_list = self.pairs[idx][0]
        output_token_list = self.pairs[idx][1]

        input_token_list = (['<SOS>'] + input_token_list + ['<EOS>'])[:self.maxlen]
        output_token_list = (['<SOS>'] + output_token_list + ['<EOS>'])[:self.maxlen]

        input_seq = tokens_to_seq(input_token_list, self.lang.tok_to_idx, self.maxlen, self.use_extended_vocab)
        output_seq = tokens_to_seq(output_token_list, self.lang.tok_to_idx, self.maxlen, self.use_extended_vocab, input_tokens=input_token_list)

        if self.use_cuda:
            input_seq = input_seq.cuda()
            output_seq = output_seq.cuda()

        return input_seq, output_seq, ' '.join(input_token_list), ' '.join(output_token_list)
