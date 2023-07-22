import os
import random
import torch
from torch.utils.data import Dataset


def get_train_test_datasets(src, train_pct, seq_len, vocab, device):
    files = [os.path.join(src, f) for f in os.listdir(src)]
    random.shuffle(files)
    no_train = int(train_pct * len(files))

    train_files = files[:no_train]
    test_files = files[no_train:]

    train_data = ChessData(
        files=train_files,
        seq_len=seq_len,
        vocab=vocab,
        device=device
    )

    test_data = ChessData(
        files=test_files,
        seq_len=seq_len,
        vocab=vocab,
        device=device
    )


    return train_data, test_data


class ChessData(Dataset):
    def __init__(self, files, seq_len, vocab, device):
        self.files = files
        self.seq_len = seq_len
        self.vocab = vocab
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        content = open(file, 'r').read()
        x_batch, y_batch = self.get_batch_from_sequence(content)
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        return x_batch, y_batch


    def get_batch_from_sequence(self, sequence):
        x_batch = []
        y_batch = []
        sequence, _ = self.vocab.encode(sequence)


        for i in range(len(sequence)-1):
            x = sequence[:i+1]
            y = sequence[i+1]

            x = self.vocab.pad(x, self.seq_len)

            x_batch.append(x)
            y_batch.append(y)


        x_batch = torch.Tensor(x_batch).long()
        y_batch = torch.Tensor(y_batch).long()

        return x_batch, y_batch