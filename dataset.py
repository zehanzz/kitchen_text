import torch
import numpy as np
import h5py
import json
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torchtext.data import get_tokenizer

# Dictionary containing the number of features for each sensor
num_features_perSensor = {
    'eye': 2,
    'emg': 16,
    'tactile': 32,
    'body': 66
}

class MultiModalTextGenDataset:
    def __init__(self, hdf5_path, activities_to_classify):
        self.tokenizer = get_tokenizer("basic_english")
        self.activities_to_classify = activities_to_classify
        self.vocab = {}
        self.vocab_size = 0

        self.load_data(hdf5_path)
        self.preprocess_data()
        self.create_dataloaders()

    def load_data(self, hdf5_path):
        with h5py.File(hdf5_path, 'r') as f:
            self.feature_matrices = np.array(f['example_matrices'])
            self.feature_label_indexes = np.array(f['example_label_indexes'])

        self.text_labels = [self.activities_to_classify[i] for i in self.feature_label_indexes]
        self.tokenized_text_labels = [self.tokenizer(text) for text in self.text_labels]

        vocab = set()
        for tokens in self.tokenized_text_labels:
            vocab.update(tokens)

        self.vocab = {word: i for i, word in enumerate(vocab)}
        self.reverse_vocab = {i: word for word, i in self.vocab.items()}

        with open('reverse_vocab.json', 'w') as f:
            json.dump(self.reverse_vocab, f)

        feature_index_start = 0
        self.sensor_data_dict = {}
        for sensor, num_features in num_features_perSensor.items():
            self.sensor_data_dict[sensor] = self.feature_matrices[:, :, feature_index_start:feature_index_start + num_features]
            feature_index_start += num_features

    def preprocess_data(self):
        self.vocab = {word: i for i, word in enumerate(set([token for tokens in self.tokenized_text_labels for token in tokens]))}
        self.vocab_size = len(self.vocab)

        self.tokenized_text_labels = [
            [self.vocab[token] for token in text] for text in self.tokenized_text_labels
        ]

        # max_length = max(len(tokens) for tokens in self.tokenized_text_labels)
        # self.tokenized_text_labels = [
        #     torch.tensor(tokens + [0] * (max_length - len(tokens)), dtype=torch.long)
        #     for tokens in self.tokenized_text_labels
        # ]
        #
        # self.train_data, self.test_data, self.train_tokenized_text_labels, self.test_tokenized_text_labels = train_test_split(
        #     self.feature_matrices, self.tokenized_text_labels, test_size=0.2, random_state=42, stratify=self.feature_label_indexes
        # )
        #
        # self.train_data, self.val_data, self.train_tokenized_text_labels, self.val_tokenized_text_labels = train_test_split(
        #     self.train_data, self.train_tokenized_text_labels, test_size=0.25, random_state=42
        # )
        #
        # self.train_lengths = [len(tokens) for tokens in self.train_tokenized_text_labels]
        # self.val_lengths = [len(tokens) for tokens in self.val_tokenized_text_labels]
        # self.test_lengths = [len(tokens) for tokens in self.test_tokenized_text_labels]
        max_length = max(len(tokens) for tokens in self.tokenized_text_labels)
        self.train_data, self.test_data, self.train_tokenized_text_labels, self.test_tokenized_text_labels = train_test_split(
            self.feature_matrices, self.tokenized_text_labels, test_size=0.2, random_state=42,
            stratify=self.feature_label_indexes
        )

        self.train_data, self.val_data, self.train_tokenized_text_labels, self.val_tokenized_text_labels = train_test_split(
            self.train_data, self.train_tokenized_text_labels, test_size=0.25, random_state=42
        )

        # 2. Calculate the lengths
        self.train_lengths_true = [len(tokens) for tokens in self.train_tokenized_text_labels]
        self.val_lengths_true = [len(tokens) for tokens in self.val_tokenized_text_labels]
        self.test_lengths_true = [len(tokens) for tokens in self.test_tokenized_text_labels]

        # 3. Then perform the padding

        self.train_tokenized_text_labels = [
            torch.tensor(tokens + [0] * (max_length - len(tokens)), dtype=torch.long)
            for tokens in self.train_tokenized_text_labels
        ]
        self.val_tokenized_text_labels = [
            torch.tensor(tokens + [0] * (max_length - len(tokens)), dtype=torch.long)
            for tokens in self.val_tokenized_text_labels
        ]
        self.test_tokenized_text_labels = [
            torch.tensor(tokens + [0] * (max_length - len(tokens)), dtype=torch.long)
            for tokens in self.test_tokenized_text_labels
        ]
        self.train_lengths = [len(tokens) for tokens in self.train_tokenized_text_labels]
        self.val_lengths = [len(tokens) for tokens in self.val_tokenized_text_labels]
        self.test_lengths = [len(tokens) for tokens in self.test_tokenized_text_labels]

    def create_dataloaders(self):
        self.train_dataset = TensorDataset(
            torch.tensor(self.train_data, dtype=torch.float32),
            torch.stack(self.train_tokenized_text_labels),
            torch.tensor(self.train_lengths, dtype=torch.long),
            torch.tensor(self.train_lengths_true, dtype=torch.long)
        )

        self.val_dataset = TensorDataset(
            torch.tensor(self.val_data, dtype=torch.float32),
            torch.stack(self.val_tokenized_text_labels),
            torch.tensor(self.val_lengths, dtype=torch.long),
            torch.tensor(self.val_lengths_true, dtype=torch.long)
        )

        self.test_dataset = TensorDataset(
            torch.tensor(self.test_data, dtype=torch.float32),
            torch.stack(self.test_tokenized_text_labels),
            torch.tensor(self.test_lengths, dtype=torch.long),
            torch.tensor(self.test_lengths_true, dtype=torch.long)
        )

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)


if __name__ == '__main__':
    # Initialize the dataset
    hdf5_path = './data_processed/data_processed_allStreams_10s_10hz_5subj_ex20-20_allActs.hdf5'
    baseline_label = 'None'
    activities_to_classify = [
        baseline_label,
        'Get/replace items from refrigerator/cabinets/drawers',
        'Peel a cucumber',
        'Clear cutting board',
        'Slice a cucumber',
        'Peel a potato',
        'Slice a potato',
        'Slice bread',
        'Spread almond butter on a bread slice',
        'Spread jelly on a bread slice',
        'Open/close a jar of almond butter',
        'Pour water from a pitcher into a glass',
        'Clean a plate with a sponge',
        'Clean a plate with a towel',
        'Clean a pan with a sponge',
        'Clean a pan with a towel',
        'Get items from cabinets: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
        'Set table: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
        'Stack on table: 3 each large/small plates, bowls',
        'Load dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
        'Unload dishwasher: 3 each large/small plates, bowls, mugs, glasses, sets of utensils',
    ]
    dataset = MultiModalTextGenDataset(hdf5_path, activities_to_classify)

    train_loader = dataset.train_dataloader
    val_loader = dataset.val_dataloader
    test_loader = dataset.test_dataloader

    print(f"Number of samples in train loader: {len(train_loader.dataset)}")
    print(f"Number of samples in validation loader: {len(val_loader.dataset)}")
    print(f"Number of samples in test loader: {len(test_loader.dataset)}")
    print(f"Vocabulary size: {dataset.vocab_size}")

    assert len(train_loader.dataset) == len(dataset.train_tokenized_text_labels), "Length mismatch between training data and labels"
    assert len(val_loader.dataset) == len(dataset.val_tokenized_text_labels), "Length mismatch between validation data and labels"
    assert len(test_loader.dataset) == len(dataset.test_tokenized_text_labels), "Length mismatch between test data and labels"

    for i, (data, labels, lengths) in enumerate(train_loader):
        print(f"Batch {i+1} - Data: {data.shape}, Labels: {labels.shape}, Lengths: {len(lengths)}")

    for data, labels, lengths in train_loader:
        assert data.dtype == torch.float32, "Unexpected data type for training data"
        assert labels.dtype == torch.long, "Unexpected data type for training labels"

    for data, labels, lengths in train_loader:
        assert torch.all(torch.isfinite(data)), "Training data contains NaN or Inf"
        assert torch.all(torch.isfinite(labels)), "Training labels contain NaN or Inf"

    print("All tests passed!")
