import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
from dataset import MultiModalTextGenDataset
from model import MultiModalTextGenLSTM

from torchviz import make_dot
# def prepare_sensor_data_dict(sensor_data, num_features_perSensor):
#     return {
#         # 'eye': sensor_data[:, :, :num_features_perSensor['eye']],
#         # 'emg': sensor_data[:, :, num_features_perSensor['eye']:num_features_perSensor['eye'] + num_features_perSensor['emg']],
#         # 'tactile': sensor_data[:, :, num_features_perSensor['eye'] + num_features_perSensor['emg']:num_features_perSensor['eye'] + num_features_perSensor['emg'] + num_features_perSensor['tactile']],
#         # 'body': sensor_data[:, :, num_features_perSensor['eye'] + num_features_perSensor['emg'] + num_features_perSensor['tactile']:]
#         'emg': sensor_data[:, :, :num_features_perSensor['emg']],
#         'tactile': sensor_data[:, :,
#                    num_features_perSensor['emg']:num_features_perSensor['emg'] + num_features_perSensor['tactile']],
#         'body': sensor_data[:, :, num_features_perSensor['emg'] + num_features_perSensor['tactile']:]
#     }
def prepare_sensor_data_dict(sensor_data, num_features_perSensor):
    indices = {
        sensor: sum(num_features_perSensor[s] for s in sorted(num_features_perSensor) if s < sensor)
        for sensor in num_features_perSensor
    }
    # print(indices)  # Print the indices to verify
    sensor_dict = {
        sensor: sensor_data[:, :, indices[sensor]:indices[sensor] + num_features_perSensor[sensor]]
        for sensor in num_features_perSensor
    }
    # for sensor, data in sensor_dict.items():
    #     print(f"Sensor: {sensor}, Shape: {data.shape}")  # Print the shape of each sensor data array to verify
    return sensor_dict
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0001, path='best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.path = path  # Path to save the best model

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif self.best_score - val_loss > self.min_delta:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)
        print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')


early_stopping = EarlyStopping(patience=5, min_delta=0.0001)


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
with open('reverse_vocab.json', 'r') as f:
    reverse_vocab = json.load(f)

def tensor_to_text(tensor, reverse_vocab):
    return ' '.join([reverse_vocab[str(i.item())] for i in tensor])


train_dataloader = dataset.train_dataloader
val_dataloader = dataset.val_dataloader
test_dataloader = dataset.test_dataloader

# Hyperparameters,
num_features_perSensor = {'eye': 2, 'emg': 16, 'tactile': 32, 'body': 66}
# num_features_perSensor = {'body': 66}
nhead = 2
num_layers = 2
vocab_size = dataset.vocab_size
hidden_size = 128
learning_rate = 0.001
num_epochs = 50
model = MultiModalTextGenLSTM(num_features_perSensor, nhead, num_layers, vocab_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.08036246056856665, patience=3, verbose=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = 0

    for sensor_data, tokenized_text_labels, lengths, true_lengths in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):

        sensor_data_dict = prepare_sensor_data_dict(sensor_data, num_features_perSensor)
        outputs = model(sensor_data_dict, tokenized_text_labels, lengths)
        targets = tokenized_text_labels.view(-1)

        loss = criterion(outputs.view(-1, vocab_size), targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    average_loss = total_loss / num_batches
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}")

    # Validation loop
    model.eval()
    total_val_loss = 0
    num_val_batches = 0

    with torch.no_grad():
        for sensor_data, tokenized_text_labels, lengths, true_lengths in val_dataloader:

            sensor_data_dict = prepare_sensor_data_dict(sensor_data, num_features_perSensor)
            outputs = model(sensor_data_dict, tokenized_text_labels, lengths)
            targets = tokenized_text_labels.view(-1)
            loss = criterion(outputs.view(-1, vocab_size), targets)
            total_val_loss += loss.item()
            num_val_batches += 1

    average_val_loss = total_val_loss / num_val_batches
    print(f"Validation Loss after Epoch [{epoch + 1}/{num_epochs}]: {average_val_loss:.4f}")

    # Update the learning rate
    scheduler.step(average_val_loss)

    if early_stopping(average_val_loss, model):
        print("Early stopping triggered.")
        break

    save_folder = "model_checkpoints"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save model checkpoint
    save_path = os.path.join(save_folder, f"model_checkpoint_epoch_{epoch + 1}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'hyperparameters': {
            'num_features_perSensor': num_features_perSensor,
            'nhead': nhead,
            'num_layers': num_layers,
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'learning_rate': learning_rate
        }
    }, save_path)

    print(f"Model saved to {save_path}")

print("Training complete.")
model.load_state_dict(torch.load('best_model.pt'))
model.eval()
total_test_loss = 0
num_test_batches = 0
total_correct_tokens = 0  # Initialize the count of correct tokens
total_tokens = 0
with torch.no_grad():
    for sensor_data, tokenized_text_labels, lengths, true_lengths in test_dataloader:

        sensor_data_dict = prepare_sensor_data_dict(sensor_data, num_features_perSensor)
        outputs = model(sensor_data_dict, tokenized_text_labels, lengths)
        predicted_token_ids = torch.argmax(outputs, dim=2)

        for i, true_length in enumerate(true_lengths):  # Here I've changed "lengths" to "true_lengths"
            predicted_text = tensor_to_text(predicted_token_ids[i, :true_length], reverse_vocab)  # Use true_length to truncate
            ground_truth_text = tensor_to_text(tokenized_text_labels[i, :true_length], reverse_vocab)  # Use true_length to truncate

            print(f"Predicted: {predicted_text}")
            print(f"Ground Truth: {ground_truth_text}")

        targets = tokenized_text_labels.view(-1)
        loss = criterion(outputs.view(-1, vocab_size), targets)
        predicted_token_ids = torch.argmax(outputs, dim=2)
        correct_tokens = (predicted_token_ids == tokenized_text_labels).float().sum()
        total_correct_tokens += correct_tokens
        total_tokens += tokenized_text_labels.numel()

        total_test_loss += loss.item()
        num_test_batches += 1

accuracy = total_correct_tokens / total_tokens
print(f"Accuracy: {accuracy:.4f}")
average_test_loss = total_test_loss / num_test_batches
print(f"Test Loss: {average_test_loss:.4f}")
