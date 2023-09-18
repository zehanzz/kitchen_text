import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Transformer Encoder block
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """Forward pass for the transformer encoder block."""
        return self.encoder(x)


# MultiModal Text Generation Model using LSTM
class MultiModalTextGenLSTM(nn.Module):
    def __init__(self, sensor_dims, nhead, num_layers, vocab_size, hidden_size):
        super(MultiModalTextGenLSTM, self).__init__()

        # Initialize encoders for each sensor type
        self.encoders = nn.ModuleDict({
            sensor: TransformerEncoder(d_model=dim, nhead=nhead, num_layers=num_layers)
            for sensor, dim in sensor_dims.items()
        })

        # Initialize LSTM-based decoder for text generation
        self.decoder = SeqTextGenerator(sum(sensor_dims.values()), hidden_size, vocab_size)

    def forward(self, x_dict, tgt, tgt_lengths):
        """Forward pass for the multi-modal text generation model."""

        # Encode each type of sensor data
        encoded = [self.encoders[sensor](x) for sensor, x in x_dict.items()]

        # Concatenate the encoded sensor data
        concatenated = torch.cat(encoded, dim=-1)

        # Initialize the hidden state for the LSTM decoder
        batch_size = concatenated.size(0)
        hidden = (torch.zeros(1, batch_size, self.decoder.hidden_size).to(concatenated.device),
                  torch.zeros(1, batch_size, self.decoder.hidden_size).to(concatenated.device))

        # Prepare the input for the LSTM decoder
        packed_input = pack_padded_sequence(concatenated, tgt_lengths, batch_first=True, enforce_sorted=False)

        # Decode the concatenated features into output
        output, _ = self.decoder(packed_input, hidden)

        return output


# LSTM-based Decoder for sequential text generation
class SeqTextGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SeqTextGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """Forward pass for the LSTM-based text generator."""

        # Pass the input through the LSTM layers
        lstm_out, hidden = self.lstm(x, hidden)

        # Unpack the output if it's a PackedSequence
        if isinstance(lstm_out, torch.nn.utils.rnn.PackedSequence):
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        # Fully connected layer
        output = self.fc_out(lstm_out)

        return output, hidden
