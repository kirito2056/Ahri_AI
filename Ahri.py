import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import numpy as np

def load_conversations(file_path):
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            input_text = lines[i].strip()
            target_text = lines[i+1].strip()
            conversations.append((input_text, target_text))
    return conversations

def build_vocab(conversations):
    vocab = set()
    for input_text, target_text in conversations:
        vocab.update(word_tokenize(input_text.lower()))
        vocab.update(word_tokenize(target_text.lower()))
    word2index = {word: idx for idx, word in enumerate(vocab)}
    return word2index

def numericalize_data(conversations, word2index):
    numericalized_data = []
    for input_text, target_text in conversations:
        input_indices = [word2index[word] for word in word_tokenize(input_text.lower())]
        target_indices = [word2index[word] for word in word_tokenize(target_text.lower())]
        numericalized_data.append((input_indices, target_indices))
    return numericalized_data

class ConversationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_indices, target_indices = self.data[index]
        return torch.tensor(input_indices), torch.tensor(target_indices)

class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(EncoderDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim)
        self.decoder = nn.GRU(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_seq, target_seq):
        embedded_input = self.embedding(input_seq)
        encoder_output, encoder_hidden = self.encoder(embedded_input)

        embedded_target = self.embedding(target_seq)
        decoder_output, _ = self.decoder(embedded_target, encoder_hidden)

        output = self.fc(decoder_output)
        return output


conversations = load_conversations('conversation_data.txt')

word2index = build_vocab(conversations)
vocab_size = len(word2index)

input_dim = output_dim = vocab_size 
embedding_dim = 100
hidden_dim = 128
learning_rate = 0.001
num_epochs = 10

numericalized_data = numericalize_data(conversations, word2index)

dataset = ConversationDataset(numericalized_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SimpleEncoderDecoder(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for input_seq, target_seq in dataloader:
        optimizer.zero_grad()
        output = model(input_seq, target_seq)
        loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")