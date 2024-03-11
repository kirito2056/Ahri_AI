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

def custom_collate(batch):
    max_input_len = max(len(seq[0]) for seq in batch)
    max_target_len = max(len(seq[1]) for seq in batch)  

    padded_input_seqs = []
    padded_target_seqs = []
    for seq in batch:
        padded_input_seq = torch.nn.functional.pad(seq[0], (0, max_input_len - len(seq[0])))

        padded_target_seq = torch.nn.functional.pad(seq[1], (0, max_target_len - len(seq[1])))
        
        padded_input_seqs.append(padded_input_seq.unsqueeze(0))  
        padded_target_seqs.append(padded_target_seq.unsqueeze(0)) 
    return torch.cat(padded_input_seqs), torch.cat(padded_target_seqs)

#인코딩, 디코딩 작업 수행하는 클래스
class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(EncoderDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_seq, target_seq):
        embedded_input = self.embedding(input_seq)
        encoder_output, _ = self.encoder(embedded_input)

        decoder_hidden = encoder_output[:, -1:, :].clone().permute(1, 0, 2).contiguous()

        embedded_target = self.embedding(target_seq)
        decoder_output, _ = self.decoder(embedded_target, decoder_hidden)

        output = self.fc(decoder_output)
        return output


conversations = load_conversations('dialogues_text.txt')

word2index = build_vocab(conversations)
vocab_size = len(word2index)

input_dim = output_dim = vocab_size
embedding_dim = 100
hidden_dim = 128
learning_rate = 0.001
num_epochs = 10

numericalized_data = numericalize_data(conversations, word2index)

dataset = ConversationDataset(numericalized_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

model = EncoderDecoder(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

"""
for epoch in range(num_epochs):
    for input_seq, target_seq in dataloader:
        optimizer.zero_grad()
        output = model(input_seq, target_seq)
        loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

PATH = '/Users/user/Desktop/Ahri_AI/engine/'
torch.save(model.state_dict(), PATH + 'Ahri.pt')
"""


state_dict_path = '/Users/user/Desktop/Ahri_AI/engine/Ahri.pt'
model = EncoderDecoder(vocab_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('mps')))
model.eval()

def numericalize_sentence(sentence, word2index):
    return [word2index[word] for word in word_tokenize(sentence.lower())]

def tensorize_sentence(sentence_indices):
    return torch.tensor([sentence_indices])

def predict(input_sentence, model, word2index, index2word):
    numericalized_input = numericalize_sentence(input_sentence, word2index)
    input_tensor = tensorize_sentence(numericalized_input)
    with torch.no_grad():
        output_tensor = model(input_tensor, input_tensor)
    predicted_indices = output_tensor.argmax(dim=-1).squeeze(0).tolist()
    predicted_words = [index2word[idx] for idx in predicted_indices]
    return predicted_words

conversations = load_conversations('dialogues_text.txt')

vocab = build_vocab(conversations)
vocab_size = len(vocab)

word2index = {word: idx for idx, word in enumerate(vocab)}
index2word = {idx: word for word, idx in word2index.items()}

index2word = {idx: word for word, idx in word2index.items()}

input_text = ''

while 'get back' not in input_text:
    input_text = input("[ 사용자 ] : ")
    predicted_words = predict(input_text, model, word2index, index2word)
    print('[ Ahri ] : ' + ' '.join(predicted_words))
