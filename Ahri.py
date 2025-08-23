import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import numpy as np
import os # 파일 경로 처리를 위해 추가
import torch.nn.functional as F # log_softmax 사용을 위해 추가
import heapq # Beam Search에서 top-k 후보 관리를 위해 추가

# Special tokens
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

def load_conversations(file_path):
    conversations = []
    inputs = []
    targets = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        # 빈 줄이나 공백만 있는 줄을 필터링
        valid_lines = [line.strip() for line in lines if line.strip()]

        # 필터링된 줄들을 2개씩 짝지어 처리
        for i in range(0, len(valid_lines), 2):
            # 마지막 줄이 홀수개일 경우 인덱스 에러 방지
            if i + 1 < len(valid_lines):
                input_text = valid_lines[i]
                target_text = valid_lines[i+1]
                conversations.append((input_text, target_text))
            else:
                print(f"Warning: Skipping last line as it doesn't form a pair: {valid_lines[i]}")
    return conversations

def build_vocab(conversations):
    # 스페셜 토큰은 고정 인덱스 부여 (PAD=0, SOS=1, EOS=2, UNK=3)
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']

    token_set = set()
    for input_text, target_text in conversations:
        token_set.update(word_tokenize(input_text.lower()))
        token_set.update(word_tokenize(target_text.lower()))

    # 스페셜 토큰을 제외하고 알파벳 순서로 정렬하여 결정적 인덱싱 보장
    token_list = sorted(t for t in token_set if t not in special_tokens)
    vocab_list = special_tokens + token_list

    word2index = {word: idx for idx, word in enumerate(vocab_list)}
    index2word = {idx: word for word, idx in word2index.items()}
    return word2index, index2word

def numericalize_data(conversations, word2index):
    numericalized_data = []
    for input_text, target_text in conversations:
        # UNK 토큰 처리 추가
        input_indices = [word2index.get(word, word2index['<UNK>']) for word in word_tokenize(input_text.lower())]
        target_indices = [word2index.get(word, word2index['<UNK>']) for word in word_tokenize(target_text.lower())]
        # <EOS> 토큰 추가 (학습 데이터에도 추가하는 것이 일반적)
        numericalized_data.append((input_indices, target_indices + [word2index['<EOS>']]))
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
        # PAD 토큰 인덱스(0)로 패딩
        padded_input_seq = torch.nn.functional.pad(seq[0], (0, max_input_len - len(seq[0])), value=PAD_token)
        padded_target_seq = torch.nn.functional.pad(seq[1], (0, max_target_len - len(seq[1])), value=PAD_token)

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

    def forward(self, input_seq, target_seq=None, max_length=50): # 예측 시 target_seq는 None, max_length 추가
        embedded_input = self.embedding(input_seq)
        encoder_output, encoder_hidden = self.encoder(embedded_input) # encoder_hidden도 받음

        # 예측 모드 (target_seq가 없을 때)
        if target_seq is None:
            batch_size = input_seq.size(0)
            decoder_input = torch.tensor([[SOS_token] * batch_size], device=input_seq.device).view(batch_size, 1) # <SOS> 토큰으로 시작 (batch 처리 고려)
            decoder_hidden = encoder_hidden # 인코더의 마지막 hidden state 사용
            decoded_outputs = []

            for _ in range(max_length):
                embedded_decoder_input = self.embedding(decoder_input)
                # decoder_hidden 유지하며 순환
                decoder_output, decoder_hidden = self.decoder(embedded_decoder_input, decoder_hidden)
                output = self.fc(decoder_output) # (batch_size, 1, vocab_size)

                # Greedy decoding
                topv, topi = output.topk(1, dim=2)
                decoder_input = topi.squeeze(2).detach() # 다음 입력으로 사용

                decoded_outputs.append(topi.squeeze().item()) # 예측된 단어 인덱스 저장

                # <EOS> 토큰 만나면 종료
                if decoder_input.item() == EOS_token:
                    break
            # 예측된 단어 인덱스 텐서 반환 (추후 후처리 필요)
            # 여기서는 간단히 리스트를 반환 (추후 predict 함수에서 처리)
            # 실제 구현에서는 텐서 형태로 반환하는 것이 더 일반적
            return torch.tensor(decoded_outputs) # 예측 결과를 텐서로 반환

        # 학습 모드 (target_seq가 있을 때)
        else:
            # Teacher Forcing: target_seq를 디코더 입력으로 사용
            decoder_hidden = encoder_hidden # 인코더의 마지막 hidden state 사용

            # <SOS> 토큰을 타겟 시퀀스 앞에 추가하여 디코더 입력 생성
            sos_tensor = torch.tensor([[SOS_token]], device=target_seq.device).repeat(target_seq.size(0), 1)
            decoder_input_seq = torch.cat((sos_tensor, target_seq[:, :-1]), dim=1) # 마지막 토큰 제외하고 <SOS> 추가
            embedded_target = self.embedding(decoder_input_seq)

            decoder_output, _ = self.decoder(embedded_target, decoder_hidden)
            output = self.fc(decoder_output)
            return output


conversations = load_conversations('dialogues_text.txt')

# build_vocab 수정 반영 (deterministic)
word2index, index2word = build_vocab(conversations)
vocab_size = len(word2index) # Special tokens 포함된 크기

input_dim = output_dim = vocab_size
embedding_dim = 100
hidden_dim = 128
learning_rate = 0.001
num_epochs = 10 # 에포크 수 증가 (예: 50 또는 100)

numericalized_data = numericalize_data(conversations, word2index)

dataset = ConversationDataset(numericalized_data)
# DataLoader의 collate_fn 수정 반영 확인
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

model = EncoderDecoder(vocab_size, embedding_dim, hidden_dim)
# PAD 토큰은 손실 계산에서 제외
criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프 (수정된 모델 forward 및 criterion 사용)
for epoch in range(num_epochs):
    for input_seq, target_seq in dataloader:
        optimizer.zero_grad()
        # 모델 forward 호출 방식 변경됨 (학습 시에는 target_seq 전달)
        output = model(input_seq, target_seq)
        # 손실 계산 시 output과 target_seq 형태 맞춰주기
        # output: (batch_size, seq_len, vocab_size)
        # target_seq: (batch_size, seq_len)
        loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
        loss.backward()
        # 기울기 클리핑 추가 (기울기 폭주 방지)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 상대 경로 사용 및 engine 디렉토리 생성 확인
model_dir = 'engine'
os.makedirs(model_dir, exist_ok=True) # 디렉토리 없으면 생성
model_path = os.path.join(model_dir, 'Ahri.pt')
# 모델 상태 + 어휘 + 설정을 함께 저장 (결정적 재현성)
checkpoint = {
    'model_state_dict': model.state_dict(),
    'word2index': word2index,
    'index2word': index2word,
    'config': {
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
    },
}
torch.save(checkpoint, model_path)
print(f"Checkpoint saved to {model_path}")

# --- 예측 부분 ---
model_dir = 'engine' # 상대 경로 사용
model_path = os.path.join(model_dir, 'Ahri.pt')

# 디바이스 설정 (가용 시 MPS 사용, 아니면 CPU)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 체크포인트 로드 (모델+어휘+설정)
ckpt = torch.load(model_path, map_location=device)
loaded_word2index = ckpt['word2index']
loaded_index2word = ckpt['index2word']
loaded_config = ckpt['config']

# 체크포인트 설정으로 모델 재생성 및 가중치 로드
vocab_size_loaded = len(loaded_word2index)
model = EncoderDecoder(vocab_size_loaded, loaded_config['embedding_dim'], loaded_config['hidden_dim'])
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

# 예측용 어휘 매핑 (학습 시 저장된 것 사용)
word2index_predict = loaded_word2index
index2word = loaded_index2word

def numericalize_sentence(sentence, word2index):
    # UNK 토큰 처리 추가
    return [word2index.get(word, word2index['<UNK>']) for word in word_tokenize(sentence.lower())]

def tensorize_sentence(sentence_indices, device): # device 인자 추가
    return torch.tensor([sentence_indices]).to(device) # 텐서를 해당 디바이스로 이동

MAX_LENGTH = 50 # 최대 생성 길이
BEAM_WIDTH = 3 # Beam Search 너비 (조절 가능)

def predict(input_sentence, model, word2index, index2word, device, max_length=MAX_LENGTH, beam_width=BEAM_WIDTH): # beam_width 추가
    model.eval() # 예측 모드 설정
    numericalized_input = numericalize_sentence(input_sentence, word2index)
    if not numericalized_input:
        return ["죄송해요, 무슨 말씀이신지 잘 모르겠어요."]
    input_tensor = tensorize_sentence(numericalized_input, device)

    with torch.no_grad():
        # 1. 인코더 실행
        embedded_input = model.embedding(input_tensor)
        encoder_outputs, encoder_hidden = model.encoder(embedded_input)

        # 2. Beam Search 초기화
        # 각 beam은 (log 확률 합계, [토큰 인덱스 리스트], 디코더 히든 상태) 튜플
        # 초기 beam: <SOS> 토큰으로 시작
        decoder_hidden = encoder_hidden # 인코더의 마지막 hidden state 사용
        start_token_index = word2index['<SOS>']
        initial_beam = (0.0, [start_token_index], decoder_hidden)
        beams = [initial_beam]
        completed_beams = []

        # 3. Beam Search 단계별 실행
        for _ in range(max_length):
            new_beams = []
            for log_prob_sum, tokens, hidden in beams:
                # 마지막 토큰이 <EOS>면 완료된 beam으로 이동
                if tokens[-1] == word2index['<EOS>']:
                    completed_beams.append((log_prob_sum, tokens))
                    # 완료된 beam은 더 이상 확장하지 않음 (아래 for문 실행 방지)
                    # beam_width 유지 위해 빈 튜플 추가 (임시 방편, heapq 사용 시 불필요)
                    # heapq 방식으로 변경하면 이 부분 필요 없어짐
                    continue # 다음 beam 처리

                # 디코더 입력 준비 (마지막 토큰)
                decoder_input = torch.tensor([[tokens[-1]]], device=device)
                embedded_decoder_input = model.embedding(decoder_input)

                # 디코더 실행
                decoder_output, next_hidden = model.decoder(embedded_decoder_input, hidden)
                output_logits = model.fc(decoder_output.squeeze(0))
                log_probs = F.log_softmax(output_logits, dim=-1) # Log 확률 계산

                # Top-k 후보 토큰 선택 (beam_width개)
                top_log_probs, top_indices = log_probs.topk(beam_width)

                # 각 후보 토큰으로 beam 확장
                for i in range(beam_width):
                    next_token_index = top_indices[0][i].item()
                    next_log_prob = top_log_probs[0][i].item()
                    new_log_prob_sum = log_prob_sum + next_log_prob
                    new_tokens = tokens + [next_token_index]
                    new_beam = (new_log_prob_sum, new_tokens, next_hidden)
                    new_beams.append(new_beam)

            # 확장된 모든 new_beams 중에서 확률 높은 상위 beam_width개만 선택
            # heapq를 사용하여 효율적으로 상위 k개 선택
            beams = heapq.nlargest(beam_width, new_beams, key=lambda x: x[0])

            # 모든 활성 beam이 종료되었는지 확인 (선택적: 조기 종료)
            if all(b[1][-1] == word2index['<EOS>'] for b in beams):
                 completed_beams.extend(beams) # 현재 beam들도 완료 처리
                 break

        # 4. 최종 결과 선택
        # 완료된 beam이 없으면 현재 가장 확률 높은 beam 사용
        if not completed_beams:
            completed_beams.extend(beams)

        # 확률 정규화 (길이로 나누기)하여 가장 좋은 beam 선택 (선택적)
        # completed_beams.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
        # 여기서는 단순 확률 합계로 정렬
        completed_beams.sort(key=lambda x: x[0], reverse=True)

        best_beam_tokens = completed_beams[0][1]

    # 5. 결과 후처리 (토큰 -> 단어 변환, 특수 토큰 제거)
    predicted_words = []
    for idx in best_beam_tokens:
        if idx == word2index['<EOS>']:
            break
        if idx not in [word2index['<PAD>'], word2index['<SOS>'], word2index['<UNK>']]:
            predicted_words.append(index2word.get(idx, "<UNK>"))

    if not predicted_words: # <SOS> 다음에 바로 <EOS>가 나온 경우 등
        return ["...",]

    return predicted_words

# index2word는 체크포인트에서 로드됨

# 메인 루프
print("[ Ahri ] : 안녕하세요! 무엇을 도와드릴까요? (종료하려면 'get back' 입력)")
input_text = ''
while 'get back' not in input_text.lower(): # 종료 조건 소문자 처리
    input_text = input("[ 사용자 ] : ")
    if 'get back' in input_text.lower(): # 종료 조건 확인
        break
    predicted_words = predict(input_text, model, word2index_predict, index2word, device) # device 전달
    print('[ Ahri ] : ' + ' '.join(predicted_words))

print("[ Ahri ] : 다음에 또 만나요!")
