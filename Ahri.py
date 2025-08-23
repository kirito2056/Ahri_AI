import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import numpy as np
import os # 파일 경로 처리를 위해 추가
import torch.nn.functional as F # log_softmax 사용을 위해 추가
import heapq # Beam Search에서 top-k 후보 관리를 위해 추가
import nltk

# Special tokens
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

def ensure_nltk_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

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
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_token)
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


def numericalize_sentence(sentence, word2index):
    # UNK 토큰 처리 추가
    return [word2index.get(word, word2index['<UNK>']) for word in word_tokenize(sentence.lower())]

def tensorize_sentence(sentence_indices, device): # device 인자 추가
    return torch.tensor([sentence_indices]).to(device) # 텐서를 해당 디바이스로 이동

MAX_LENGTH = 50 # 최대 생성 길이
BEAM_WIDTH = 3 # Beam Search 너비 (조절 가능)

def _apply_repetition_penalty(logits, generated_tokens, repetition_penalty: float):
    if repetition_penalty == 1.0 or len(generated_tokens) == 0:
        return logits
    unique_tokens = set(generated_tokens)
    for token_id in unique_tokens:
        logits[0, token_id] /= repetition_penalty
    return logits

def _get_ngram_banned_tokens(generated_tokens, no_repeat_ngram_size: int):
    if no_repeat_ngram_size <= 1 or len(generated_tokens) < no_repeat_ngram_size - 1:
        return set()
    n = no_repeat_ngram_size
    prefix_to_next = {}
    for i in range(len(generated_tokens) - n + 1):
        prefix = tuple(generated_tokens[i:i + n - 1])
        next_tok = generated_tokens[i + n - 1]
        prefix_to_next.setdefault(prefix, set()).add(next_tok)
    current_prefix = tuple(generated_tokens[-(n - 1):])
    return prefix_to_next.get(current_prefix, set())

def _top_k_top_p_filtering(logits, top_k: int = 0, top_p: float = 1.0):
    # logits: (1, vocab)
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k)[0][..., -1, None]
        logits = torch.where(logits < threshold, torch.full_like(logits, float('-inf')), logits)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits

def _decode_greedy(model, device, word2index, max_length, initial_hidden, repetition_penalty=1.0, no_repeat_ngram_size=0):
    decoder_hidden = initial_hidden
    decoded_tokens = [word2index['<SOS>']]
    for _ in range(max_length):
        decoder_input = torch.tensor([[decoded_tokens[-1]]], device=device)
        embedded_decoder_input = model.embedding(decoder_input)
        decoder_output, decoder_hidden = model.decoder(embedded_decoder_input, decoder_hidden)
        logits = model.fc(decoder_output.squeeze(0)) # (1, vocab)
        logits = _apply_repetition_penalty(logits, decoded_tokens, repetition_penalty)
        banned = _get_ngram_banned_tokens(decoded_tokens, no_repeat_ngram_size)
        if len(banned) > 0:
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[0, list(banned)] = True
            logits = logits.masked_fill(mask, float('-inf'))
        next_token = torch.argmax(logits, dim=-1).item()
        decoded_tokens.append(next_token)
        if next_token == word2index['<EOS>']:
            break
    return decoded_tokens

def _decode_beam(model, device, word2index, max_length, initial_hidden, beam_width, length_alpha=0.0, repetition_penalty=1.0, no_repeat_ngram_size=0):
    start_token_index = word2index['<SOS>']
    beams = [(0.0, [start_token_index], initial_hidden)] # (score, tokens, hidden)
    completed = []
    for _ in range(max_length):
        new_beams = []
        for score, tokens, hidden in beams:
            if tokens[-1] == word2index['<EOS>']:
                completed.append((score, tokens))
                continue
            decoder_input = torch.tensor([[tokens[-1]]], device=device)
            embedded_decoder_input = model.embedding(decoder_input)
            decoder_output, next_hidden = model.decoder(embedded_decoder_input, hidden)
            logits = model.fc(decoder_output.squeeze(0))
            logits = _apply_repetition_penalty(logits, tokens, repetition_penalty)
            banned = _get_ngram_banned_tokens(tokens, no_repeat_ngram_size)
            if len(banned) > 0:
                mask = torch.zeros_like(logits, dtype=torch.bool)
                mask[0, list(banned)] = True
                logits = logits.masked_fill(mask, float('-inf'))
            log_probs = F.log_softmax(logits, dim=-1)
            top_log_probs, top_indices = log_probs.topk(beam_width)
            for i in range(beam_width):
                next_token = top_indices[0, i].item()
                next_log_prob = top_log_probs[0, i].item()
                new_tokens = tokens + [next_token]
                new_score = score + next_log_prob
                new_beams.append((new_score, new_tokens, next_hidden))
        if not new_beams:
            break
        beams = heapq.nlargest(beam_width, new_beams, key=lambda x: x[0])
        if all(b[1][-1] == word2index['<EOS>'] for b in beams):
            completed.extend([(s, t) for s, t, _ in beams])
            break
    if not completed:
        completed.extend([(s, t) for s, t, _ in beams])
    if length_alpha > 0.0:
        completed.sort(key=lambda x: x[0] / (len(x[1]) ** length_alpha), reverse=True)
    else:
        completed.sort(key=lambda x: x[0], reverse=True)
    return completed[0][1]

def _decode_sample(model, device, word2index, max_length, initial_hidden, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0):
    decoder_hidden = initial_hidden
    decoded_tokens = [word2index['<SOS>']]
    for _ in range(max_length):
        decoder_input = torch.tensor([[decoded_tokens[-1]]], device=device)
        embedded_decoder_input = model.embedding(decoder_input)
        decoder_output, decoder_hidden = model.decoder(embedded_decoder_input, decoder_hidden)
        logits = model.fc(decoder_output.squeeze(0))
        if temperature != 1.0:
            logits = logits / temperature
        logits = _apply_repetition_penalty(logits, decoded_tokens, repetition_penalty)
        banned = _get_ngram_banned_tokens(decoded_tokens, no_repeat_ngram_size)
        if len(banned) > 0:
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask[0, list(banned)] = True
            logits = logits.masked_fill(mask, float('-inf'))
        logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        decoded_tokens.append(next_token)
        if next_token == word2index['<EOS>']:
            break
    return decoded_tokens

def _decode_self_consistency(sample_fn, num_samples):
    sequences = []
    counts = {}
    for _ in range(num_samples):
        seq = sample_fn()
        sequences.append(tuple(seq))
        counts[sequences[-1]] = counts.get(sequences[-1], 0) + 1
    best = max(counts.items(), key=lambda x: x[1])[0]
    return list(best)

def predict(input_sentence, model, word2index, index2word, device, max_length=MAX_LENGTH, beam_width=BEAM_WIDTH, strategy='beam', temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, no_repeat_ngram_size=0, num_samples=5, length_alpha=0.0): # AZR 옵션 추가
    model.eval() # 예측 모드 설정
    numericalized_input = numericalize_sentence(input_sentence, word2index)
    if not numericalized_input:
        return ["죄송해요, 무슨 말씀이신지 잘 모르겠어요."]
    input_tensor = tensorize_sentence(numericalized_input, device)

    with torch.no_grad():
        # 1. 인코더 실행
        embedded_input = model.embedding(input_tensor)
        encoder_outputs, encoder_hidden = model.encoder(embedded_input)
        # 2. 전략별 디코딩
        if strategy == 'greedy':
            best_beam_tokens = _decode_greedy(model, device, word2index, max_length, encoder_hidden, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size)
        elif strategy == 'beam':
            best_beam_tokens = _decode_beam(model, device, word2index, max_length, encoder_hidden, beam_width, length_alpha=length_alpha, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size)
        elif strategy == 'sample':
            best_beam_tokens = _decode_sample(model, device, word2index, max_length, encoder_hidden, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size)
        elif strategy == 'self_consistency':
            def _sample_once():
                return _decode_sample(model, device, word2index, max_length, encoder_hidden, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size)
            best_beam_tokens = _decode_self_consistency(_sample_once, num_samples=num_samples)
        else: # auto
            if ('?' in input_sentence) or (len(numericalized_input) >= 12):
                best_beam_tokens = _decode_sample(model, device, word2index, max_length, encoder_hidden, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size)
            else:
                best_beam_tokens = _decode_beam(model, device, word2index, max_length, encoder_hidden, beam_width, length_alpha=length_alpha, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size)

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

def train_model():
    ensure_nltk_punkt()

    conversations = load_conversations('dialogues_text.txt')

    # build_vocab 수정 반영 (deterministic)
    word2index, index2word = build_vocab(conversations)
    vocab_size = len(word2index) # Special tokens 포함된 크기

    embedding_dim = 100
    hidden_dim = 128
    learning_rate = 0.001
    num_epochs = 10 # 에포크 수 증가 (예: 50 또는 100)

    numericalized_data = numericalize_data(conversations, word2index)

    dataset = ConversationDataset(numericalized_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

    # 디바이스 설정 (가용 시 MPS 사용, 아니면 CPU)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = EncoderDecoder(vocab_size, embedding_dim, hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 학습 루프
    for epoch in range(num_epochs):
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            optimizer.zero_grad()
            output = model(input_seq, target_seq)
            loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 체크포인트 저장
    model_dir = 'engine'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'Ahri.pt')
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

def run_inference(args=None):
    ensure_nltk_punkt()

    # 디바이스 설정 (가용 시 MPS 사용, 아니면 CPU)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_dir = 'engine'
    model_path = os.path.join(model_dir, 'Ahri.pt')

    # 체크포인트 로드 (모델+어휘+설정)
    ckpt = torch.load(model_path, map_location=device)
    loaded_word2index = ckpt['word2index']
    loaded_index2word = ckpt['index2word']
    loaded_config = ckpt['config']

    vocab_size_loaded = len(loaded_word2index)
    model = EncoderDecoder(vocab_size_loaded, loaded_config['embedding_dim'], loaded_config['hidden_dim'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    word2index_predict = loaded_word2index
    index2word = loaded_index2word
 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'infer'], default='infer')
    # AZR/디코딩 옵션
    parser.add_argument('--strategy', choices=['auto', 'beam', 'greedy', 'sample', 'self_consistency'], default='auto')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--beam_width', type=int, default=BEAM_WIDTH)
    parser.add_argument('--max_length', type=int, default=MAX_LENGTH)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--length_alpha', type=float, default=0.0)
    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    else:
        run_inference(args)
