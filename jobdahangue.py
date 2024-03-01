from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2" 
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def chat(prompt, max_length=100, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, temperature=temperature, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id, do_sample=True)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

print("종료할땐 exit")
while True:
    user_input = input("사용자: ")
    if user_input.lower() == 'exit':
        print("대화 종료.")
        break
    elif user_input.strip() == "":
        print("사용자: (빈 문자열)")
        print("GPT-2: 입력 없슴")
        continue
    response = chat(user_input)
    print("GPT-2:", response)
