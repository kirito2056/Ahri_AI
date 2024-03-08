import openai

# OpenAI API 액세스 키 설정

def generate_response(prompt, max_tokens=50):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo", 
        prompt=prompt,
        max_tokens=max_tokens
    )
    return response.choices[0].text.strip()


while True:
    user_input = input("User: ")
    if user_input.lower() == 'quit':
        break
    prompt = f"You: {user_input}\nAI:"
    response = generate_response(prompt)
    print("AI:", response)
