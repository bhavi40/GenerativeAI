from huggingface_hub import InferenceClient

client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token="YOUR_HF_TOKEN"   # move to env var later
)

def ask_llm(question: str) -> str:
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ],
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("Type 'exit' to quit.\n")

    while True:
        user_question = input("You: ")

        if user_question.lower() in ["exit", "quit"]:
            print("Goodbye 👋")
            break

        if not user_question.strip():
            print("Please enter a valid question.")
            continue

        answer = ask_llm(user_question)
        print("\nLLM:", answer, "\n")
