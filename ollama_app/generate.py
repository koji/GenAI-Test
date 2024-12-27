import ollama

model = "llama3.2"

response = ollama.generate(
    model=model,
    prompt="why is the sea blue?",
    # max_tokens=100,
    # temperature=0.5,
    # top_p=0.5,
    # frequency_penalty=0.5,
    # presence_penalty=0.5,
)


print(ollama.show("llama3.2"))
