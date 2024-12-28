import ollama

modelfile = """
FROM llama3.2
SYSTEM You are vert smart assistant who knows everything about the world. You are very succinct and informative.
PARAMETER temperature 0.5
"""

model = "knoweverything"
ollama.create(model=model, modelfile=modelfile)


res = ollama.generate(model=model, prompt="why is the sea blue?")

print(res["response"])


# delete model
ollama.delete(model=model)
