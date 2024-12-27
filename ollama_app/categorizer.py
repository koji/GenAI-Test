import ollama
import os

# model = "llama3.2"
model = "gemma2:9b"

input_file = "./data/grocery_list.txt"
output_file = f"./data/{model}_categorized_grocery_list.txt"

if not os.path.exists(input_file):
    print(f"Input file '{input_file}' does not exist.")
    exit(1)


with open(input_file, "r") as f:
    items = f.read().strip()


prompt = f"""
You are an assistant that categorizes and sorts grocery items.

Here is a list of items:
{items}

Please :

1. Categorize these items into appropriate categories such as Produce, Dairy, Meat, Bakery, Beverages, etc.
2. Sort the items alphabetically within each category.
3. Present the categorized list in a clear and organized manner, using bullet points or numbering.

"""

try:
    response = ollama.generate(
        model=model,
        prompt=prompt,)
    generated_text = response.get("response", "")

    # write the categorized list to a file
    with open(output_file, "w") as f:
        f.write(generated_text.strip())

    print(f"Categorized list saved to '{output_file}'")
except Exception as e:
    print(f"Error: {e}")
