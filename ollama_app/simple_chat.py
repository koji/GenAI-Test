import ollama

model = "gemma2:9b"


def main():
    # Ask the user for input in a loop
    while True:
        user_input = input("You: ")

        # Exit the loop if the user types 'exit'
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Call the Ollama API with the user's input
        res = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": user_input},
            ],
            stream=True,
        )

        # Print the response as it streams
        print("Bot: ", end="", flush=True)
        for chunk in res:
            print(chunk["message"]["content"], end="", flush=True)
        print()  # Add a newline after the response


if __name__ == "__main__":
    main()
