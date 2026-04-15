import ollama

try:
    response = ollama.generate(
        model="lfm2.5-thinking",
        prompt="How much wood did a woodchuck chuck if a woodchuck could chuck wood?",
        system="You are a rude but accurate assistant.",
        options={"temperature": 0.9,
                    "top_p": 0.9,
                    "top_k": 40
                    },
        stream=True
    )

    for i in response:
        print(i["response"], end="")

except Exception as e:
    print(f"An error occurred: {e}")