import tiktoken


def main():
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    print(enc, "enc")

    texts = [
        "Hello world",
        "Transformer",
        "人工智能",
        "大语言模型",
        "中文比英文消耗更多Token吗？",
    ]

    for text in texts:
        token = enc.encode(text)
        print(f"Text: {text}")
        print(f"Tokens: {token}")
        print(f"Number of tokens: {len(token)}")
        print("-" * 20)


if __name__ == "__main__":
    main()
