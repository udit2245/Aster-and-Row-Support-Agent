from src.agent.agent import run_agent


def main():

    print("=" * 70)
    print("ASTER & ROW AI SUPPORT AGENT")
    print("=" * 70)

    print(
        "\nKnowledge base loaded from ChromaDB."
    )

    print(
        "Type 'exit' or 'quit' to stop."
    )

    history = []

    while True:

        question = input(
            "\nAsk a question: "
        ).strip()

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit"
        }:
            print("\nGoodbye!")
            break

        print(
            "\nRunning agent..."
        )

        try:

            answer = run_agent(
                question,
                history
            )

            print(
                "\n" + "=" * 70
            )

            print(
                "ANSWER"
            )

            print(
                "=" * 70
            )

            print(answer)

            history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as error:

            print(
                "\nAn error occurred:"
            )

            print(
                f"{type(error).__name__}: {error}"
            )


if __name__ == "__main__":
    main()