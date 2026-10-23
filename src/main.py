# main.py
"""
Step 4: CLI chatbot with source citations.
Run from the 'src' folder:  python main.py
"""
import os
import urllib.parse

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from rag_chain import build_rag_chain, model_name


def main():
    print("=======================================================")
    print("        Bangla Book RAG Chatbot — 'মেজদিদি'             ")
    print("   Ask questions in Bangla about the selected book.    ")
    print("   Type 'quit' to exit.                                ")
    print("=======================================================")
    print(f"LLM in use: {model_name}\n")

    ask_book = build_rag_chain()

    while True:
        question = input("You: ").strip()
        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print("\nAssistant: ")
        result = ask_book({"question": question})

        print(result["answer"])

        # Show sources only when the answer was found in the book
        if result["found"] and result["citations"]:
            print("\nRetrieved Sources (Citations):")
            for c in result["citations"]:
                url = urllib.parse.unquote(c["source_url"])
                print(f" - {c['book_name']} | {c['chapter_name']} | URL: {url}")

        print("\n" + "-" * 55 + "\n")


if __name__ == "__main__":
    main()