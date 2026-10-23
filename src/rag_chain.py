# rag_chain.py
"""
Step 3: Builds the RAG pipeline with an OpenAI-compatible LLM, conversation memory and chapter citations.
"""
import os
import re
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from retriever import get_retriever

BASE_DIR = Path(__file__).resolve().parent.parent

# The .env file lives in the project root (works no matter where you run the script from)
load_dotenv(BASE_DIR / ".env")

token = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_TOKEN")
endpoint = os.getenv("BASE_URL")
model_name = os.getenv("MODEL_NAME")  # the LLM is always taken from your .env file

# Number of passages sent to the LLM. If your free API plan hits its token limit often,
# add RETRIEVE_K=6 to your .env (fewer passages = fewer tokens, but slightly less context).
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "10"))

# Set RAG_DEBUG=1 in your .env to print the retrieved chunks in the terminal
DEBUG = os.getenv("RAG_DEBUG", "0") == "1"

if not token:
    raise ValueError("API key not found. Please add it to the .env file in the project root.")

if not model_name:
    raise ValueError("MODEL_NAME not found. Please add MODEL_NAME=<your model> to the .env file.")

# LLM that writes the final answer (low temperature = more factual)
llm = ChatOpenAI(
    model=model_name,
    api_key=token,
    base_url=endpoint,
    temperature=0,  # 0 = deterministic: the same question gives the same answer every time
    max_retries=5,  # on a rate-limit (429) error, wait and retry automatically before giving up
    timeout=90,
)

# The exact sentence the LLM must reply with when the answer is not in the book
NOT_FOUND_MSG = "দুঃখিত, আপনার এই প্রশ্নের উত্তর 'মেজদিদি' বইটিতে পাওয়া যায়নি।"

# Shown when the LLM returns an empty response (this is a model/API problem, not a "not in the book" case)
EMPTY_ANSWER_MSG = "দুঃখিত, মডেল এবার কোনো উত্তর তৈরি করতে পারেনি। অনুগ্রহ করে প্রশ্নটি আবার করুন।"

SYSTEM_PROMPT = """You are a helpful literary AI assistant for the Bengali book 'মেজদিদি' by শরৎচন্দ্র চট্টোপাধ্যায়.
Your task is to answer the user's question in Bengali based ONLY on the provided context below.

Core Rules:
1. Every answer must come ONLY from the provided Context. Do not make up or assume any outside information, and do not use your own knowledge about the book.
2. Write your answer clearly in Bengali. Keep it short and direct (1 to 4 sentences) unless the user asks for details. Do not add your own interpretation, comments in brackets, or explanations about how characters are related unless the Context states it explicitly.
3. The book is written in old literary Bengali (e.g. "করিলেন", "খাইয়া", "বলিল"), while the user usually asks in modern Bengali (e.g. "করলেন", "খেয়ে", "বলল"). Match by MEANING, not by exact words. Character names may appear in different forms (e.g. কেষ্ট, কেষ্টর, কেষ্টকে); treat them as the same person.
4. If the Context contains the answer or even a part of it, answer with the information that is available. You may combine details from several passages (for example, relationships between characters). If only part of the question can be answered, say which part is not found in the book.
5. At the end of every valid answer, you MUST cite the book name and the chapter name exactly as shown in the Context headers, in this format:
   (সূত্র: মেজদিদি, পরিচ্ছেদ: তিন)
   Cite only the chapter(s) of the passages you actually used. If the information comes from more than one chapter, cite all of them. Questions about the book's name or author can be answered from the "বইয়ের পরিচিতি" entry.
6. Never guess names, ages, numbers or family relationships. A name or relationship must be stated explicitly in the Context. Do not assume that a character is someone's son, brother or sister just because both appear in the same passage, or because they have similar names. If the specific fact is not explicitly stated, treat the answer as not found.
7. ONLY if the Context contains nothing relevant to the question, or the question is outside the scope of the book, do NOT guess and do NOT add a citation. Reply with exactly this sentence and nothing else: "দুঃখিত, আপনার এই প্রশ্নের উত্তর 'মেজদিদি' বইটিতে পাওয়া যায়নি।"

Context:
{context}
"""


def is_not_found(answer: str) -> bool:
    """Returns True if the answer is the 'not found in the book' reply."""
    return "পাওয়া যায়নি" in answer


def format_docs(docs: list[Document]) -> str:
    """Formats retrieved chunks together with their book name, chapter, section and URL."""
    formatted_chunks = []
    for doc in docs:
        book = doc.metadata.get("book_name", "মেজদিদি")
        chapter = doc.metadata.get("chapter_name", "Unknown Chapter")
        section = doc.metadata.get("section_name", "")
        url = doc.metadata.get("source_url", "")
        formatted_chunks.append(
            f"[Book: {book} | Chapter: {chapter} | Section: {section} | URL: {url}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted_chunks)


def _cited_in_answer(chapter_name: str, answer: str) -> bool:
    """Checks whether the LLM actually mentioned this chapter in its answer."""
    answer = unicodedata.normalize("NFC", answer)
    name = unicodedata.normalize("NFC", chapter_name)
    if name.startswith("পরিচ্ছেদ") and ":" in name:
        word = name.split(":", 1)[1].strip()
        # e.g. "পরিচ্ছেদ: এক" must not match the ordinary word "একটি"
        pattern = r"পরিচ্ছেদ\s*[:：]?\s*" + re.escape(word) + r"(?![\u0980-\u09FF])"
        return re.search(pattern, answer) is not None
    return name in answer


def extract_unique_citations(docs: list[Document], answer: str = "") -> list[dict]:
    """Extracts unique chapter citations from the retrieved documents.

    If the answer text names specific chapters, only those chapters are returned
    (so the UI does not show chapters the LLM never used). If none can be matched,
    all retrieved chapters are returned as a fallback.
    """
    seen = set()
    citations = []
    for doc in docs:
        chapter = doc.metadata.get("chapter_name", "Unknown Chapter")
        if chapter not in seen:
            seen.add(chapter)
            citations.append({
                "book_name": doc.metadata.get("book_name", "মেজদিদি"),
                "chapter_name": chapter,
                "section_name": doc.metadata.get("section_name", ""),
                "source_url": doc.metadata.get("source_url", ""),
            })

    cited = [c for c in citations if _cited_in_answer(c["chapter_name"], answer)]
    return cited or citations


def build_rag_chain():
    """Creates the RAG pipeline by combining the retriever and the LLM.

    Every question is answered independently (no chat memory), so each answer comes ONLY from the
    passages retrieved from the book for that question, never from earlier answers.
    """
    retriever = get_retriever(k=RETRIEVE_K)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    answering_chain = prompt | llm | StrOutputParser()

    def ask_book(inputs: dict) -> dict:
        question = inputs["question"]

        # 1. Retrieve the top-k most relevant chunks from ChromaDB
        docs = retriever.invoke(question)

        if DEBUG:
            print(f"\n[DEBUG] Retrieved {len(docs)} chunks for question: {question}")
            for d in docs:
                print(f"[DEBUG] {d.metadata.get('chapter_name')} | {d.page_content[:70]}...")

        # 2. Format the context and generate the answer with the LLM
        context_str = format_docs(docs)
        answer = ""
        for attempt in range(3):  # some LLMs occasionally return an empty response, so retry
            answer = answering_chain.invoke({
                "context": context_str,
                "question": question,
            }).strip()
            if answer:
                break
            if DEBUG:
                print(f"[DEBUG] LLM returned an empty answer (attempt {attempt + 1}/3), retrying...")

        if not answer:
            return {
                "answer": EMPTY_ANSWER_MSG,
                "found": False,
                "error": True,  # the LLM failed to respond (not a "not in the book" case)
                "citations": [],
                "retrieved_docs": docs,
            }

        # 3. Return the answer with citations (no citations when the answer was not found)
        found = not is_not_found(answer)
        if not found:
            answer = NOT_FOUND_MSG  # always return the exact, untruncated sentence
        citations = extract_unique_citations(docs, answer) if found else []
        return {
            "answer": answer,
            "found": found,
            "error": False,
            "citations": citations,
            "retrieved_docs": docs,
        }

    return ask_book