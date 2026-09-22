
"""
RFID chatbot agent — single tool, grounded in a Vertex AI RAG Engine corpus
(your own RAG: custom chunking, embeddings, top_k + semantic reranking).
Deployed to Agent Engine (no Docker/Cloud Run/Streamlit needed).
"""

import os

import vertexai
from vertexai import rag

from google.adk.agents import Agent

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ID = os.environ.get("PROJECT_ID", "rta-genai-explorations-406d")
REGION = os.environ.get("REGION", "europe-west1")
AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-2.5-pro")

# Resource name of the RAG Engine corpus you built with rag.create_corpus(),
# e.g. "projects/<num>/locations/europe-west1/ragCorpora/<id>"
RAG_CORPUS_NAME = os.environ.get("RAG_CORPUS_NAME", "")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", PROJECT_ID)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", REGION)
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")

vertexai.init(project=PROJECT_ID, location=REGION)


# ─────────────────────────────────────────────────────────────────────────────
# Tool
# ─────────────────────────────────────────────────────────────────────────────
def ppt_knowledge_retriever(query: str) -> str:
    """
    Searches the RFID knowledge base — a Vertex AI RAG Engine corpus built
    from documents in gs://asad_enver/RFID TRAINING/ — to retrieve precise
    facts, definitions, and process details.

    Args:
        query: The user's question, in natural language.

    Returns:
        Relevant excerpts from the indexed documents, with source file names.
    """
    if not RAG_CORPUS_NAME:
        return "❌ RAG corpus is not configured (RAG_CORPUS_NAME env var missing)."

    try:
        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=RAG_TOP_K,
            ranking=rag.Ranking(
                rank_service=rag.RankService(
                    model_name="semantic-ranker-default@latest"
                )
            ),
        )

        response = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=RAG_CORPUS_NAME)],
            text=query,
            rag_retrieval_config=rag_retrieval_config,
        )

        contexts = response.contexts.contexts
        if not contexts:
            return "No relevant information found in the documentation."

        parts = []
        for i, ctx in enumerate(contexts, start=1):
            source = ctx.source_uri or "unknown source"
            parts.append(f"**[{i}] Source: {source}**\n{ctx.text.strip()}")

        return "\n\n---\n\n".join(parts)

    except Exception as e:
        return f"❌ Error retrieving content from RAG corpus: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are the RFID Knowledge Assistant. You answer questions about RFID
concepts, processes, and definitions using ONLY the ppt_knowledge_retriever
tool

Rules:
1. Always call ppt_knowledge_retriever for any factual question - never
   answer from your own general knowledge.
2. If the tool returns no relevant information, say so plainly rather than
   guessing or fabricating an answer.
3. Present answers in plain, clear English. Keep the tool's grounded content
   as the basis for your response.
4. Do not use Google Search or any other information to answer any query, ALWAYS call
   the ppt_knowledge_retriever tool.
5. Give the source of the documents or chunks you used to answer the question.
6. TRY TO KEEP ANSWERS SHORT, PRECISE AND TO THE POINT. OF COURSE YOU SHOULD GO INTO DETAILS BUT TRY TO SUMMARIZE BEFORE ANSWERING
"""

root_agent = Agent(
    model=AGENT_MODEL,
    name="RFIDKnowledgeAssistant",
    description="Answers questions about RFID material using a custom Vertex AI RAG Engine corpus",
    instruction=SYSTEM_PROMPT,
    tools=[ppt_knowledge_retriever],
)