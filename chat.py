#!/usr/bin/env python3
"""
NASA mission chat: a Streamlit RAG prototype.

Answers questions from embedded NASA mission documents and scores each answer
with RAGAS (faithfulness and answer relevancy) in the sidebar.
"""

import math
import os
from typing import Dict, List, Optional

import streamlit as st

import config
import llm_client
import rag_client
import ragas_evaluator

st.set_page_config(
    page_title="NASA RAG Chat with Evaluation",
    page_icon="🚀",
    layout="wide"
)


def retrieve_documents(openai_api_key: str, collection, query: str, n_results: int = 3,
                       mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant chunks, showing an error in the UI instead of crashing"""
    try:
        return rag_client.retrieve_documents(openai_api_key, collection, query, n_results, mission_filter)
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return None


def generate_response(openai_key: str, user_message: str, context: str,
                      conversation_history: List[Dict], model: str = config.CHAT_MODELS[0]) -> str:
    """Generate an answer, returning the error text if the API call fails"""
    try:
        return llm_client.generate_response(openai_key, user_message, context, conversation_history, model)
    except Exception as e:
        return f"Error generating response: {e}"


def evaluate_response_quality(openai_key: str, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Score the answer with RAGAS, returning {"error": ...} if evaluation fails"""
    try:
        return ragas_evaluator.evaluate_response_quality(openai_key, question, answer, contexts)
    except Exception as e:
        return {"error": f"Evaluation failed: {e}"}


def display_evaluation_metrics(scores: Dict[str, float]):
    """Display evaluation metrics in the sidebar"""
    if "error" in scores:
        st.sidebar.error(f"Evaluation Error: {scores['error']}")
        return

    st.sidebar.subheader("📊 Response Quality")

    for metric_name, score in scores.items():
        label = metric_name.replace('_', ' ').title()
        # RAGAS returns NaN when a metric could not be computed
        if not isinstance(score, (int, float)) or math.isnan(score):
            st.sidebar.metric(label=label, value="n/a")
            continue

        st.sidebar.metric(label=label, value=f"{score:.3f}")
        st.sidebar.progress(min(max(score, 0.0), 1.0))


def main():
    st.title("🚀 NASA Space Mission Chat with Evaluation")
    st.markdown("Chat with AI about NASA space missions with real-time quality evaluation")

    if not ragas_evaluator.RAGAS_AVAILABLE:
        st.warning("RAGAS not available. Install with: pip install ragas")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []

    with st.sidebar:
        st.header("🔧 Configuration")

        with st.spinner("Discovering ChromaDB backends..."):
            available_backends = rag_client.discover_chroma_backends()

        if not available_backends:
            st.error("No ChromaDB backends found!")
            st.info("Build one first:\n`python embedding_pipeline.py`")
            st.stop()

        st.subheader("📊 ChromaDB Backend")
        selected_backend_key = st.selectbox(
            "Select Document Collection",
            options=list(available_backends.keys()),
            format_func=lambda key: available_backends[key]["display_name"],
            help="Choose which document collection to use for retrieval"
        )
        selected_backend = available_backends[selected_backend_key]

        st.subheader("🔑 OpenAI Settings")
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )
        if not openai_key:
            st.warning("Please enter your OpenAI API key")
            st.stop()

        model_choice = st.selectbox(
            "OpenAI Model",
            options=config.CHAT_MODELS,
            help="Choose the OpenAI model for responses"
        )

        st.subheader("🔍 Retrieval Settings")
        n_docs = st.slider("Documents to retrieve", 1, 10, 3)

        st.subheader("📊 Evaluation Settings")
        enable_evaluation = st.checkbox("Enable RAGAS Evaluation", value=ragas_evaluator.RAGAS_AVAILABLE)

    with st.spinner("Initializing RAG system..."):
        collection, success, error = rag_client.initialize_rag_system(
            selected_backend["directory"],
            selected_backend["collection_name"]
        )
    if not success:
        st.error(f"Failed to initialize RAG system: {error}")
        st.stop()

    if st.session_state.last_evaluation and enable_evaluation:
        display_evaluation_metrics(st.session_state.last_evaluation)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about NASA space missions..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating response..."):
                docs_result = retrieve_documents(openai_key, collection, prompt, n_docs)

                context = ""
                contexts_list = []
                if docs_result and docs_result.get("documents"):
                    contexts_list = docs_result["documents"][0]
                    context = rag_client.format_context(contexts_list, docs_result["metadatas"][0])
                    st.session_state.last_contexts = contexts_list

                response = generate_response(
                    openai_key,
                    prompt,
                    context,
                    st.session_state.messages[:-1],
                    model_choice
                )
                st.markdown(response)

                if enable_evaluation and ragas_evaluator.RAGAS_AVAILABLE:
                    with st.spinner("Evaluating response quality..."):
                        st.session_state.last_evaluation = evaluate_response_quality(
                            openai_key, prompt, response, contexts_list
                        )

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
