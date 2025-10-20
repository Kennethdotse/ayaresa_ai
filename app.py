import os
import sys
import gradio as gr
import torch
from transformers import pipeline, BitsAndBytesConfig
from datasets import load_dataset
import pandas as pd
from PIL import Image
from typing import Optional
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.dataframe import DataFrameLoader
from langchain_text_splitters import CharacterTextSplitter

MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "4b-it")
MODEL_ID = f"google/medgemma-{MODEL_VARIANT}"
USE_QUANTIZATION = True
LOCAL_DOCS_PATH = Path("./hb_db")
CHROMA_PERSIST_DIR = "./chroma_db"

_pipe = None
_rag_vectorstore = None
_embeddings = None

def _init_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if USE_QUANTIZATION:
        try:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except Exception:
            pass

    task = "image-text-to-text" if "image" in MODEL_VARIANT or "it" in MODEL_VARIANT else "text-generation"

    print(f"Initializing pipeline: {MODEL_ID} task={task}")
    _pipe = pipeline(
        task,
        model=MODEL_ID,
        device_map=model_kwargs.get("device_map"),
        torch_dtype=model_kwargs.get("torch_dtype"),
       **({} if "quantization_config" not in model_kwargs else {"quantization_config": model_kwargs["quantization_config"]}),
   )
    try:
        _pipe.model.generation_config.do_sample = False
    except Exception:
        pass

    return _pipe

def _init_rag():
    global _rag_vectorstore, _embeddings
    if _rag_vectorstore is not None:
        return _rag_vectorstore

    docs = []

    try:
        ds = load_dataset("knowrohit07/know_medical_dialogue_v2")
        df = pd.DataFrame(ds["train"])
        if "instruction" in df.columns and "output" in df.columns:
            df["full_dialogue"] = df["instruction"].astype(str) + " \n\n" + df["output"].astype(str)
            loader = DataFrameLoader(df, page_content_column="full_dialogue")
            docs += loader.load()
    except Exception as e:
        print("Warning: could not load HF dataset:", e)

    csv_path = LOCAL_DOCS_PATH / "Final_Dataset.csv"
    if csv_path.exists():
        try:
            csv_loader = CSVLoader(str(csv_path))
            docs += csv_loader.load()
        except Exception as e:
            print("Warning loading CSV:", e)

    if LOCAL_DOCS_PATH.exists() and LOCAL_DOCS_PATH.is_dir():
        for pdf_file in LOCAL_DOCS_PATH.glob("*.pdf"):
            try:
                pdf_loader = PyPDFLoader(str(pdf_file))
                docs += pdf_loader.load()
            except Exception as e:
                print(f"Warning loading PDF {pdf_file}: {e}")

    if len(docs) == 0:
        from langchain.schema import Document
        docs = [Document(page_content="No local documents found. Upload PDFs/CSV into ./hb_db or commit them to the Space repo.")]

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    try:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _rag_vectorstore = Chroma.from_documents(chunks, _embeddings, persist_directory=CHROMA_PERSIST_DIR)
        try:
            _rag_vectorstore.persist()
        except Exception:
            pass
    except Exception as e:
        print("Error initializing vectorstore:", e)
        _rag_vectorstore = None

    return _rag_vectorstore

def generate_medgemma_rag_response(query: str, image: Optional[Image.Image] = None) -> str:
    vs = _init_rag()

    context = ""
    if vs is not None:
        try:
            retrieved = vs.similarity_search(query, k=4)
            context = "\n\n".join([d.page_content for d in retrieved])
        except Exception as e:
            print("Warning during similarity search:", e)

    rag_prompt = f"""
    [System]: You are a respectful and knowledgeable medical AI assistant. 
    Use the provided context to answer clearly and do not include system or context in your final response.

    [Context]: {context}

    [User]: {query}

    [Assistant]:
    """

    pipe = _init_pipeline()

    if image is not None:
        input_for_pipe = {"image": image, "text": rag_prompt}
        try:
            out = pipe(input_for_pipe, max_new_tokens=512)
        except Exception:
            out = pipe(rag_prompt, max_new_tokens=512)
    else:
        out = pipe(rag_prompt, max_new_tokens=512)

    try:
        if isinstance(out, list) and len(out) > 0:
            if isinstance(out[0], dict):
                text = out[0].get("generated_text") or out[0].get("text") or str(out[0])
            else:
                text = str(out[0])
        else:
            text = str(out)
    except Exception:
        text = str(out)

    return text

with gr.Blocks() as iface:
    chatbot = gr.Chatbot(label="Ayaresa chat")
    with gr.Row():
        with gr.Column(scale=3):
            txt = gr.Textbox(label="Enter a prompt", placeholder="Type your question here...", lines=2)
        with gr.Column(scale=1):
            img = gr.Image(type="pil", label="Image (optional)")
    with gr.Row():
        send = gr.Button("Send")
        clear = gr.Button("Clear")

    state = gr.State([])

    def submit_fn(message, image, history):
        history = history or []
        if (not message or message.strip() == "") and image is None:
            return history, "", history
        resp = generate_medgemma_rag_response(message or "", image)
        history.append((message or "", resp))
        return history, "", history

    send.click(submit_fn, inputs=[txt, img, state], outputs=[chatbot, txt, state])
    txt.submit(submit_fn, inputs=[txt, img, state], outputs=[chatbot, txt, state])
    clear.click(lambda: ([], "", []), inputs=None, outputs=[chatbot, txt, state])

if __name__ == "__main__":
    iface.launch()