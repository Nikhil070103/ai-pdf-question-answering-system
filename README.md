# AI PDF Question Answering System

This project is a Generative AI application that allows users to ask questions from a PDF document.

## Technologies Used
- Python
- LangChain
- FAISS
- OpenAI / LLM
- Streamlit

## Features
- Upload PDF
- Extract text
- Create embeddings
- Store vectors using FAISS
- Retrieve relevant context
- Generate answers using LLM

## Project Architecture

User Question → Retriever → FAISS Vector DB → Relevant Text → LLM → Answer

## Run the Project

Install dependencies:

pip install -r requirements.txt

Run application:

streamlit run app.py


## Application Screenshot

![App Screenshot](screenshots/app_ui.png)
