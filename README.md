# End-to-End-RAG-using-LangChain

This repository contains an implementation of a Retrieval-Augmented Generation (RAG) system using LangChain and FAISS, with a focus on querying PDF documents. The system utilizes Google's Generative AI Embeddings and integrates with Streamlit for a user-friendly interface.

## Features

- Upload and process multiple PDF documents.
- Generate embeddings using Google's Generative AI.
- Perform similarity search with FAISS.
- Interactive querying through a Streamlit interface.

## Setup Instructions

1. Clone the repository

```
git clone https://github.com/Niharrrrrr/End-to-End-RAG-using-LangChain.git
cd End-to-End-RAG-using-LangChain
```

2. Set up a virtual environment

```
conda create --name rag-env python=3.8
conda activate rag-env
```

3.Install the dependencies

```
pip install -r requirements.txt
```

