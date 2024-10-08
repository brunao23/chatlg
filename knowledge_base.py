import streamlit as st
import openai
from docx import Document
import glob
import os
import json
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Função para extrair texto de um arquivo DOCX
def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return " ".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip() != ""])
    except Exception as e:
        st.error(f"Erro ao ler o arquivo {file_path}: {e}")
        return ""

# Função para extrair texto de um arquivo TXT
def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        st.error(f"Erro ao ler o arquivo {file_path}: {e}")
        return ""

# Função para extrair texto de um arquivo PDF
def extract_text_from_pdf(file_path):
    try:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Erro ao ler o arquivo {file_path}: {e}")
        return ""

# Função para extrair texto de um arquivo CSV
def extract_text_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.to_string()
    except Exception as e:
        st.error(f"Erro ao ler o arquivo {file_path}: {e}")
        return ""

# Função para carregar toda a base de conhecimento em um dicionário organizado
@st.cache_data
def load_knowledge_base():
    knowledge_base = {}
    
    if not os.path.exists("knowledge_base"):
        st.error("O diretório 'knowledge_base' não foi encontrado. Por favor, crie o diretório e adicione arquivos.")
        return knowledge_base

    # Carregar arquivos DOCX
    for file_path in glob.glob("knowledge_base/*.docx"):
        file_name = os.path.basename(file_path)
        content = extract_text_from_docx(file_path)
        if content:
            knowledge_base[file_name] = content

    # Carregar arquivos TXT
    for file_path in glob.glob("knowledge_base/*.txt"):
        file_name = os.path.basename(file_path)
        content = extract_text_from_txt(file_path)
        if content:
            knowledge_base[file_name] = content

    # Carregar arquivos PDF
    for file_path in glob.glob("knowledge_base/*.pdf"):
        file_name = os.path.basename(file_path)
        content = extract_text_from_pdf(file_path)
        if content:
            knowledge_base[file_name] = content

    # Carregar arquivos CSV
    for file_path in glob.glob("knowledge_base/*.csv"):
        file_name = os.path.basename(file_path)
        content = extract_text_from_csv(file_path)
        if content:
            knowledge_base[file_name] = content

    # Verificar se algum conteúdo foi carregado
    if not knowledge_base:
        st.error("Nenhum conteúdo válido foi carregado dos arquivos fornecidos. Verifique os arquivos e tente novamente.")
    else:
        st.success(f"{len(knowledge_base)} arquivos carregados com sucesso na base de conhecimento.")

    return knowledge_base

# Função para encontrar as seções mais relevantes usando TF-IDF e similaridade coseno
def get_most_relevant_sections(query, knowledge_base, top_n=3):
    documents = list(knowledge_base.values())
    doc_names = list(knowledge_base.keys())

    if not documents:
        return "Base de conhecimento vazia ou não carregada corretamente."

    # Vetorização da consulta e documentos
    vectorizer = TfidfVectorizer().fit_transform([query] + documents)
    query_vec = vectorizer[0]
    doc_vecs = vectorizer[1:]

    # Calcular a similaridade entre a consulta e cada documento
    similarities = cosine_similarity(query_vec, doc_vecs).flatten()
    relevant_indices = similarities.argsort()[-top_n:][::-1]

    # Selecionar os documentos mais relevantes
    most_relevant_sections = [documents[i] for i in relevant_indices]
    relevant_names = [doc_names[i] for i in relevant_indices]

    # Concatenar os textos mais relevantes
    result = "\n\n".join([f"{relevant_names[i]}: {most_relevant_sections[i]}" for i in range(len(most_relevant_sections))])
    return result

# Função para buscar informações na base de conhecimento
def search_knowledge_base(query):
    knowledge_base = load_knowledge_base()
    if not knowledge_base:
        return "Base de conhecimento vazia ou não carregada corretamente."

    relevant_content = get_most_relevant_sections(query, knowledge_base)
    
    # Usar a API da OpenAI para encontrar a resposta mais relevante
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Você é um assistente que responde perguntas com base no conteúdo fornecido. Se a informação não estiver disponível na base de conhecimento, diga que não tem essa informação."},
                {"role": "user", "content": f"Com base no seguinte conteúdo, responda à pergunta: '{query}'\n\nConteúdo: {relevant_content[:4000]}"}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Erro ao se comunicar com a API OpenAI: {e}"

