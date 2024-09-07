import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import VectorStoreIndex, Document, SimpleDirectoryReader
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from io import StringIO
import os
import PyPDF2  # Biblioteca para processar PDFs
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print(torch.cuda.is_available())  # Deve retornar True se CUDA estiver funcionando

# Configuração da página
st.set_page_config(page_title="Converse sobre os seus documentos", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)

# Carregar o modelo LLaMA 2
@st.cache_resource
def load_llama2_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # LLaMA 2 7B modelo da Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
    return model, tokenizer

# Carregar modelo de embeddings HuggingFace compatível com LlamaIndex
@st.cache_resource
def load_embed_model():
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embed_model

model, tokenizer = load_llama2_model()
embed_model = load_embed_model()

st.title("Converse sobre os seus documentos 💬🦙")

# Configuração do banco de dados usando SQLAlchemy
def get_db_connection():
    db_url = st.secrets["connections"]["dev_db"]["url"]
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    return Session()

session = get_db_connection()

# Função para salvar documentos no banco de dados
def save_to_db(session, filename, content):
    session.execute(
        text("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, content TEXT)")
    )
    session.execute(
        text("INSERT INTO documents (filename, content) VALUES (:filename, :content)"),
        {"filename": filename, "content": content}
    )
    session.commit()

# Função para carregar documentos do banco de dados
def load_documents_from_db(session):
    result = session.execute(text("SELECT content FROM documents")).fetchall()
    return [row[0] for row in result]

# Função para processar o upload de arquivos (suporte a PDFs)
def handle_file_upload(uploaded_file):
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        if file_extension == ".txt":
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            file_content = stringio.read()
        elif file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            file_content = ""
            for page in pdf_reader.pages:
                file_content += page.extract_text()
        else:
            st.error("Somente arquivos .txt e .pdf são suportados.")
            return None
        
        save_to_db(session, uploaded_file.name, file_content)
        st.success(f"Arquivo '{uploaded_file.name}' foi carregado e salvo com sucesso!")
        
        # Atualizar e carregar o índice imediatamente após o upload
        index = load_and_index_documents(session)
        st.session_state.index = index  # Salvar o índice no estado da sessão para ser usado no chat
        return file_content
    return None

# Função para carregar e indexar os documentos usando embeddings locais
def load_and_index_documents(session):
    documents = load_documents_from_db(session)
    if documents:
        indexed_docs = [Document(text=doc) for doc in documents]
        index = VectorStoreIndex.from_documents(indexed_docs, embed_model=embed_model)
        return index
    else:
        st.warning("Nenhum documento encontrado no banco de dados.")
        return None

# Função para gerar resposta usando o LLaMA 2 localmente
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# ======= Seção de Upload de Documentos =======
st.header("Carregue seus documentos (suporte a .txt e .pdf)")
uploaded_file = st.file_uploader("Faça o upload (.txt ou .pdf)", type=["txt", "pdf"])

if uploaded_file:
    handle_file_upload(uploaded_file)

# ======= Chatbot Original =======
# Carregar e indexar os documentos
if "index" not in st.session_state:
    st.session_state.index = load_and_index_documents(session)

if "messages" not in st.session_state:  # Inicializar histórico de mensagens
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Por favor, faça o upload de um documento e pergunte algo sobre ele.",
        }
    ]

# Interface de entrada do chat
if prompt := st.chat_input("Faça uma pergunta sobre o documento carregado"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Exibir o histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Se a última mensagem não for do assistente, gerar uma nova resposta usando LLaMA 2
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        prompt = st.session_state.messages[-1]["content"]
        response = generate_response(prompt)
        st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
