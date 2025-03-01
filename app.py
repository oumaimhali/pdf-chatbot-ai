import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
import tempfile
import os
import hashlib
from dotenv import load_dotenv
import requests
import base64
import json

# Charger les variables d'environnement
load_dotenv()

# Configurer les clés API
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

class PDFChatbot:
    def __init__(self, pdf_path=None, pdf_content=None, pdf_name=None):
        self.pdf_path = pdf_path
        self.pdf_content = pdf_content
        self.pdf_name = pdf_name
        self.vectorstore = None
        self.conversation = None
        self.chat_id = None
        self.github_repo = None
        self.app_url = None

    def generate_chat_id(self):
        """Génère un ID unique pour le chat basé sur le nom du PDF."""
        if self.pdf_name:
            # Créer un hash unique basé sur le nom du fichier
            hash_object = hashlib.md5(self.pdf_name.encode())
            self.chat_id = hash_object.hexdigest()[:8]
            return self.chat_id
        return None

    def create_github_repo(self):
        """Crée un nouveau dépôt GitHub pour ce chatbot."""
        headers = {
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        # Créer un nom de repo unique basé sur le nom du PDF
        repo_name = f"pdf-chat-{self.chat_id}"
        
        # Créer le repo
        response = requests.post(
            'https://api.github.com/user/repos',
            headers=headers,
            json={
                'name': repo_name,
                'private': False,
                'auto_init': True
            }
        )
        
        if response.status_code == 201:
            self.github_repo = response.json()['html_url']
            return True
        return False

    def push_code_to_github(self):
        """Pousse le code du chatbot vers le nouveau dépôt."""
        headers = {
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json'
        }

        # Créer app.py spécifique pour ce PDF
        app_code = f'''
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# Configuration
st.title("Chat avec {self.pdf_name}")

# Initialisation
@st.cache_resource
def init_chatbot():
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.load_local("vectorstore")
    llm = ChatOpenAI(temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return conversation

# Interface de chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

conversation = init_chatbot()

question = st.text_input("Pose ta question sur le PDF :")
if question:
    response = conversation({{"question": question, "chat_history": st.session_state.chat_history}})
    st.session_state.chat_history.append({{"question": question, "answer": response["answer"]}})
    
    for chat in st.session_state.chat_history:
        st.markdown(f"**Q:** {{chat['question']}}")
        st.markdown(f"**R:** {{chat['answer']}}")
        st.markdown("---")
'''

        # Encoder le contenu en base64
        content = base64.b64encode(app_code.encode()).decode()

        # Créer app.py dans le nouveau repo
        repo_name = self.github_repo.split('/')[-1]
        owner = self.github_repo.split('/')[-2]
        
        response = requests.put(
            f'https://api.github.com/repos/{owner}/{repo_name}/contents/app.py',
            headers=headers,
            json={
                'message': 'Initial commit with chatbot code',
                'content': content
            }
        )

        # Sauvegarder le vectorstore
        self.vectorstore.save_local("vectorstore")
        
        # TODO: Push le vectorstore vers GitHub
        
        if response.status_code in [201, 200]:
            return True
        return False

    def process_pdf(self):
        """Traite le fichier PDF et crée une nouvelle application."""
        if self.pdf_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(self.pdf_content)
                self.pdf_path = tmp_file.name

        # Générer un ID unique
        self.generate_chat_id()

        # Créer les embeddings
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        docs = text_splitter.split_documents(pages)
        
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.vectorstore = FAISS.from_documents(docs, embeddings)

        # Créer un nouveau dépôt GitHub
        if self.create_github_repo():
            if self.push_code_to_github():
                # Générer l'URL de l'application Streamlit
                repo_name = self.github_repo.split('/')[-1]
                self.app_url = f"https://{repo_name}.streamlit.app"
                return True

        return False

    def get_response(self, question, chat_history=[]):
        """Obtient une réponse à une question."""
        if self.conversation:
            response = self.conversation({"question": question, "chat_history": chat_history})
            return response["answer"]
        return "Veuillez d'abord charger un document PDF."

def main():
    st.title("PDF Chatbot Creator")
    
    st.markdown("""
    ## Comment ça marche
    1. Uploadez votre PDF
    2. Cliquez sur "Créer Chatbot"
    3. Obtenez un lien vers votre chatbot dédié
    """)

    uploaded_file = st.file_uploader("Choisis un fichier PDF", type="pdf")
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"PDF sélectionné : {uploaded_file.name}")
        with col2:
            if st.button("Créer Chatbot"):
                with st.spinner("Création de votre chatbot en cours..."):
                    chatbot = PDFChatbot(pdf_content=uploaded_file.getvalue(), pdf_name=uploaded_file.name)
                    if chatbot.process_pdf():
                        st.success("Chatbot créé avec succès!")
                        st.markdown("## Votre Chatbot")
                        st.markdown(f"Voici le lien vers votre chatbot dédié : [{chatbot.app_url}]({chatbot.app_url})")
                        st.markdown("""
                        **Note** : Le déploiement peut prendre quelques minutes. 
                        Si le lien ne fonctionne pas immédiatement, attendez un peu et réessayez.
                        """)
                    else:
                        st.error("Une erreur est survenue lors de la création du chatbot.")

if __name__ == '__main__':
    main()
