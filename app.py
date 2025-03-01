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

class PDFChatbotCreator:
    def __init__(self, pdf_content, pdf_name):
        self.pdf_content = pdf_content
        self.pdf_name = pdf_name
        self.chat_id = self._generate_chat_id()
        self.repo_name = f"pdf-chat-{self.chat_id}"
        self.vectorstore = None

    def _generate_chat_id(self):
        """Génère un ID unique pour le chat."""
        hash_object = hashlib.md5(self.pdf_name.encode())
        return hash_object.hexdigest()[:8]

    def _create_github_repo(self):
        """Crée un nouveau dépôt GitHub."""
        headers = {
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        response = requests.post(
            'https://api.github.com/user/repos',
            headers=headers,
            json={
                'name': self.repo_name,
                'private': False,
                'auto_init': True
            }
        )
        
        if response.status_code != 201:
            raise Exception("Erreur lors de la création du dépôt GitHub")

        return response.json()['html_url']

    def _process_pdf(self):
        """Traite le PDF et crée le vectorstore."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(self.pdf_content)
            pdf_path = tmp_file.name

        # Charger et traiter le PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        docs = text_splitter.split_documents(pages)

        # Créer les embeddings
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.vectorstore = FAISS.from_documents(docs, embeddings)

        # Sauvegarder le vectorstore
        self.vectorstore.save_local("vectorstore")

        # Nettoyer
        os.unlink(pdf_path)

    def _push_code_to_github(self):
        """Pousse le code vers GitHub."""
        headers = {
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json'
        }

        # Créer requirements.txt
        requirements = """
streamlit==1.31.0
langchain==0.1.0
langchain-community==0.0.16
langchain-core==0.1.17
openai==1.10.0
python-dotenv==1.0.0
pypdf==3.17.4
faiss-cpu==1.7.4
tiktoken==0.5.2
"""
        self._create_github_file("requirements.txt", requirements, headers)

        # Créer .gitignore
        gitignore = """
.env
__pycache__/
*.pyc
"""
        self._create_github_file(".gitignore", gitignore, headers)

        # Créer app.py
        app_code = f'''
import streamlit as st
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

st.title("Chat avec {self.pdf_name}")

# Initialiser la conversation
@st.cache_resource
def init_conversation():
    embeddings = OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"])
    vectorstore = FAISS.load_local("vectorstore")
    llm = ChatOpenAI(temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return conversation

# Initialiser l'historique des messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

try:
    conversation = init_conversation()

    # Afficher l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Zone de chat
    if prompt := st.chat_input("Posez votre question sur le PDF"):
        # Afficher la question
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({{"role": "user", "content": prompt}})

        # Obtenir et afficher la réponse
        with st.chat_message("assistant"):
            response = conversation({{"question": prompt, "chat_history": []}})
            st.markdown(response["answer"])
        st.session_state.messages.append({{"role": "assistant", "content": response["answer"]}})

except Exception as e:
    st.error("Erreur lors du chargement du chatbot. Veuillez réessayer plus tard.")
'''
        self._create_github_file("app.py", app_code, headers)

        # Créer le fichier vectorstore
        files = os.listdir("vectorstore")
        for file in files:
            with open(os.path.join("vectorstore", file), 'rb') as f:
                content = f.read()
                encoded = base64.b64encode(content).decode()
                self._create_github_file(f"vectorstore/{file}", encoded, headers, is_binary=True)

    def _create_github_file(self, path, content, headers, is_binary=False):
        """Crée un fichier dans le dépôt GitHub."""
        if not is_binary:
            content = base64.b64encode(content.encode()).decode()

        response = requests.put(
            f'https://api.github.com/repos/{self._get_github_username()}/{self.repo_name}/contents/{path}',
            headers=headers,
            json={
                'message': f'Add {path}',
                'content': content
            }
        )

        if response.status_code not in [201, 200]:
            raise Exception(f"Erreur lors de la création du fichier {path}")

    def _get_github_username(self):
        """Récupère le nom d'utilisateur GitHub."""
        headers = {
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json'
        }
        response = requests.get('https://api.github.com/user', headers=headers)
        return response.json()['login']

    def create_chatbot(self):
        """Crée un nouveau chatbot pour le PDF."""
        try:
            # 1. Traiter le PDF
            self._process_pdf()

            # 2. Créer le dépôt GitHub
            repo_url = self._create_github_repo()

            # 3. Pousser le code
            self._push_code_to_github()

            # 4. Générer l'URL Streamlit
            streamlit_url = f"https://{self.repo_name}.streamlit.app"

            return {
                'success': True,
                'repo_url': repo_url,
                'streamlit_url': streamlit_url
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

def main():
    st.title("Créateur de PDF Chatbots")
    
    st.markdown("""
    ## Comment ça marche
    1. Uploadez votre PDF
    2. Cliquez sur "Créer Chatbot"
    3. Attendez quelques minutes que le chatbot soit déployé
    4. Utilisez le lien fourni pour accéder à votre chatbot
    """)

    uploaded_file = st.file_uploader("Choisissez un fichier PDF", type="pdf")
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"PDF sélectionné : {uploaded_file.name}")
        with col2:
            if st.button("Créer Chatbot"):
                with st.spinner("Création de votre chatbot en cours..."):
                    creator = PDFChatbotCreator(
                        pdf_content=uploaded_file.getvalue(),
                        pdf_name=uploaded_file.name
                    )
                    result = creator.create_chatbot()
                    
                    if result['success']:
                        st.success("Chatbot créé avec succès!")
                        st.markdown("## Liens de votre chatbot")
                        st.markdown(f"🔗 **URL du chatbot:** [{result['streamlit_url']}]({result['streamlit_url']})")
                        st.markdown("""
                        ⚠️ **Note:** Le déploiement initial peut prendre quelques minutes. 
                        Si le lien ne fonctionne pas immédiatement, attendez un peu et réessayez.
                        """)
                        st.markdown(f"📦 **Code source:** [{result['repo_url']}]({result['repo_url']})")
                    else:
                        st.error(f"Erreur lors de la création du chatbot : {result['error']}")

if __name__ == '__main__':
    main()
