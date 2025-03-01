import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import os
import json
import base64
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configurer la clé API OpenAI
if 'OPENAI_API_KEY' not in st.secrets and 'OPENAI_API_KEY' not in os.environ:
    st.error("Veuillez configurer votre clé API OpenAI dans les secrets Streamlit ou le fichier .env")
    st.stop()

OPENAI_API_KEY = st.secrets.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

class PDFChatbot:
    def __init__(self, pdf_path=None, pdf_content=None):
        self.pdf_path = pdf_path
        self.pdf_content = pdf_content
        self.vectorstore = None
        self.conversation = None
        
    def process_pdf(self):
        """Traite le fichier PDF et crée une base de données vectorielle."""
        if self.pdf_content:
            # Sauvegarder temporairement le fichier PDF
            with open("temp.pdf", "wb") as f:
                f.write(self.pdf_content)
            self.pdf_path = "temp.pdf"
        
        # Charger le PDF
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        
        # Diviser le texte en chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(pages)
        
        # Créer les embeddings et la base de données vectorielle
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Créer la chaîne de conversation
        self.conversation = self.get_conversation_chain()
        
        # Nettoyer si fichier temporaire
        if self.pdf_path == "temp.pdf":
            os.remove("temp.pdf")
            
        # Sauvegarder la base vectorielle
        self.save_vectorstore()
        
    def get_conversation_chain(self):
        """Crée une chaîne de conversation avec la base de données vectorielle."""
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True
        )
        return chain
        
    def save_vectorstore(self):
        """Sauvegarde la base vectorielle."""
        if self.vectorstore:
            self.vectorstore.save_local("vectorstore")
            
    @staticmethod
    def load_vectorstore():
        """Charge une base vectorielle existante."""
        if os.path.exists("vectorstore"):
            embeddings = OpenAIEmbeddings()
            return FAISS.load_local("vectorstore", embeddings)
        return None
        
    def get_response(self, question, chat_history=[]):
        """Obtient une réponse à une question."""
        if self.conversation:
            response = self.conversation({
                "question": question,
                "chat_history": chat_history
            })
            return response["answer"]
        return "Veuillez d'abord charger un PDF."

def main():
    # Configuration de la page
    st.set_page_config(
        page_title="ChatPDF Creator",
        page_icon="",
        layout="wide"
    )
    
    # Sidebar avec les informations
    with st.sidebar:
        st.title("")
        st.markdown("""
        Cette application vous permet de créer un chatbot pour n'importe quel PDF.
        
        ### Comment utiliser :
        1. Chargez votre PDF
        2. Attendez le traitement
        3. Posez vos questions !
        
        ### Technologies utilisées :
        - Streamlit
        - LangChain
        - OpenAI GPT
        - FAISS
        
        ### Liens :
        - [Code source](https://github.com/votre-username/pdf-chatbot)
        - [Portfolio](https://votre-portfolio.com)
        """)
    
    # Contenu principal
    st.title("")
    st.markdown("""
    Transformez n'importe quel PDF en chatbot intelligent !
    """)
    
    # Initialiser les variables de session
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Section upload et traitement
    uploaded_file = st.file_uploader("", type=["pdf"])
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"PDF sélectionné : {uploaded_file.name}")
        with col2:
            if st.button "":
                with st.spinner("Traitement du PDF en cours..."):
                    chatbot = PDFChatbot(pdf_content=uploaded_file.getvalue())
                    chatbot.process_pdf()
                    st.session_state.chatbot = chatbot
                    st.success("PDF traité avec succès!")

    # Section chat
    if st.session_state.chatbot:
        st.markdown("---")
        st.subheader("")
        
        # Zone de chat
        chat_container = st.container()
        with chat_container:
            for question, answer in st.session_state.chat_history:
                st.markdown(f" **Question:** {question}")
                st.markdown(f" **Réponse:** {answer}")
                st.markdown("---")
        
        # Zone de saisie
        user_question = st.text_input("Posez votre question :")
        if user_question:
            with st.spinner("Recherche de la réponse..."):
                response = st.session_state.chatbot.get_response(
                    user_question,
                    st.session_state.chat_history
                )
                st.session_state.chat_history.append((user_question, response))
                st.experimental_rerun()

if __name__ == '__main__':
    main()
