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

# Charger les variables d'environnement
load_dotenv()

# Configurer la clé API OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialiser le stockage des chatbots
if 'chatbots' not in st.session_state:
    st.session_state.chatbots = {}

class PDFChatbot:
    def __init__(self, pdf_path=None, pdf_content=None, pdf_name=None):
        self.pdf_path = pdf_path
        self.pdf_content = pdf_content
        self.pdf_name = pdf_name
        self.vectorstore = None
        self.conversation = None
        self.chat_id = None

    def generate_chat_id(self):
        """Génère un ID unique pour le chat basé sur le nom du PDF."""
        if self.pdf_name:
            hash_object = hashlib.md5(self.pdf_name.encode())
            self.chat_id = hash_object.hexdigest()[:8]
            return self.chat_id
        return None

    def process_pdf(self):
        """Traite le fichier PDF et crée une base de données vectorielle."""
        if self.pdf_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(self.pdf_content)
                self.pdf_path = tmp_file.name

        # Générer un ID unique
        self.generate_chat_id()

        # Charger le PDF
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()

        # Diviser le texte
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        docs = text_splitter.split_documents(pages)

        # Créer les embeddings
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self.vectorstore = FAISS.from_documents(docs, embeddings)

        # Nettoyer le fichier temporaire
        if self.pdf_content:
            os.unlink(self.pdf_path)

        # Initialiser la conversation
        llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)
        self.conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True
        )

        # Sauvegarder dans la session
        st.session_state.chatbots[self.chat_id] = self

    def get_response(self, question, chat_history=[]):
        """Obtient une réponse à une question."""
        if self.conversation:
            response = self.conversation({"question": question, "chat_history": chat_history})
            return response["answer"]
        return "Veuillez d'abord charger un document PDF."

def main():
    st.title("PDF Chatbot")

    # Vérifier si on est sur une page de chat spécifique
    chat_id = st.query_params.get("chat_id", None)

    if chat_id and chat_id in st.session_state.chatbots:
        # Interface de chat pour un PDF spécifique
        chatbot = st.session_state.chatbots[chat_id]
        st.markdown(f"## Chat avec {chatbot.pdf_name}")
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Afficher l'historique des messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Zone de chat
        if prompt := st.chat_input("Posez votre question sur le PDF"):
            # Afficher la question de l'utilisateur
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Obtenir et afficher la réponse
            with st.chat_message("assistant"):
                response = chatbot.get_response(prompt)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        # Interface principale pour l'upload
        st.markdown("## Upload ton PDF")
        uploaded_file = st.file_uploader("Choisis un fichier PDF", type="pdf")
        
        if uploaded_file:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"PDF sélectionné : {uploaded_file.name}")
            with col2:
                if st.button("Créer Chatbot"):
                    with st.spinner("Création du chatbot en cours..."):
                        chatbot = PDFChatbot(pdf_content=uploaded_file.getvalue(), pdf_name=uploaded_file.name)
                        chatbot.process_pdf()
                        
                        # Générer le lien
                        chat_url = f"?chat_id={chatbot.chat_id}"
                        st.success("Chatbot créé avec succès!")
                        st.markdown("## Lien de votre chatbot")
                        st.markdown(f"[Cliquez ici pour accéder à votre chatbot]({chat_url})")

if __name__ == '__main__':
    main()
