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
            # Créer un hash unique basé sur le nom du fichier
            hash_object = hashlib.md5(self.pdf_name.encode())
            self.chat_id = hash_object.hexdigest()[:8]
            return self.chat_id
        return None

    def get_embed_code(self):
        """Génère le code d'intégration HTML pour ce PDF spécifique."""
        if not self.chat_id:
            self.generate_chat_id()
        
        return f'''
<!-- Code d'intégration pour {self.pdf_name} -->
<div id="pdf-chat-{self.chat_id}"></div>
<script src="https://your-domain.com/static/chatpdf-widget.js"></script>
<script>
    new PDFChatWidget({{
        containerId: "pdf-chat-{self.chat_id}",
        chatId: "{self.chat_id}",
        pdfName: "{self.pdf_name}"
    }});
</script>
'''

    def process_pdf(self):
        """Traite le fichier PDF et crée une base de données vectorielle."""
        if self.pdf_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(self.pdf_content)
                self.pdf_path = tmp_file.name

        # Générer un ID unique pour ce chat
        self.generate_chat_id()

        # Charger le PDF
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()

        # Diviser le texte en chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        docs = text_splitter.split_documents(pages)

        # Créer les embeddings avec la clé API explicite
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        
        # Créer la base de données vectorielle
        self.vectorstore = FAISS.from_documents(docs, embeddings)

        # Nettoyer le fichier temporaire si nécessaire
        if self.pdf_content:
            os.unlink(self.pdf_path)

        # Initialiser la chaîne de conversation
        llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)
        self.conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(),
            return_source_documents=True
        )

    def get_response(self, question, chat_history=[]):
        """Obtient une réponse à une question."""
        if self.conversation:
            response = self.conversation({"question": question, "chat_history": chat_history})
            return response["answer"]
        return "Veuillez d'abord charger un document PDF."

def main():
    st.title("PDF Chatbot")
    
    # Initialiser l'état de session
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'embed_code' not in st.session_state:
        st.session_state.embed_code = None

    # Section upload
    st.markdown("## Upload ton PDF")
    uploaded_file = st.file_uploader("Choisis un fichier PDF", type="pdf")
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"PDF sélectionné : {uploaded_file.name}")
        with col2:
            if st.button("Traiter le PDF"):
                with st.spinner("Traitement du PDF en cours..."):
                    chatbot = PDFChatbot(pdf_content=uploaded_file.getvalue(), pdf_name=uploaded_file.name)
                    chatbot.process_pdf()
                    st.session_state.chatbot = chatbot
                    st.session_state.embed_code = chatbot.get_embed_code()
                    st.success("PDF traité avec succès!")

    # Section code d'intégration
    if st.session_state.embed_code:
        st.markdown("## Code d'intégration")
        st.markdown("Utilisez ce code pour intégrer le chatbot sur votre site web :")
        st.code(st.session_state.embed_code, language="html")

    # Section chat
    if st.session_state.chatbot:
        st.markdown("---")
        st.markdown("## Chat avec ton PDF")
        
        # Zone de saisie pour la question
        question = st.text_input("Pose ta question sur le contenu du PDF :")
        
        if question:
            # Obtenir et afficher la réponse
            response = st.session_state.chatbot.get_response(question, st.session_state.chat_history)
            
            # Ajouter à l'historique
            st.session_state.chat_history.append({"question": question, "answer": response})
            
            # Afficher l'historique
            for chat in st.session_state.chat_history:
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**R:** {chat['answer']}")
                st.markdown("---")

if __name__ == '__main__':
    main()
