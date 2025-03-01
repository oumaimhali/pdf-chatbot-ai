import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import tempfile
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class PDFChatbot:
    def __init__(self, pdf_path=None, pdf_content=None):
        self.pdf_path = pdf_path
        self.pdf_content = pdf_content
        self.vectorstore = None
        self.conversation = None

    def process_pdf(self):
        """Traite le fichier PDF et crée une base de données vectorielle."""
        # Si le contenu est fourni directement (upload Streamlit)
        if self.pdf_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(self.pdf_content)
                self.pdf_path = tmp_file.name

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

        # Créer les embeddings
        embeddings = OpenAIEmbeddings()
        
        # Créer la base de données vectorielle
        self.vectorstore = FAISS.from_documents(docs, embeddings)

        # Nettoyer le fichier temporaire si nécessaire
        if self.pdf_content:
            os.unlink(self.pdf_path)

    def get_conversation_chain(self):
        """Crée une chaîne de conversation avec la base de données vectorielle."""
        if not self.vectorstore:
            return None
        
        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        return chain

    def get_response(self, question, chat_history=[]):
        """Obtient une réponse à une question."""
        if self.vectorstore:
            # Rechercher les documents pertinents
            docs = self.vectorstore.similarity_search(question)
            
            # Obtenir la réponse
            chain = self.get_conversation_chain()
            response = chain.run(input_documents=docs, question=question)
            
            return response
        return "Veuillez d'abord charger un document PDF."

def main():
    st.title("PDF Chatbot 📚")
    
    # Initialiser l'état de session
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Section upload
    st.markdown("## 📤 Upload ton PDF")
    uploaded_file = st.file_uploader("Choisis un fichier PDF", type="pdf")
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"PDF sélectionné : {uploaded_file.name}")
        with col2:
            if st.button("Traiter le PDF"):
                with st.spinner("Traitement du PDF en cours..."):
                    chatbot = PDFChatbot(pdf_content=uploaded_file.getvalue())
                    chatbot.process_pdf()
                    st.session_state.chatbot = chatbot
                    st.success("PDF traité avec succès!")

    # Section chat
    if st.session_state.chatbot:
        st.markdown("---")
        st.markdown("## 💬 Chat avec ton PDF")
        
        # Zone de saisie pour la question
        question = st.text_input("Pose ta question sur le contenu du PDF :")
        
        if question:
            # Obtenir et afficher la réponse
            response = st.session_state.chatbot.get_response(question)
            
            # Ajouter à l'historique
            st.session_state.chat_history.append({"question": question, "answer": response})
            
            # Afficher l'historique
            for chat in st.session_state.chat_history:
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**R:** {chat['answer']}")
                st.markdown("---")

if __name__ == '__main__':
    main()
