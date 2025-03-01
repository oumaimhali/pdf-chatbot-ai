def _process_pdf(self):
    """Traite le PDF et crée le vectorstore."""
    logger.info("Début du traitement du PDF")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(self.pdf_content)
        pdf_path = tmp_file.name
        logger.info(f"Fichier PDF temporaire créé : {pdf_path}")

    # Charger et traiter le PDF
    logger.info("Chargement du PDF avec PyPDFLoader")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    logger.info(f"Nombre de pages chargées : {len(pages)}")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents(pages)
    logger.info(f"Nombre de documents après découpage : {len(docs)}")

    # Créer les embeddings
    logger.info("Création des embeddings avec OpenAI")
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    self.vectorstore = FAISS.from_documents(docs, embeddings)
    logger.info("Vectorstore créé avec succès")

    # Sauvegarder le vectorstore
    self.vectorstore.save_local("vectorstore")
    logger.info("Vectorstore sauvegardé localement")

    # Nettoyer
    os.unlink(pdf_path)
    logger.info("Fichier PDF temporaire supprimé")
