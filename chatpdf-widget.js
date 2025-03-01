class PDFChatWidget {
    constructor(config) {
        this.containerId = config.containerId;
        this.chatId = config.chatId;
        this.pdfName = config.pdfName;
        this.initialize();
    }

    initialize() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`Container ${this.containerId} not found`);
            return;
        }

        // Créer l'interface du chat
        container.innerHTML = `
            <div class="pdf-chat-widget" style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; max-width: 600px;">
                <div class="pdf-chat-header" style="margin-bottom: 16px;">
                    <h3 style="margin: 0;">Chat avec ${this.pdfName}</h3>
                </div>
                <div class="pdf-chat-messages" style="height: 300px; overflow-y: auto; margin-bottom: 16px; padding: 8px; border: 1px solid #eee;"></div>
                <div class="pdf-chat-input" style="display: flex; gap: 8px;">
                    <input type="text" placeholder="Posez votre question..." style="flex: 1; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                    <button style="padding: 8px 16px; background-color: #0066cc; color: white; border: none; border-radius: 4px; cursor: pointer;">Envoyer</button>
                </div>
            </div>
        `;

        // Ajouter les événements
        const input = container.querySelector('input');
        const button = container.querySelector('button');
        const messagesContainer = container.querySelector('.pdf-chat-messages');

        button.addEventListener('click', () => this.sendMessage(input, messagesContainer));
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage(input, messagesContainer);
            }
        });
    }

    async sendMessage(input, messagesContainer) {
        const question = input.value.trim();
        if (!question) return;

        // Afficher la question
        this.addMessage('question', question, messagesContainer);
        input.value = '';

        try {
            // Envoyer la requête à l'API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    chatId: this.chatId,
                    question: question
                })
            });

            const data = await response.json();
            
            // Afficher la réponse
            this.addMessage('answer', data.answer, messagesContainer);
        } catch (error) {
            console.error('Error:', error);
            this.addMessage('error', 'Désolé, une erreur est survenue.', messagesContainer);
        }
    }

    addMessage(type, content, container) {
        const messageDiv = document.createElement('div');
        messageDiv.style.marginBottom = '8px';
        messageDiv.style.padding = '8px';
        messageDiv.style.borderRadius = '4px';
        
        if (type === 'question') {
            messageDiv.style.backgroundColor = '#f0f0f0';
            messageDiv.innerHTML = `<strong>Q:</strong> ${content}`;
        } else if (type === 'answer') {
            messageDiv.style.backgroundColor = '#e3f2fd';
            messageDiv.innerHTML = `<strong>R:</strong> ${content}`;
        } else {
            messageDiv.style.backgroundColor = '#fee';
            messageDiv.innerHTML = `<strong>Erreur:</strong> ${content}`;
        }

        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;
    }
}
