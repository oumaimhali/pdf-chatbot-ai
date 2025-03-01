class ChatPDFWidget {
    constructor(config) {
        this.containerId = config.containerId;
        this.chatbotId = config.chatbotId;
        this.apiUrl = 'https://votre-domaine.com/api/chat';
        this.initialize();
    }

    initialize() {
        // Créer l'interface du widget
        const container = document.getElementById(this.containerId);
        container.innerHTML = `
            <div class="chatpdf-widget">
                <div class="chatpdf-messages"></div>
                <div class="chatpdf-input">
                    <input type="text" placeholder="Posez votre question...">
                    <button>Envoyer</button>
                </div>
            </div>
        `;

        // Ajouter le style
        const style = document.createElement('style');
        style.textContent = `
            .chatpdf-widget {
                width: 100%;
                max-width: 500px;
                height: 400px;
                border: 1px solid #ccc;
                border-radius: 8px;
                display: flex;
                flex-direction: column;
                font-family: Arial, sans-serif;
            }
            .chatpdf-messages {
                flex: 1;
                overflow-y: auto;
                padding: 15px;
            }
            .chatpdf-input {
                display: flex;
                padding: 10px;
                border-top: 1px solid #ccc;
            }
            .chatpdf-input input {
                flex: 1;
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 4px;
                margin-right: 8px;
            }
            .chatpdf-input button {
                padding: 8px 16px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            .chatpdf-input button:hover {
                background-color: #0056b3;
            }
            .message {
                margin-bottom: 10px;
                padding: 8px;
                border-radius: 4px;
            }
            .user-message {
                background-color: #e3f2fd;
                margin-left: 20%;
            }
            .bot-message {
                background-color: #f5f5f5;
                margin-right: 20%;
            }
        `;
        document.head.appendChild(style);

        // Ajouter les événements
        const input = container.querySelector('input');
        const button = container.querySelector('button');
        const messages = container.querySelector('.chatpdf-messages');

        button.addEventListener('click', () => this.sendMessage(input, messages));
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage(input, messages);
            }
        });
    }

    async sendMessage(input, messages) {
        const message = input.value.trim();
        if (!message) return;

        // Afficher le message de l'utilisateur
        this.addMessage(messages, message, 'user');
        input.value = '';

        try {
            // Envoyer la requête à l'API
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    chatbotId: this.chatbotId,
                    message: message
                })
            });

            const data = await response.json();
            
            // Afficher la réponse du bot
            this.addMessage(messages, data.response, 'bot');
        } catch (error) {
            console.error('Erreur:', error);
            this.addMessage(messages, 'Désolé, une erreur est survenue.', 'bot');
        }
    }

    addMessage(container, text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        messageDiv.textContent = text;
        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;
    }
}
