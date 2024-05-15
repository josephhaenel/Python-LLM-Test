async function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() === '') return;

    appendMessage('user', userInput);

    const response = await fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userInput })
    });
    
    const data = await response.json();
    appendMessage('bot', data.response);
    document.getElementById('user-input').value = '';
}

function appendMessage(sender, text) {
    const message = document.createElement('div');
    message.classList.add('message', sender);

    const icon = document.createElement('i');
    icon.classList.add('fas', sender === 'user' ? 'fa-user' : 'fa-robot', 'message-icon');

    const messageText = document.createElement('span');
    messageText.textContent = text;

    message.appendChild(icon);
    message.appendChild(messageText);
    document.getElementById('chat-messages').appendChild(message);
    document.getElementById('chat-box').scrollTop = document.getElementById('chat-box').scrollHeight;
}

async function resetConversation() {
    await fetch('/reset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    });
    document.getElementById('chat-messages').innerHTML = '';
    appendMessage('bot', 'Conversation reset.');
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}
