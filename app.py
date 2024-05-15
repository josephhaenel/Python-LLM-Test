from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__, static_folder='static')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# Initialize chat history
chat_history_ids = None

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history_ids

    user_input = request.json['message']

    # Encode the new user input, add the eos_token and return a tensor in PyTorch
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    # Generate a response while limiting the total chat history to 1000 tokens
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the last output tokens from the bot
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return jsonify({'response': bot_response})

@app.route('/reset', methods=['POST'])
def reset():
    global chat_history_ids
    chat_history_ids = None
    return jsonify({'message': 'Conversation reset.'})

if __name__ == '__main__':
    app.run(debug=True)
