import os
import json
import random

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import requests
from dotenv import load_dotenv
import config


'''
load_dotenv() 
GOOGLE_ROUTES_API_KEY = os.getenv("GOOGLE_ROUTES_API_KEY")
payload = config.DEFAULT_ROUTES_PAYLOAD
#Local for now -> supposed to set up with Pay now.
account_balance_limit = config.DEFAULT_ACCOUNT_BALANCE_LIMIT
account_balance_curr = config.ACCOUNT_BALANCE
'''
#Setting Up default values for code to function 
account_balance_limit = 50.0
account_balance_curr = 20.0

class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.f1 = nn.Linear(input_size, 128)
        self.f2 = nn.Linear(128, 64)
        self.f3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.f1(x))
        x = self.dropout(x)
        x = self.relu(self.f2(x))
        x = self.dropout(x)
        x = self.f3(x)

        return x


class Assistant:

    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.function_mappings = function_mappings

        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            for intent in intents_data['intents']:
                if intent['tag'] not in self.intents:
                    self.intents.append(intent['tag'])
                    self.intents_responses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent['tag']))

                self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = Model(self.X.shape[1], len(self.intents)) 

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss
            
            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as f:
            json.dump({ 'input_size': self.X.shape[1], 'output_size': len(self.intents) }, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.model = Model(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None


def find_parking():
    """
    global payload
    print("\n[FUNCTION_CALLED: find_parking]")
    
    routes_api_url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_ROUTES_API_KEY,
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline"
    }

    try:
        response = requests.post(routes_api_url, json=payload, headers=headers)
            
        if response.status_code == 200:
            print("Google Routes API call Success.")
        else:
            print(f"Failure: status: {response.status_code}")

    except Exception as e:
        print(f"Error occurred during the API call: {e}")
        
    """
    print("\n[FUNCTION_CALLED: find_parking]")
    print("Not Implemented")

def pay_parking():
    global account_balance_curr
    #Example Parking Fee. Actually supposed to get and set from gov_api and use pay now integration.
    parking_fee = 10.50
    
    print("\n[FUNCTION_CALLED: pay_parking]")
    
    if parking_fee > account_balance_curr:
        print(f"PAYMENT FAILED: Parking fee (${parking_fee}) exceeds your balance (${account_balance_curr}).")
    else:
        print(f"Payment of ${parking_fee} (Limit: ${account_balance_curr})...")
        account_balance_curr -= parking_fee
        


def get_decided_parking():
    #Haven't made yet
    print("Not Implemented")


def set_account_balance_limit():
    global account_balance_limit
    
    print("\n[FUNCTION_CALLED: set_account_balance_limit]")
    print(f"Current autopay limit is: ${account_balance_limit}")
    
    try:
        new_limit_str = input("Enter new limit: ")
        new_limit = float(new_limit_str)
        account_balance_limit = new_limit
        print(f"SUCCESS: Your autopay limit has been updated to ${account_balance_limit}")
    except ValueError:
        print("ERROR: Invalid input.")
    except Exception as e:
        print(f"Error: {e}")

def get_balance():
    #Locally for now but cloud api later.
    global account_balance_curr
    print("\n[FUNCTION_CALLED: get_balance]")
    print(f"Your account balance is: ${account_balance_curr}")

if __name__ == '__main__':
    function_mappings = {
        'find_parking': find_parking,
        'pay_parking': pay_parking,
        'get_decided_parking': get_decided_parking,
        'set_account_balance_limit': set_account_balance_limit,
        'get_balance': get_balance
    }
    '''
    assistant = Assistant('intents.json', function_mappings)
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=100)
    assistant.save_model('chatbot_model.pth', 'dimensions.json')
    '''
    assistant = Assistant('intents.json', function_mappings)
    assistant.parse_intents()
    assistant.load_model('chatbot_model.pth', 'dimensions.json')

    while True:
        message = input('Enter your message:')

        if message == '/quit':
            break

        print(assistant.process_message(message))


    

    
