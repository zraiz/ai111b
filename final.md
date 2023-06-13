#110911542 final

Check out this program that utilizes a pre-trained Transformer model (BERT) and the Hugging Face library to perform sentiment analysis:
```
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def perform_sentiment_analysis(text):
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = inputs.to(device)

    # Forward pass through the model
    outputs = model(**inputs)
    logits = outputs.logits

    # Predict the sentiment label
    _, predicted_label = torch.max(logits, dim=1)
    sentiment_label = predicted_label.item()

    # Map sentiment label to actual sentiment
    sentiments = ['Negative', 'Neutral', 'Positive']
    sentiment = sentiments[sentiment_label]

    return sentiment

# Main program
while True:
    # Take user input
    text = input("Enter text to analyze sentiment (or 'exit' to quit): ")

    if text.lower() == "exit":
        break

    # Perform sentiment analysis
    sentiment = perform_sentiment_analysis(text)
    print("Sentiment:", sentiment)
```
Using Hugging Face transformers library, this program loads a pre-trained Transformer model (BERT) and tokenizer. Once the program is initiated, the user will be prompted to input text for sentiment analysis. The perform_sentiment_analysis function will then tokenize the input text, pass it through the loaded model, predict the sentiment label and map it to the corresponding sentiment (negative, neutral or positive). Finally, the predicted sentiment will be displayed.
Please note that to run this program, you will need to have the transformers library installed (pip install transformers) and a compatible version of PyTorch installed. Additionally, you may need to download additional resources or pre-trained models depending on the specific Transformer model you wish to use.
I hope this information is helpful. Please feel free to reach out if you have any further questions.
>CHATGPT (NO MODIFY)
