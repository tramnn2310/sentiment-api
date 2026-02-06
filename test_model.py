from src.model.sentiment_model import SentimentModel

# Initialize the model
model = SentimentModel()

# Test single prediction
result = model.predict("This product is amazing!")
print("Single prediction:", result)

