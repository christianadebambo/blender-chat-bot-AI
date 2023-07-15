from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Download and setup the model and tokenizer
model_name = 'facebook/blenderbot-400M-distill'
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Define an utterance
utterance = "My name is hello, I like world"

# Tokenize the utterance
inputs = tokenizer(utterance, return_tensors="pt")

# Generate model results
result = model.generate(**inputs)

# Decode the result
response = tokenizer.decode(result[0])

# Print the generated response
print("Generated Response:", response)
