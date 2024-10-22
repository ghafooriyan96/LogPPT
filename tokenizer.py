'''
Generate the Tokenizer file based on the generated structured logs from Logppt
'''
import pandas as pd
from transformers import GPT2Tokenizer
import json

# Load the CSV files
structured_logs = pd.read_csv('Structured/32shot/Apache_2k.log_structured.csv')

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Tokenize the 'Content' column in the structured logs
structured_logs['tokenized_content'] = structured_logs['Content'].apply(lambda x: tokenizer.encode(x, truncation=True, max_length=512))

# Save the tokenized data
structured_logs.to_csv('Structured/32shot/structured_logs_tokenized.csv', index=False)

output_json_file = "Structured/32shot/structured_logs_tokenized.json"

# Read the CSV
df = pd.read_csv("Structured/32shot/structured_logs_tokenized.csv")

# Convert DataFrame to a list of dictionaries (this step will be adapted to your structure)
json_data = df.to_dict(orient="records")

# Save to JSON file
with open(output_json_file, "w") as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"JSON file has been saved to: {output_json_file}")