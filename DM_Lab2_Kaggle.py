import json
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from safetensors.torch import load_file
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast

# ### Import and preprocess data

# Load JSON data
with open('tweets_DM.json', 'r') as f:
    data = [json.loads(line) for line in f]

# Load CSV files
emotion = pd.read_csv('emotion.csv')
data_identification = pd.read_csv('data_identification.csv')

# Transform JSON data into a DataFrame
df = pd.DataFrame(data)
_source = df['_source'].apply(lambda x: x['tweet'])
df = pd.DataFrame({
    'tweet_id': _source.apply(lambda x: x['tweet_id']),
    'hashtags': _source.apply(lambda x: x['hashtags']),
    'text': _source.apply(lambda x: x['text']),
})

# Preprocess text data
df['text'] = df['text'].str.replace('<LH>', '', regex=False).str.strip()# remove <LH>
df['text'] = df['text'].str.replace(r'@\w+', '', regex=True)# remove @ 
df['text'] = df['text'].str.replace(r'#\w+', '', regex=True)# remove #
df['text'] = df['text'].str.replace(r'http\S+|www.\S+', '', regex=True)# remove URL
df['text'] = df['text'].str.lower()# convert to lower case

# Remove duplicate rows
# df.drop_duplicates(subset=['text'], keep=False, inplace=True)

# Merge with identification data
df = df.merge(data_identification, on='tweet_id', how='left')

# Split into train and test sets
train_data = df[df['identification'] == 'train']
test_data = df[df['identification'] == 'test']
train_data = train_data.merge(emotion, on='tweet_id', how='left')
test_data = test_data.drop(['hashtags', 'identification'], axis=1)

# ### Data preparation for BERT

x_data = train_data["text"]
y_data = train_data['emotion']
label_encoder = LabelEncoder()
y_data_encoded = label_encoder.fit_transform(y_data)# Encode labels

# Split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data_encoded, test_size=0.2, random_state=42)

# Tokenize text data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(x_train), truncation=True, padding=True, max_length=128, return_tensors="pt")
val_encodings = tokenizer(list(x_val), truncation=True, padding=True, max_length=128, return_tensors="pt")

# Build PyTorch Dataset
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# ### Load and configure BERT model

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained BERT model
num_labels = len(label_encoder.classes_)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
model = model.to(device)

# Training parameters
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available()
)

train_dataset = EmotionDataset(train_encodings, y_train)
val_dataset = EmotionDataset(val_encodings, y_val)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# ### Predict on test data

# Preprocess test data
batch_size = 32
inputs = tokenizer(list(test_data['text']), return_tensors="pt", padding=True, truncation=True, max_length=128)
dataset = TensorDataset(*inputs.values())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Switch to evaluation mode
model.eval()

# Make predictions
all_predictions = []
with torch.no_grad():
    for batch in dataloader:
        batch = {key: val.to(device, non_blocking=True) for key, val in zip(inputs.keys(), batch)}

        outputs = model(**batch)
        batch_predictions = torch.argmax(outputs.logits, dim=-1)

        # Collect prediction results
        all_predictions.extend(batch_predictions.cpu().numpy())

        # Clean up memory
        del batch, outputs, batch_predictions
        torch.cuda.empty_cache()

# Convert results format
all_predictions = np.array(all_predictions)


# Convert predictions to labels
y_pred_labels = label_encoder.inverse_transform(all_predictions)

# Create submission file
submission = pd.DataFrame({
    'id': test_data['tweet_id'],
    'emotion': y_pred_labels
})
submission.to_csv('submission.csv', index=False)
print("Submission file saved successfully!")