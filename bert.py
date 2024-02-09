from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import os

from datasets import Dataset
from transformers import Trainer, TrainingArguments
import pandas as pd

torch.cuda.empty_cache()
print(torch.cuda.memory_summary())

if os.path.isdir('./parsbert_model') and os.path.isdir('./parsbert_tokenizer'):
    model = AutoModelForMaskedLM.from_pretrained('./parsbert_model')
    tokenizer = AutoTokenizer.from_pretrained('./parsbert_tokenizer')

else:
    # Load ParsBERT model and tokenizer
    model_name = "HooshvareLab/bert-fa-base-uncased"
    # model_name = "HooshvareLab/roberta-fa-zwnj-base"
    # model_name = "HooshvareLab/albert-fa-zwnj-base-v2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    model.save_pretrained('./parsbert_model')
    tokenizer.save_pretrained('./parsbert_tokenizer')


def create_dataset():
    # Replace 'your_dataset.csv' with the path to your CSV file
    # Assume the queries are stored in a column named 'query'
    csv_file_path = 'user_query.csv'
    df = pd.read_csv(csv_file_path)

    # Convert the relevant column to a list of queries
    queries = df['search_input'].tolist()

    # Now, `queries` contains all your queries from the CSV file
    # Example dataset preparation
    data = {"text": queries}
    dataset = Dataset.from_dict(data)

    # Tokenize the dataset
    # def tokenize_function(examples):
    #     return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    def preprocess_function(examples):
        # Replace this with your dataset's text column
        sentences = examples["text"]
        # Tokenize the texts and prepare inputs and labels
        inputs = tokenizer(sentences, padding="max_length", truncation=True, max_length=2, return_tensors="pt")
        inputs["labels"] = inputs.input_ids.detach().clone()
        # Masking operation here (this is just a simple example)
        # You should implement a more sophisticated masking strategy
        rand = torch.rand(inputs.input_ids.shape)
        mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
        selection = []
        for i in range(inputs.input_ids.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )
        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = 103  # 103 is the mask token ID for BERT
        # Set to -100 the labels of non-masked tokens so they are not considered in the loss
        inputs["labels"][~mask_arr] = -100
        return inputs



    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    return tokenized_datasets


def fine_tune(tokenized_datasets):
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-3,
        per_device_train_batch_size=64,
        num_train_epochs=30,
        weight_decay=0.01,
        # fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    print(torch.cuda.memory_summary())
    trainer.train()
    print(torch.cuda.memory_summary())



def predict_next_words(text, top_n=4):
    text_with_mask = text + " " + tokenizer.mask_token
    input_ids = tokenizer.encode(text_with_mask, return_tensors="pt")

    with torch.no_grad():
        predictions = model(input_ids)[0]

    masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    logits = predictions[0, masked_index, :]
    top_indices = logits.topk(top_n).indices[0].tolist()

    predicted_tokens = [tokenizer.decode([index]).strip() for index in top_indices]

    return predicted_tokens


tokenized_datasets = create_dataset()
fine_tune(tokenized_datasets)
# Adjust the example usage accordingly
predicted_next_words = predict_next_words("میدان انقلاب", 4)
print(f"Predicted next words: {predicted_next_words}")


# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# from transformers import AutoTokenizer
# import torch
#
# class CustomDataset(Dataset):
#     def __init__(self, csv_file, tokenizer, max_length=512):
#         self.dataframe = pd.read_csv(csv_file)
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#
#     def __len__(self):
#         return len(self.dataframe)
#
#     def __getitem__(self, idx):
#         text = self.dataframe.iloc[idx]['search_input']
#         inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
#         return inputs
#
# # Example usage with your tokenizer and file path
# tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/albert-fa-zwnj-base-v2")
# custom_dataset = CustomDataset(csv_file='user_query.csv', tokenizer=tokenizer)
#
# # DataLoader for batch processing
# data_loader = DataLoader(custom_dataset, batch_size=16, shuffle=True)
#
# # In your training loop, iterate over `data_loader` instead of `tokenized_datasets`
# for batch in data_loader:
#     # Process each batch
#     # Your training logic here, e.g., pass `batch` to your model
#     pass

