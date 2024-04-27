import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, AutoTokenizer
import random
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


df = pd.read_csv('customer_product_1.csv')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Define the Recommender System using BERT embeddings and updated architecture
class RecommenderSystem(nn.Module):
    def __init__(self, num_products):
        super(RecommenderSystem, self).__init__()
        self.bert_model = bert_model
        self.user_embeddings = nn.Parameter(torch.randn(len(df), bert_model.config.hidden_size), requires_grad=True)
        self.product_embeddings = nn.Embedding(num_products, bert_model.config.hidden_size)

    def forward(self, texts, product_ids):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            bert_output = self.bert_model(**inputs)
        user_features = bert_output.pooler_output
        user_features = user_features + self.user_embeddings  # Update user features with learned embeddings
        product_features = self.product_embeddings(product_ids)

        scores = torch.matmul(user_features, product_features.T)
        return torch.softmax(scores, dim=1)

# Instantiate the recommender system
recommender = RecommenderSystem(num_products=10)

# Prepare and process descriptions
user_descs = df['sum_text'].tolist()
product_ids = torch.arange(0, 10)  # Assuming 10 different product types for simplicity

# Generate predictions
recommendations = recommender(user_descs, product_ids)
top_two_products = torch.topk(recommendations, 2)

data_rows = []
# Loop over each customer's top two product recommendations
for idx, (products, scores) in enumerate(zip(top_two_products.indices, top_two_products.values)):
    product_names = [f"product_0{i+1}" for i in products.tolist()]
    score_values = [round(score, 6) for score in scores.tolist()] 
    
    # Prepare the row as a tuple of tuples, each containing a product name and score
    row = (tuple(zip(product_names, score_values)))
    
    # Append each row to the data_rows list
    data_rows.append(row)

# Create a DataFrame with the collected rows and specified column names
df_res = pd.DataFrame(data_rows, columns=["product1", "product2"])

print(df_res)


##############agrregated customer level: BertModel using clustering#######################
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import pandas as pd
import random
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

df = pd.read_csv('customer_product_2.csv')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Text vectorization and clustering
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['sum_text'])
kmeans = KMeans(n_clusters=5, random_state=42).fit(X)
df['cluster'] = kmeans.labels_

# Define the Recommender System using BERT embeddings and updated architecture
class RecommenderSystem(nn.Module):
    def __init__(self, num_users, num_products=10):
        super(RecommenderSystem, self).__init__()
        self.bert_model = bert_model
        self.user_embeddings = nn.Parameter(torch.randn(num_users, bert_model.config.hidden_size), requires_grad=True)
        self.product_embeddings = nn.Embedding(num_products, bert_model.config.hidden_size)

    def forward(self, texts, product_ids):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        bert_output = self.bert_model(**inputs)
        user_features = bert_output.pooler_output + self.user_embeddings
        product_features = self.product_embeddings(product_ids)

        scores = torch.matmul(user_features, product_features.T)
        return torch.softmax(scores, dim=1)

# Prepare and process descriptions grouped by cluster
for cluster in sorted(df['cluster'].unique()):
    cluster_indices = df.index[df['cluster'] == cluster].tolist()
    cluster_descs = df.loc[cluster_indices, 'sum_text'].tolist()
    recommender = RecommenderSystem(num_users=len(cluster_indices))  # Adjust the number of user embeddings dynamically

    recommendations = recommender(cluster_descs, torch.arange(0, 10))
    top_two_products = torch.topk(recommendations, 2)

    for idx, (products, scores) in enumerate(zip(top_two_products.indices, top_two_products.values)):
        product_names = [f"product_0{i+1}" for i in products.tolist()]
        score_values = [round(score, 4) for score in scores.tolist()]  # Round scores to four decimal places
        print(f"Cluster {cluster}, Customer {df.loc[cluster_indices[idx], 'Customer ID']} recommendations: {list(zip(product_names, score_values))}")
        

##############transactional format rating GPT2LMHeadModel method#####################
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, TensorDataset
import random


# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set pad token for padding

# Enhanced data generation with structured profiles for purchase quantity
random.seed(42)

df_transactions = pd.read_csv('customer_product_3.csv')

# Now use this DataFrame to generate the 'data' list
data = []
for index, row in df_transactions.iterrows():
    user = row['User ID']
    product = row['Product ID']
    if product in preferences.get(user, []):
        # Here we format the string to include transaction details in a structured way
        structured_data = f"{product}{{recency:{row['Recency']}, frequency:{row['Frequency']}, T:{row['Time Since First Purchase']}}}"
        data.append((user, structured_data, row['Purchase Quantity']))

# Display the formatted data list
for d in data[:10]:  # Show first 10 elements for brevity
    print(d)

# Process data
texts = ["[CLS] " + u + " [SEP] " + p + " [SEP]" for u, p, _ in data]
inputs = tokenizer(texts, return_tensors="pt", padding=True, max_length=100, truncation=True)
quantities = torch.tensor([q for _, _, q in data], dtype=torch.long)

# DataLoader
dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, quantities)
loader = DataLoader(dataset, batch_size=5, shuffle=True)

# Training configuration
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
model.train()

# Training loop
for epoch in range(1):  # Short loop for demonstration
    for batch in loader:
        b_input_ids, b_attention_mask, b_quantities = batch
        optimizer.zero_grad()
        outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# Prediction and ranking logic
model.eval()
customer_products = {u: set() for u, _, _ in data}
for _, p, _ in data:
    for u in customer_products:
        customer_products[u].add(p.split('{')[0])  # Split to only use the base product ID for comparison

customer_scores = {u: {} for u in customer_products}
with torch.no_grad():
    predictions = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    logits = predictions.logits
    scores = torch.softmax(logits, dim=-1)

    for idx, ((u, p, _), input_id) in enumerate(zip(data, inputs.input_ids)):
        product_scores = scores[idx, -1, :]
        top_scores, top_idx = torch.topk(product_scores, k=5)
        for score, vocab_idx in zip(top_scores, top_idx):
            product = tokenizer.decode([vocab_idx]).strip().split('{')[0]  # Again, split to use base product ID
            if product.isdigit() and product in customer_products[u]:
                if product not in customer_scores[u]:
                    customer_scores[u][product] = 0.0
                customer_scores[u][product] += score.item()

for u in sorted(customer_scores):
    sorted_products = sorted(customer_scores[u].items(), key=lambda x: x[1], reverse=True)[:2]
    print(f"For customer {u}, recommend products {sorted_products}")

