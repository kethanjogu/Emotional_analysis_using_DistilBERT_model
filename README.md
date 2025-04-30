💬 Emotion Analysis Using DistilBERT and GCN (Hybrid Architecture)
This project implements a hybrid deep learning model that combines DistilBERT (a transformer-based language model) with a Graph Convolutional Network (GCN) for emotion classification from text. The model is capable of detecting six core emotions — joy, sadness, love, anger, fear, and surprise — with improved accuracy by leveraging both semantic features and inter-emotional relationships.


📌 Features
✅ Uses DistilBERT-base-uncased-emotion for fast, contextualized sentence embeddings.

✅ Implements a 2-layer Graph Convolutional Network (GCN) to capture emotion co-occurrence patterns.

✅ Supports both BERT-only and Hybrid (BERT+GCN) classification paths.

✅ Trained using HuggingFace’s Trainer API with integrated evaluation metrics.

✅ Achieves high accuracy and weighted F1-score with faster convergence.

✅ Modular design allows for easy future upgrades (e.g., GATs, multi-label classification).


📁 Dataset
This project uses a custom sentence-level JSON dataset, where each entry contains:

A "text" field (natural language sentence)

An "emotion" label mapped to one of six categories:
joy, sadness, love, anger, fear, surprise

The labels are normalized using a dictionary to handle variations (e.g., “happy” → “joy”).

The dataset is loaded, parsed, and converted to a Pandas DataFrame, then split into training and testing sets (80/20).

Each sentence is tokenized using the DistilBERT tokenizer and converted to HuggingFace's Dataset format for use with the Trainer API.



🧠 Model Architecture
Text Encoder:

DistilBERT-base-uncased-emotion

Outputs 768-dimensional sentence embeddings via mean pooling.

GCN Component:

2 GCN layers (768 → 128 → 64) process emotion co-occurrence graph.

Classification Paths:

BERT-only Classifier: Uses DistilBERT embeddings directly.

Hybrid Classifier: Concatenates BERT + GCN outputs (768 + 64 → 6 classes).

Loss Function: CrossEntropyLoss

Optimizer: Adam

Regularization: Dropout (rate = 0.3)

🛠️ Tech Stack

Component	Library
Text Modeling	HuggingFace Transformers (transformers)
Graph Learning	PyTorch Geometric (torch_geometric)
Data Processing	pandas, json, datasets, numpy
Training & Eval	HuggingFace Trainer, sklearn.metrics
Environment	Python 3.8+, Google Colab / Jupyter
🚀 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/emotion-analysis-hybrid.git
cd emotion-analysis-hybrid
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Add or download the dataset file (e.g., sentence_level_tokenized.json) into the working directory.

Run the training script in a Jupyter notebook or Colab:

python
Copy
Edit
# Run training and evaluation
trainer.train()
trainer.evaluate()
📊 Results

Metric	BERT-only	Hybrid (BERT + GCN)
Accuracy	~72.2%	87.6%
F1 Score	~72.2%	85.5%
Epochs to Peak	10+	3
✅ The hybrid model outperformed the standard BERT baseline with faster convergence and better F1-score.

🔍 Future Work
Extend to multi-label emotion detection.

Replace GCN with Graph Attention Networks (GATs).

Integrate SHAP/LIME for explainability.

Deploy the model via a REST API or Streamlit interface.

Explore multi-modal emotion detection (e.g., text + audio + image).

📄 License
This project is open-source and free to use under the MIT License.

🙌 Acknowledgements
HuggingFace Transformers

PyTorch Geometric

Google GoEmotions Dataset

Feel free to contribute, raise issues, or suggest improvements! 🎯

