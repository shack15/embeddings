from transformers import AutoTokenizer, AutoModel
import torch
from embeddings.embedding import Embedding

class BGEEmbedding(Embedding):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
        self.model = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')
        self.model.eval()

    def emb(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings