from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from embeddings.embedding import Embedding

class MiniLMEmbedding(Embedding):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def emb(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


if __name__ == '__main__':
    from time import time
    emb = MiniLMEmbedding()
    sentences = ['This is an example sentence', 'Each sentence is converted']
    for s in sentences:
        start = time()
        print('embedding {}'.format(s))
        print(emb.emb([s]))
        print('took {}s'.format(time() - start))
