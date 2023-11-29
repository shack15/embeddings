from transformers import AutoTokenizer, AutoModel
import torch
from embeddings.embedding import Embedding

class BGEEmbedding(Embedding):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
        self.model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')
        self.model.eval()

    def emb(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


if __name__ == '__main__':
    from time import time
    emb = BGEEmbedding()
    sentences = ['This is an example sentence', 'Each sentence is converted']
    for s in sentences:
        start = time()
        print('embedding {}'.format(s))
        print(emb.emb([s]))
        print('took {}s'.format(time() - start))