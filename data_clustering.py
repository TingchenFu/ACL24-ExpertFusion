'''
Use torch dataset and data loader
'''
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import BitsAndBytesConfig
import faiss
import numpy as np
import json
import os

import argparse
args = argparse.ArgumentParser()
args.add_argument("--dataset_name", type=str, default="")
args.add_argument("--dump_dir", type=str, default="")
args.add_argument("--num_clusters", type=int, default=4)
args.add_argument("--num_iters", type=int, default=1500)
args.add_argument("--model_name_or_path", type=str, default="all-mpnet-base-v2")
args.add_argument("--batch_size", type=int, default=256)
args = args.parse_args()



# Load the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
# tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.padding_side = "right"
# model = AutoModel.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16,load_in_8bit=False,device_map='auto')
# model.eval()

from sentence_transformers import SentenceTransformer
print(os.path.exists(args.model_name_or_path))
model = SentenceTransformer(args.model_name_or_path,cache_folder='')
#model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
print(model)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
model.eval()
model_to_access = model.module if torch.cuda.device_count() > 1 else model
device = model_to_access._target_device
model.to(device)


# Get the hidden states from the model in batches

# Load the sentences
class corpus_dataset(Dataset):
    def __init__(self, dataset_name):
        super(corpus_dataset, self).__init__()
        self.corpus=[]
        self.corpus_length=[0]
        
        for dataset in dataset_name:
            file_path = '/data/'+dataset+'.json'
            for line in open(file_path).readlines():
                data = json.loads(line)
                self.corpus.append(data['output'])  # NOTE: here we use the instruction as the sentence
            self.corpus_length.append(len(self.corpus))
        assert len(self.corpus_length) == len(args.dataset_name)+1

        print(f"Total number of datasets: {len(dataset_name)}")
        print(f"Total number of sentences: {len(self.corpus)}")

    def __getitem__(self, index):
        return self.corpus[index]
    
    def __len__(self):
        return len(self.corpus)
    
    @classmethod
    def collate_fn(cls, batch):
        return batch

if ',' in args.dataset_name:
    args.dataset_name = args.dataset_name.split(',')
else:
    args.dataset_name = [args.dataset_name]
corpus_dataset = corpus_dataset(args.dataset_name)
corpus_dataloader = DataLoader(corpus_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)

# length_idx = np.argsort([len(sen.split(' ')) for sen in corpus])[::-1]
# corpus_sorted = [corpus[idx] for idx in length_idx]

from tqdm import trange
from tqdm import tqdm
hidden_states = []
with torch.no_grad():
    for batch in tqdm(corpus_dataloader):
        # print(batch)
        # exit()
    #for i in trange(0, len(corpus_sorted), args.batch_size, desc="Batches",disable=False):
        #print(f"Processing batch {i} to {i+100}")
        feature = model_to_access.tokenize(batch) 
        # for key in feature:
        #     if isinstance(batch[key], torch.Tensor):
        #         feature[key] = feature[key].to(device)
        out_features = model(feature)
        embeddings = out_features['sentence_embedding']
        embeddings = embeddings.detach()
        embeddings = embeddings.cpu()
        hidden_states.extend(embeddings)

# Concatenate the tensors of hidden states

hidden_states = np.vstack(hidden_states)
print(hidden_states.shape)


# Normalize the hidden states
hidden_states = hidden_states / np.linalg.norm(hidden_states, axis=1)[:, np.newaxis]

# Set up the Faiss index
d = hidden_states.shape[1]
index = faiss.IndexFlatL2(d)
hidden_states = hidden_states.astype(np.float32)
index.add(hidden_states)

# Perform clustering
kmeans = faiss.Kmeans(d, args.num_clusters, niter=args.num_iters, verbose=True)
kmeans.train(hidden_states)


# Assign sentences to clusters according the the trained Faiss index
# Note: This is not the most efficient way to do this, but it is the most straightforward; A more efficient way would be to use the Faiss index to find the nearest neighbors of each hidden state and assign the sentence to the cluster of the nearest neighbor
#length_idx = length_idx.tolist()
#ori_hidden_states = [hidden_states[length_idx.index(idx)] for idx in range(len(corpus_sorted))]
cluster_assignments = kmeans.index.search(np.array(hidden_states), 1)[1].flatten().tolist()

for i in range(len(corpus_dataset.corpus_length)-1):
    output_path = f'{args.model_name_or_path.split("/")[-1]}_fp16_output_cluster{args.num_clusters}/{args.dataset_name[i]}.json'
    output_path = os.path.join(args.dump_dir, output_path)
    if not os.path.exists('/'.join(output_path.split('/')[:-1])):
        os.makedirs('/'.join(output_path.split('/')[:-1]),exist_ok=True)
    json.dump(cluster_assignments[corpus_dataset.corpus_length[i]:corpus_dataset.corpus_length[i+1]],open(output_path,'w'))

# # Print cluster assignments
# for sentence, cluster in zip(corpus, cluster_assignments):
#     print(f"Sentence: '{sentence}' is in cluster {cluster}")