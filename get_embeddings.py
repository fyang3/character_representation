from transformers import BertModel,BertTokenizer,BertForMaskedLM, AutoTokenizer, AutoModel
import time
import numpy as np
import math
import torch
import sys
import os
import csv
import random
from collections import Counter

random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_names = {'bert':'bert-base-cased','bert-wholeword':'bert-base-cased-whole-word-masking','bert-large':'bert-large-cased','bert-large-wholeword':'bert-large-cased-whole-word-masking'}
models = {'bert':BertForMaskedLM,'bert-wholeword':BertModel,'bert-large':BertModel,'bert-large-wholeword':BertModel}
tokenizers = {'bert':BertTokenizer,'bert-wholeword':BertTokenizer,'bert-large':BertTokenizer,'bert-large-wholeword':BertTokenizer}


model_name = model_names[sys.argv[1]]
model = sys.argv[1]
tokenizer = tokenizers[sys.argv[1]].from_pretrained(model_name)
my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(my_device)
chatty = True
criterion = torch.nn.CrossEntropyLoss()
batch_size = 4
log_interval = 5
lr = 0.001
hidden_size = 768

def main():
    
    print("Using model:",model_name,file=sys.stderr)
    
    trainf = sys.argv[2]
    #save_path = sys.argv[3] temporarily disable
    bert = models[model].from_pretrained(model_name)
    bert.to(my_device)
    print("Retreiving word embeddings for training file: ", trainf)
    data = get_data(trainf)[1:] #strip headers
    output = []
    for entry in data:
        embeds = get_embeddings_rev(entry[0], entry[1], layers=None)
        print("word {} in sentence {}, embedding shape of {}".format(entry[1],entry[0],embeds.shape))
        embed_list = embeds.tolist()
        new_entry = [entry[0],entry[1]]
        for num in embed_list:
            new_entry.append(num)
        print("length of new entry: {}, should be 768 + 2 = 770 columns. ".format(len(new_entry)))
        output.append(new_entry)
    #with open('{file_path}.csv'.format(file_path=os.path.join(save_path, trainf+"embeds.csv"), 'w+',new_line="") as csv_file:
    with open(trainf+"embeds.csv",'w') as csvf:
        csvwriter = csv.writer(csvf)
        csvwriter.writerows(output)

    return



def get_data(corpus_name):
    '''process csv file to a list of sentences for embedding retrieval'''
    processed = []
    with open(corpus_name,'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            processed.append([row[0],row[1]])
        return processed

def get_word_idx(sent: str, word: str):
     return sent.split(" ").index(word)
 
 
def get_hidden_states(encoded, token_ids_word, model, layers):
     """Push input IDs through model. Stack and sum `layers` (last four by default).
        Select only those subword token outputs that belong to our word of interest
        and average them."""
     with torch.no_grad():
        output = model(**encoded)
 
     # Get all hidden states
     states = output.hidden_states
     last_hidden_states = states[-1]
     print("last hidden states shape", last_hidden_states.shape)
     #print("last hidden states value: ", last_hidden_states)
     # Stack and sum all requested layers
     output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
     # Only select the tokens that constitute the requested word
     word_tokens_output = output[token_ids_word]
 
     return word_tokens_output.mean(dim=0)
 
 
def get_word_vector(sent, idx, tokenizer, model, layers):
     """Get a word vector by first tokenizing the input sentence, getting all token idxs
        that make up the word of interest, and then `get_hidden_states`."""
     encoded = tokenizer.encode_plus(sent, return_tensors="pt")
     # get all token idxs that belong to the word of interest
     token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
 
     return get_hidden_states(encoded, token_ids_word, model, layers)
 
 
def get_embeddings_rev(sent, word, layers=None): #meta-function to retrive contextual embeddings for a given sentence
     # Use last four layers by default
     layers = [-4, -3, -2, -1] if not layers else layers
     tokenizer = AutoTokenizer.from_pretrained(model_name) #should change to accomodate more models
     model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
  
     idx = get_word_idx(sent, word)
     word_embedding = get_word_vector(sent, idx, tokenizer, model, layers)
     
     return word_embedding 
 
 
if __name__ == '__main__':
     main()
    


def get_bert_preds(bert,emb_line,forms,seq_length=0): #from Carolyn's scripts, and modifying
    pre = emb_line[0]
    post = emb_line[1]
    if seq_length == 0:
      tokens = tokenizer([pre+' [MASK] '+post],padding=True, return_tensors="pt")
      tens = [tokens['input_ids'].to(my_device),tokens['token_type_ids'].to(my_device),tokens['attention_mask'].to(my_device)]
    #else:
    #    tokens = tokenizer([pre+' [MASK] '+post],padding='max_length',max_length=seq_length, return_tensors="pt")
    #    tens = [tokens['input_ids'].to(my_device),tokens['token_type_ids'].to(my_device),tokens['attention_mask'].to(my_device)]
    target_idx = len(tokens['input_ids'][0])
    word_idx = len(tokenizer(pre,return_tensors="pt")["input_ids"][0])-1 #minus 1 is necessary to match indexing in new_eval_gen_bert.py. Difference in tokenization?
    word_ids = tokenizer.convert_tokens_to_ids(forms)
    with torch.no_grad():
        res=bert(*tens,output_hidden_states=True)
    logits = res[0][0,word_idx]
    probs = torch.nn.functional.log_softmax(logits,-1)
    scores = probs[word_ids]
    
    # Get embedding for [MASK] token
    hidden_states = res[1][1:]
    layer_hidden_t = torch.stack(hidden_states,dim=0)
    embeds = layer_hidden_t[:,:,word_idx,:]
    embed = torch.reshape(embeds,(-1,))
    return (embed,[float(x.item()) for x in scores])