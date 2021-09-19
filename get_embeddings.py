from transformers import BertModel,BertTokenizer,BertForMaskedLM
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
# comparison = sys.argv[2]
chatty = True
criterion = torch.nn.CrossEntropyLoss()
batch_size = 4
log_interval = 5
lr = 0.001
hidden_size = 768

def main():
    
    print("Using model:",model_name,file=sys.stderr)
    
    trainf = sys.argv[3]
    traintype = sys.argv[4]
    save_path = sys.argv[5]
    # datasize = sys.argv[6]
    mask_lookup = sys.argv[7]=="mask"
    bert = models[model].from_pretrained(model_name)
    bert.to(my_device)
    print("Retreiving word embeddings for training file: ", trainf)
    tr_full_data,tr_emb_data,tr_labels = get_data(trainf,traintype)
    tr_size = len(tr_full_data)
    outlines = []
    for l in range(len(tr_emb_data)):
        forms = tr_full_data[l][-6:-1]
        correct = tr_full_data[l][1]
        if not mask_lookup:
            tar_embed,_, _ = get_bert_word_embed(bert,tr_emb_data[l],correct)
        mask_embed,scores = get_bert_preds(bert,tr_emb_data[l],forms)
        outcomes = score_preds(scores,forms,correct)
        if mask_lookup:
            final_emb = mask_embed
        else:
            final_emb = tar_embed
        nl = make_line(outcomes,final_emb)
        outlines.append(nl)
    with open(datasize+"embeds.csv",'w') as csvf:
        csvwriter = csv.writer(csvf, delimiter='\t')
        for l in outlines:
            pl = '\t'.join(l)+'\n'
            #print(pl)
            csvwriter.writerow(l)
    return






def get_bert_preds(bert,emb_line,forms,seq_length=0):
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