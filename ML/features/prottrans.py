# @title Import dependencies. { display-mode: "form" }
# Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50)
import transformers
import torch
import re
import pickle

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))

# # @title Load encoder-part of ProtT5 in half-precision. { display-mode: "form" }
# # Load ProtT5 in half-precision (more specifically: the encoder-part of ProtT5-XL-U50 in half-precision)
# transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
# print("Loading: {}".format(transformer_link))
# model = T5EncoderModel.from_pretrained(transformer_link)
# model.full() if device == 'cpu' else model.half()  # only cast to full-precision if no GPU is available
# model = model.to(device)
# model = model.eval()
# tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)
# print("Loading OK")

# 指定模型的本地路径
# C:\Users\LENOVO\Downloads\prot_t5_xl_half_uniref50-enc 使用的时候不需要加prot_t5_xl_half_uniref50-enc
model_path = "prot_t5_xl_half_uniref50-enc"

# 加载模型和分词器
print("Loading model from local path: {}".format(model_path))
model = transformers.T5EncoderModel.from_pretrained(model_path)
model = model.to(device)
model = model.eval()
tokenizer = transformers.T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
print("Loading OK")

import argparse

parser = argparse.ArgumentParser(description='Generate prottrans embeding of fasta sequences')
parser.add_argument("input", type=str, action="store", help='input  of fasta format file')
parser.add_argument("output", type=str, action="store", help='output of pkl file')
args = parser.parse_args()

from Bio import SeqIO

# 优化数据处理
# seqs = [(i.id, str(i.seq)) for i in SeqIO.parse(args.input, "fasta")]
seqs = [(record.id, str(record.seq)) for record in SeqIO.parse(args.input, "fasta")]
seqs_ids = [_[0] for _ in seqs]
seqs_seq = [_[1] for _ in seqs]

sequence_examples = seqs_seq
# sequence_examples = ["PRTEINO", "SEQWENCE"]
# this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
# 非标准氨基酸已被去除
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

print("Computing...")
prot_embs = dict()
for idx_prot in range(len(sequence_examples)):
    print(f"{idx_prot}...")
    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer.batch_encode_plus(sequence_examples[idx_prot:idx_prot + 1], add_special_tokens=True,
                                      padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    print("generate embeddings...")
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    print("generate embeddings OK")

    # extract embeddings for the first ([0,:]) sequence in the batch while removing padded & special tokens ([0,:7])
    emb_0 = embedding_repr.last_hidden_state[0, :]  # shape (7 x 1024)
    print(f"Shape of per-residue embedding of first sequences: {emb_0.shape}")
    # do the same for the second ([1,:]) sequence in the batch while taking into account different sequence lengths (
    # [1,:8]) emb_1 = embedding_repr.last_hidden_state[1,:8] # shape (8 x 1024)

    # if you want to derive a single representation (per-protein embedding) for the whole protein
    emb_0_per_protein = emb_0.mean(dim=0)  # shape (1024)

    prot_embs[seqs_ids[idx_prot]] = emb_0_per_protein.cpu().numpy()
    print(f"{idx_prot}/{len(seqs_seq)} length:", len(seqs_seq[idx_prot]), f"Shape : {emb_0_per_protein.shape}")

# # 减少内存分配和重复计算
# prot_embs = {}
# for idx_prot, sequence in enumerate(seqs_seq):
#     # 将嵌入计算移动到循环外部
#     ids = tokenizer.batch_encode_plus([sequence], add_special_tokens=True, padding="longest")
#     input_ids = torch.tensor(ids['input_ids']).to(device)
#     attention_mask = torch.tensor(ids['attention_mask']).to(device)
#     with torch.no_grad():
#         embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
#     emb_0 = embedding_repr.last_hidden_state[0, :]  # shape (7 x 1024)
#     emb_0_per_protein = emb_0.mean(dim=0)  # shape (1024)
#     prot_embs[seqs_ids[idx_prot]] = emb_0_per_protein.cpu().numpy()
#     print(f"{idx_prot}/{len(seqs_seq)} length:", len(sequence), f"Shape : {emb_0_per_protein.shape}")

with open(args.output, "wb") as f:
    pickle.dump(prot_embs, f)

# 运行
# python E:/PyrhonProjects/PPI_Project/ML/standard/features/prottrans.py E:/PyrhonProjects/PPI_Project/ML/standard/Dataset/v_and_h.fasta E:/PyrhonProjects/PPI_Project/ML/standard/features/v_and_h_prottrans.pkl
# 模型需要的内存太大，建议用服务器运行
# nohup python prottrans.py v_and_h.fasta v_and_h_prottrans.pkl > output.log 2> error.log &
