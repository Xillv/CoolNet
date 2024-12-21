import os
import cv2 as cv
import torchvision.transforms as transforms
import torch

from parsing import tree_to_matrix
import pickle
from transformers import (BertModel, BertTokenizer,
                          RobertaModel,RobertaTokenizer,
                          BertForSequenceClassification,
                          RobertaForSequenceClassification, AutoModel, RobertaForMaskedLM,AutoTokenizer)
from load_data import load_data, load_image_dataset, load_imgtext, load_labels, loda_pic_path
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import argparse
import copy
import math
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import vig
from timm.models import create_model
import random




class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, opt, mem_dim, layers, heads):
        super(MultiGraphConvLayer, self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = nn.Dropout(opt)

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()

        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj_list, gcn_inputs):

        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)  # self loop
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))

            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs

            multi_head_list.append(gcn_ouputs)

        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)

        return out




class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn

def select(matrix, top_num):
    batch = matrix.size(0)
    len = matrix.size(1)
    matrix = matrix.reshape(batch, -1)
    maxk, _ = torch.topk(matrix, top_num, dim=1)

    for i in range(batch):
        matrix[i] = (matrix[i] >= maxk[i][-1])
    matrix = matrix.reshape(batch, len, len)
    matrix = matrix + matrix.transpose(-2, -1)

    # selfloop
    for i in range(batch):
        matrix[i].fill_diagonal_(1)

    return matrix


class BertCoAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(BertCoAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states):
        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer



class SentimentClassifier(nn.Module):
    
    def __init__(self, opt):
        super(SentimentClassifier, self).__init__()
        self.opt = opt
        self.head2 = opt.heads2



        self.d_proj1 = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.layers1 = nn.ModuleList()
        self.layers1.append(MultiGraphConvLayer(0.1, opt.hidden_dim, opt.sublayer_c, opt.heads1))
        self.aggregate_W1 = nn.Linear(len(self.layers1) * opt.hidden_dim, opt.hidden_dim)
        self.attn1 = MultiHeadAttention(opt.heads1, opt.hidden_dim)



        self.d_proj2 = nn.Linear(opt.hidden_dim, opt.hidden_dim)
        self.layers2 = nn.ModuleList()
        self.layers2.append(MultiGraphConvLayer(0.1, opt.hidden_dim, opt.sublayer_a, opt.heads2))
        self.aggregate_W2 = nn.Linear(len(self.layers2) * opt.hidden_dim, opt.hidden_dim)
        self.attn2 = MultiHeadAttention(opt.heads2, opt.hidden_dim)

        self.a_t_i = nn.Linear(opt.hidden_dim, 196)
        self.txt2img_attention = BertCoAttention(196, num_attention_heads=4, attention_probs_dropout_prob=0.1)




        self.drop_output = nn.Dropout(opt.drop_output)
        self.MLP = nn.Linear(196, opt.polarities_dim)


    def forward(self, text_g, img_g, text_emd, img_emd, pic):


        layer_list1 = []
        word_bert = self.d_proj1(text_emd)
        for i in range(len(self.layers1)):
            attn_tensor = self.attn1(word_bert, word_bert)
            attn_tensor = torch.sum(attn_tensor, dim=1)
            attn_tensor = select(attn_tensor, 3) * attn_tensor
            attn_adj_list1 = torch.split(attn_tensor, 1, dim=1)
            outputs1 = self.layers1[i](attn_adj_list1, word_bert)
            layer_list1.append(outputs1)
        aggregate_out1 = torch.cat(layer_list1, dim=2)
        com1 = self.aggregate_W1(aggregate_out1)

        layer_list_ = []
        word_bert = self.d_proj1(text_emd)
        for i in range(len(self.layers1)):
            attn_adj_list_ = torch.split(text_g, 1, dim=1)
            outputs_ = self.layers1[i](attn_adj_list_, word_bert)
            layer_list_.append(outputs_)
        aggregate_out_ = torch.cat(layer_list_, dim=2)
        com2 = self.aggregate_W1(aggregate_out_)


        layer_list2 = []
        word_cap= self.d_proj2(img_emd)
        for i in range(len(self.layers2)):
            attn_tensor = self.attn2(word_cap, word_cap)
            attn_tensor = torch.sum(attn_tensor, dim=1)
            attn_tensor = select(attn_tensor, 3) * attn_tensor
            attn_adj_list2 = torch.split(attn_tensor, 1, dim=1)
            outputs2 = self.layers2[i](attn_adj_list2, word_cap)
            layer_list2.append(outputs2)
        aggregate_out2 = torch.cat(layer_list2, dim=2)
        com_cap1 = self.aggregate_W2(aggregate_out2)

        layer_list_2 = []
        word_cap= self.d_proj2(img_emd)
        for i in range(len(self.layers2)):
            attn_adj_list_2 = torch.split(img_g, 1, dim=1)
            outputs_2 = self.layers2[i](attn_adj_list_2, word_cap)
            layer_list_2.append(outputs_2)
        aggregate_out_2 = torch.cat(layer_list_2, dim=2)
        com_cap2 = self.aggregate_W2(aggregate_out_2)


        Xcom = (com1 + com2)/2

        Icom = (com_cap1 + com_cap2)/2

        text_cap_emb = torch.cat([Icom, Xcom], dim=1)

        text_cross = self.a_t_i(text_cap_emb)

        B, C, W, H = pic.shape
        img_fea = pic.view(B, C, -1)
        cross_output_layer = self.txt2img_attention(text_cross, img_fea)

        mean = torch.mean(cross_output_layer, 1)
        emb = self.drop_output(F.relu(mean))


        output = self.MLP(emb)

        return output

def create_tensor(x, shape, device):
    x = torch.Tensor(x)
    if x.size(0) < shape[0]:
        x = torch.cat((x, torch.zeros((shape[0] - x.size(0), x.size(1)))), dim=0)
    if x.size(1) < shape[1]:
        x = torch.cat((x, torch.zeros((x.size(0), shape[1] - x.size(1)))), dim=1)
    assert x.size() == shape
    return x.to(device)

class TwitterDataset(data.Dataset):
    
    def __init__(self, text_g, text_emd, img_g, img_emd, labels, device):
        super().__init__()
        text_dim, img_dim = 0, 0
        for text_e, img_e in zip(text_emd, img_emd):
            text_dim = max(text_dim, len(text_e))
            img_dim = max(img_dim, len(img_e))
        emd_dim = len(text_emd[0][0])
        self.labels =  [torch.tensor(x).to(device) for x in labels]
        self.text_g = [create_tensor(x, (text_dim, text_dim), device) for x in text_g]
        self.text_emd = [create_tensor(x, (text_dim, emd_dim), device) for x in text_emd]
        self.img_g = [create_tensor(x, (img_dim, img_dim), device) for x in img_g]
        self.img_emd = [create_tensor(x, (img_dim, emd_dim), device) for x in img_emd]
        
        
    def __getitem__(self, index):
        return self.text_g[index], self.text_emd[index], self.img_g[index], self.img_emd[index], self.labels[index]
    
    def __len__(self):
        return len(self.text_emd)

class TwitterDatasetWithPic(data.Dataset):
    
    def __init__(self, text_g, text_emd, img_g, img_emd, labels, pic, device):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])
        text_dim, img_dim = 0, 0
        for text_e, img_e in zip(text_emd, img_emd):
            text_dim = max(text_dim, len(text_e))
            img_dim = max(img_dim, len(img_e))
        emd_dim = len(text_emd[0][0])
        self.labels =  [torch.tensor(x).to(device) for x in labels]
        self.text_g = [create_tensor(x, (text_dim, text_dim), device) for x in text_g]
        self.text_emd = [create_tensor(x, (text_dim, emd_dim), device) for x in text_emd]
        self.img_g = [create_tensor(x, (img_dim, img_dim), device) for x in img_g]
        self.img_emd = [create_tensor(x, (img_dim, emd_dim), device) for x in img_emd]
        self.pic = [self.transform(cv.imread(p)) for p in pic]
        
        
    def __getitem__(self, index):
        return self.text_g[index], self.text_emd[index], self.img_g[index], self.img_emd[index], self.labels[index], self.pic[index]
    
    def __len__(self):
        return len(self.text_emd)

 
def get_word_emd(model, tokenizer, word_data):
    model.eval()
    embedding = []
    with torch.no_grad():
        for sentence, _, mapping in tqdm(word_data):
            text1 = ' '.join(sentence[:-1])
            text2 = sentence[-1]
            mapping = np.array(mapping, dtype=np.int32)
            encoded_input = tokenizer(text1, text2, return_tensors='pt').to('cuda')
            output = model(**encoded_input)
            emd = output[0]
            emd_dim = emd.shape[-1]
            emd = emd.view(-1, emd_dim)
            word_emd_list = []
            for i, word in enumerate(sentence):
                word_emd = emd[mapping == i]
                word_emd = torch.mean(word_emd, dim=0)
                word_emd_list.append(word_emd.cpu().numpy())

            embedding.append(word_emd_list)
    return embedding

def get_img_emd(model, tokenizer, img_data):
    model.eval()
    img_embedding = []
    with torch.no_grad():
        for sentence, _, mapping in tqdm(img_data):
                text = ' '.join(sentence)
                mapping = np.array(mapping, dtype=np.int32)
                encoded_input = tokenizer(text, return_tensors='pt').to('cuda')
                output = model(**encoded_input)
                emd = output[0]
                emd_dim = emd.shape[-1]
                emd = emd.view(-1, emd_dim)
                word_emd_list = []
                for i, word in enumerate(sentence):
                    word_emd = emd[mapping == i]
                    word_emd = torch.mean(word_emd, dim=0)
                    word_emd_list.append(word_emd.cpu().numpy())
                img_embedding.append(word_emd_list)
    return img_embedding

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sublayer_c', default=4, type=int)
    parser.add_argument('--sublayer_a', default=4, type=int)
    parser.add_argument('--heads1', default=3, type=int)
    parser.add_argument('--heads2', default=3, type=int)
    parser.add_argument('--heads3', default=3, type=int)
    parser.add_argument('--drop_output', default=0.1, type=int)
    parser.add_argument('--drop_CG', default=0.1, type=int)
    parser.add_argument('--drop_G1', default=0.1, type=int)
    parser.add_argument('--hidden_dim', default=768, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--seed', default=105, type=int)#98!!!!
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--num_epoch', default=35, type=int)
    parser.add_argument('--NUM_RUNS', default=10, type=int)
    parser.add_argument('--MAX_LEN', default=80, type=int)
    parser.add_argument('--save_dir', default='out', type=str)
    parser.add_argument('--dataset', choices=[15, 17], default=15, type=int)
    parser.add_argument('--pretrain', type=str)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    tree_file_path = 'roberta_dist_{}/11_tree.pkl'.format(args.dataset)
    img_tree_file_path = 'roberta_dist_img_{}/11_tree.pkl'.format(args.dataset)
    with open(tree_file_path, 'rb') as f:
        tree = pickle.load(f)
        matrix = tree_to_matrix(tree)
    with open(img_tree_file_path, 'rb') as f2:
        img_tree = pickle.load(f2)
        img_matrix = tree_to_matrix(img_tree)
        
    test_tree_file_path = 'roberta_dist_test_{}/11_tree.pkl'.format(args.dataset)
    test_img_tree_file_path = 'roberta_dist_img_test_{}/11_tree.pkl'.format(args.dataset)
    with open(test_tree_file_path, 'rb') as f3:
        test_tree = pickle.load(f3)
        test_matrix = tree_to_matrix(test_tree)
    with open(test_img_tree_file_path, 'rb') as f4:
        test_img_tree = pickle.load(f4)
        test_img_matrix = tree_to_matrix(test_img_tree)
        
    model = RobertaForSequenceClassification.from_pretrained('finetune/roberta_{}/final/'.format(args.dataset)).roberta


    encoder = create_model("vig_s_224_gelu")
    state_dict = torch.load("vig_s_80.6.pth")
    encoder.load_state_dict(state_dict, strict=False)
    model.to('cuda')
    model.eval()
    tokenizer = RobertaTokenizer.from_pretrained('./model/roberta-base-cased/')
    dataset_name = 'twitter20{}'.format(args.dataset)
    text_data = load_data(dataset_name,
                          tokenizer)
    img_data = load_image_dataset(dataset_name,
                                  tokenizer)
    labels = load_labels(dataset_name)
    pic = loda_pic_path(dataset_name)
    embedding = get_word_emd(model,
                             tokenizer,
                             text_data)
    img_embedding = get_img_emd(model,
                                tokenizer,
                                img_data)
    
    test_text_data = load_data(dataset_name,
                               tokenizer,
                               mode='test')
    test_img_data = load_image_dataset(dataset_name,
                                       tokenizer,
                                       mode='test')
    test_labels = load_labels(dataset_name,
                              mode='test')
    test_pic = loda_pic_path(dataset_name)                          
    test_embedding = get_word_emd(model,
                                  tokenizer,
                                  test_text_data)
    test_img_embedding = get_img_emd(model,
                                     tokenizer,
                                     test_img_data)
    
    device = torch.device('cuda')
    train_dataset = TwitterDatasetWithPic(matrix,
                                        embedding,
                                        img_matrix,
                                        img_embedding,
                                        labels,
                                        pic,
                                        device)
    test_datalset = TwitterDatasetWithPic(test_matrix,
                                   test_embedding,
                                   test_img_matrix,
                                   test_img_embedding,
                                   test_labels,
                                   test_pic,
                                   device)

    model = SentimentClassifier(args).to(device)
    if args.pretrain:
        model.load_state_dict(torch.load(args.pretrain), strict=False)
    dataloader = DataLoader(dataset=train_dataset,
                            batch_size=8,
                            shuffle=False)
    test_dataloader = DataLoader(dataset=test_datalset,
                                 batch_size=32,
                                 shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(),
                            lr=0.00003)


    loss_f = nn.CrossEntropyLoss()
    log_f = open(os.path.join(args.save_dir, 'log.txt'), 'w')
    max_acc, max_id, max_f1 = 0, 0, 0
    best_model_state_dict = model.state_dict()
    for epoch_id in range(args.num_epoch): 
        tot_loss = 0
        model.train()
        encoder.train()
        encoder.zero_grad()
        for batch_id, x in enumerate(dataloader):
            optimizer.zero_grad()
            text_g, text_emd, img_g, img_emd, target, pic = x
            with torch.no_grad():
                imgs_f = encoder(pic).to('cuda')
            out = model(text_g,
                        img_g,
                        text_emd,
                        img_emd,
                        imgs_f)
            loss = loss_f(out, target)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * out.size(0)
        print('epoch:{} \t tot_loss:{} \t lr:{}\n'.format(epoch_id,
                                                          tot_loss,
                                                          optimizer.param_groups[0]['lr']))  
        # scheduler.step()
                  
        if (epoch_id + 1) % 1 == 0 or epoch_id + 1 == args.num_epoch:
            torch.save(model.state_dict(), os.path.join(args.save_dir, '{}.pth'.format(epoch_id)))
            model.eval()
            encoder.eval()
            pred, gt = [], []
            with torch.no_grad():
                for batch_id, x in enumerate(test_dataloader):
                    text_g, text_emd, img_g, img_emd, target, test_pic = x
                    with torch.no_grad():
                        imgs_t = encoder(test_pic).to('cuda')
                    out = model(text_g, img_g, text_emd, img_emd, imgs_t)
                    pred.append(out.argmax(dim=1).cpu().numpy())
                    gt.append(target.cpu().numpy())
            pred = np.concatenate(pred)
            gt = np.concatenate(gt)
            acc = accuracy_score(gt, pred)
            precision, recall, f1, _ = precision_recall_fscore_support(gt,
                                                                       pred,
                                                                       average='macro',
                                                                       zero_division=0)
            if acc > max_acc:
                max_acc =acc
                max_f1 = f1
                
                max_id = epoch_id
                best_model_state_dict = model.state_dict()
            print('epcho:{}\tacc:{:.4f}\tprecision:{:.4f}\trecall:{:.4f}\tf1:{:.4f}\n'.format(epoch_id, acc, precision, recall, f1)) 
            print('epcho:{}\tacc:{:.4f}\tprecision:{:.4f}\trecall:{:.4f}\tf1:{:.4f}\n'.format(epoch_id, acc, precision, recall, f1), file=log_f)
    print('best epoch:{}\tacc:{:.4f}\tf1:{:.4f}\n'.format(max_id, max_acc, max_f1))
    print('best epoch:{}\tacc:{:.4f}\tf1:{:.4f}\n'.format(max_id, max_acc, max_f1), file=log_f)
    torch.save(best_model_state_dict, os.path.join(args.save_dir, 'best.pth'))
    log_f.close()
