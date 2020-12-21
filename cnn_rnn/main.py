# -*- coding: utf-8 -*-
import sys
import argparse
import os
import time
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import Levenshtein

from model import SimpleCNN,ParallelCNN,SiameseCNN
# from load_data import load_data
from keras.utils.np_utils import to_categorical

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=100,
    help='Batch size for mini-batch training and evaluating. Default: 100')
parser.add_argument('--num_epochs', type=int, default=50,
    help='Number of training epoch. Default: 20')
parser.add_argument('--learning_rate', type=float, default=1e-3,
    help='Learning rate during optimization. Default: 1e-3')
parser.add_argument('--drop_rate', type=float, default=0.5,
    help='Drop rate of the Dropout Layer. Default: 0.5')
parser.add_argument('--is_test', action = "store_true",
    help='True to train and False to inference. Default: False')
parser.add_argument('--inference_version', type=int, default=0,
    help='The version for inference. Set 0 to use latest checkpoint. Default: 0')
parser.add_argument('--pathname',type=str,default='./data/',
    help='Pathname of training data')
parser.add_argument('--train_dir',type=str,default='./train',
    help='Train Direction')
parser.add_argument('--dataset',type=str,default='NLV',
    help='Name of dataset')
parser.add_argument('--model',type=str,default='cnn',
    help='Choose an ML Model')
parser.add_argument('--name',type=str,default='cnn',
    help='Name of this model')
parser.add_argument('--data_dir',type=str,default='./data/GIL',
    help='Test path')
parser.add_argument('--savepred',type=str,default='./predict.txt',
    help='Prediction file')

args = parser.parse_args()


# bases = ['A', 'C', 'G', 'U']
bases = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','O','B']
base_dict = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
             'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
             'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
             'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
             'O': 20, 'B': 21}
amino_num=22
#base_dict = {}
bases_len = len(bases)
max_len = 376
bags = []
def convert_to_index(str,word_len):
   '''
   convert a sequence 'str' of length 'word_len' into index in 0~4^(word_len)-1
   '''
   output_index = 0
   for i in range(word_len):
       output_index = output_index * bases_len + base_dict[str[i]]
   return output_index

def shuffle(X, y, shuffle_parts):
    chunk_size = int(len(X) / shuffle_parts)
    shuffled_range = list(range(chunk_size))

    X_buffer = np.copy(X[0:chunk_size])
    y_buffer = np.copy(y[0:chunk_size])

    for k in range(shuffle_parts):
        np.random.shuffle(shuffled_range)
        for i in range(chunk_size):
            X_buffer[i] = X[k * chunk_size + shuffled_range[i]]
            y_buffer[i] = y[k * chunk_size + shuffled_range[i]]

        X[k * chunk_size:(k + 1) * chunk_size] = X_buffer
        y[k * chunk_size:(k + 1) * chunk_size] = y_buffer

    return X, y


def split_dataset(posX,posY,negX,negY,num=5):
    '''
    5-fold cross validation
    '''
    fold_list=[]
    step_pos = len(posY) // num
    step_neg = len(negY) // num
    for i in range(num):
        X_data=[]
        Y_data=[]
        if i < num - 1:
            X_data = np.vstack([posX[i * step_pos:(i + 1) * step_pos,:],negX[i * step_neg:(i + 1) * step_neg,:]])
            Y_data = np.hstack([posY[i * step_pos:(i + 1) * step_pos],negY[i * step_neg:(i + 1) * step_neg]])
        else:
            X_data = np.vstack([posX[i * step_pos:,:],negX[i * step_neg:,:]])
            Y_data = np.hstack([posY[i * step_pos:],negY[i * step_neg:]])
        X_data,Y_data = shuffle(X_data, Y_data, 1)
        fold_list.append((X_data,Y_data))

    return fold_list

def extract_features(line):
   '''
   extract features from a sequence of RNA
   
   To do: alternative ways to generate features can be used. 
   '''
   core_seq = line
   for i in 'agctu\n':
      core_seq = core_seq.replace(i, '')
   core_seq = core_seq.replace('T','U')
   core_seq = core_seq.replace('N','')
   final_output=[]
   for word_len in [1,2,3]:
      output_count_list = [0 for i in range(bases_len ** word_len)]
      for i in range(len(core_seq)-word_len+1):
         output_count_list[convert_to_index(core_seq[i:i+word_len],word_len)] +=1
      final_output.extend(output_count_list)
   return final_output

def get_bags(s,k,base):
    if len(s) == k-1:
        for i in base:
            bags.append(s+i)
        return
    else:
        for i in base:
            next = s + i
            get_bags(next,k,base)
    return

def get_kmers(sequence,k):
    begin = 0
    end = begin+k
    k_list = []
    while end <= len(sequence):
        k_list.append(sequence[begin:end])
        begin+=1
        end+=1
    return k_list

def one_hot_features(line):
    '''
    extract one_hot features from a sequence of polypeptide
    '''
    global max_len
    '''
    core_seq = line
    for i in 'agctu\n':
        core_seq = core_seq.replace(i, '')
    core_seq = core_seq.replace('T','U')
    core_seq = core_seq.replace('N','')
    '''
    # line = core_seq
    line = line.upper()
    line = line.replace('X', '')
    indexes = []
    # k_mers = get_kmers(line,3)
    for peptide in line:
        indexes.append(base_dict[peptide])
    indexes = np.array(indexes)
    one_hot = to_categorical(indexes, num_classes=None)
    #for i in range(150,one_hot.shape[0]-150):
        #one_hot[i] = one_hot[i] * 5
        # print(one_hot[i])
    if one_hot.shape[1] != amino_num:
        add = np.zeros((one_hot.shape[0],amino_num-one_hot.shape[1]))
        one_hot = np.append(one_hot,add,axis=1)
    if one_hot.shape[0] < max_len:
        padding = np.zeros((max_len-one_hot.shape[0],amino_num))
        one_hot = np.append(one_hot,padding,axis=0)
    # print(one_hot.shape)
    # print(line)
    # print(one_hot)
    return one_hot

def get_maxlen(filep, filen):
    maxlen = 0
    f1=open(filep,'r')
    lines=f1.readlines()
    f1.close()
    f2 = open(filen, 'r')
    line2 = f2.readlines()
    f2.close()
    lines.extend(line2)
    for line in lines:
        line = line.strip('\n').strip('\r')
        regexp = re.compile(r'[^A-Za-z]')
        if regexp.search(line):
            continue
        else:
            cur_len = len(line)
            if cur_len > maxlen:
                maxlen = cur_len
    return maxlen

def load_data(filename,check=False,savecheck='check'):
    '''
    use the extract_features function to extract features for all sequences in the file specified by 'filename'
    '''
    print('Processing ',filename)
    start=time.time()
    total_output=[]
    valid=[]
    f=open(filename,"r")
    lines=f.readlines()
    f.close()
    global max_len
    print(max_len)
    for line in lines:
        line=line.strip('\n').strip('\r')
        regexp = re.compile(r'[^A-Za-z]')
        if regexp.search(line):
            valid.append(0)
            continue
        else:
            valid.append(1)
            total_output.append(one_hot_features(line))
    output_arr=np.array(total_output)
    if (check):
        np.save(savecheck,np.array(valid))
    print(output_arr.shape)
    end=time.time()
    print ('Finished loading in',end-start,'s\n')
    return output_arr


def train_epoch(model, X, y, optimizer): # Training Process
    model.train()
    loss, acc, roc = 0.0, 0.0, 0.0
    st, ed, times = 0, args.batch_size, 0
    count = 0
    while st < len(X) and ed <= len(X):
        count += 1
        print(count,end='\r')
        optimizer.zero_grad()
        X_batch, y_batch = torch.from_numpy(X[st:ed]).to(device), torch.from_numpy(y[st:ed]).to(device)
        if args.model == 'siamese':
            Nx,Ny = edit_distance(X_batch,y_batch)
            loss_,acc_,roc_ = model(X_batch,y_batch,Nx,Ny)
        else:
            loss_, acc_, roc_ = model(X_batch, y_batch)
        
        loss_.backward()
        optimizer.step()

        loss += loss_.cpu().data.numpy()
        acc += acc_.cpu().data.numpy()
        roc += roc_
        st, ed = ed, ed + args.batch_size
        times += 1
    loss /= times
    acc /= times
    roc /= times
    return acc, loss, roc


def valid_epoch(model, X, y): # Valid Process
    model.eval()
    loss, acc, roc = 0.0, 0.0, 0.0
    st, ed, times = 0, args.batch_size, 0
    while st < len(X) and ed <= len(X):
        X_batch, y_batch = torch.from_numpy(X[st:ed]).to(device), torch.from_numpy(y[st:ed]).to(device)
        if args.model == 'siamese':
            Nx,Ny = edit_distance(X_batch,y_batch)
            loss_,acc_,roc_ = model(X_batch,y_batch,Nx,Ny)
        else:
            loss_, acc_, roc_ = model(X_batch, y_batch)
        # loss_, acc_, roc_= model(X_batch, y_batch)

        loss += loss_.cpu().data.numpy()
        acc += acc_.cpu().data.numpy()
        roc += roc_
        st, ed = ed, ed + args.batch_size
        times += 1
    loss /= times
    acc /= times
    roc /= times
    return acc, loss, roc


def inference(model, X): # Test Process
    model.eval()
    pred_ = model(torch.from_numpy(X).to(device))
    return pred_.cpu().data.numpy()

def to_seq(one_hot):
    seq = ''
    # print(one_hot.shape)
    # print(one_hot)
    for i in range(one_hot.shape[0]):
        num = np.where(one_hot[i,:]==1)
        # print(num[0].shape)
        if num[0].shape[0]==0:
            break
        char = bases[int(num[0])]
        seq += char
    # print(seq)
    return seq

def edit_distance(X,Y):
    length = X.shape[0]
    seq_list = []
    dists = 100*np.ones((X.shape[0],X.shape[0]))
    count = 0
    for i in range(length):
        sequence = to_seq(X[i])
        seq_list.append(sequence)
    for i in range(len(seq_list)):
        for j in range(len(seq_list)):
            count += 1
            # print(count)
            if i == j:
                continue
            a = seq_list[i]
            b = seq_list[j]
            dist = Levenshtein.distance(a, b)
            dists[i,j] = dist
    # print(dists)
    min = dists.argmin(axis=1)
    # print(np.min(dists[10,:]))
    # print(dists[10,min[10]])
    Nx = X[min,:,:]
    Ny = Y[min]
    # print(Nx.shape)
    # print(Ny.shape)
    return Nx,Ny

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)
    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)

    if not args.is_test:

        pos_filename=os.path.join(args.pathname,'new_'+args.dataset+'_positive.txt')
        neg_filename=os.path.join(args.pathname,'new_'+args.dataset+'_negative.txt')
        #global max_len
        max_len = get_maxlen(pos_filename,neg_filename)
        pos_trainX=load_data(pos_filename)
        pos_trainY=np.ones(len(pos_trainX))
        neg_trainX=load_data(neg_filename)
        neg_trainY=np.zeros(len(neg_trainX))

        pos_trainX, pos_trainY=shuffle(pos_trainX,pos_trainY,1)
        neg_trainX, neg_trainY=shuffle(neg_trainX,neg_trainY,1)
        fold_list=split_dataset(pos_trainX,pos_trainY,neg_trainX,neg_trainY,5)

        #if args.model == "rnn":
        #    model = LSTM(max_len,drop_rate=args.drop_rate)
        #else:
        #    model = CNN(max_len,drop_rate=args.drop_rate)
        #model.to(device)
        # f = open("../result/cnn_nobn.txt",'w')
        # f.write("TrainAcc\tTrainLoss\tValAcc\tValLoss\n")
        #optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        # model_path = os.path.join(args.train_dir, 'checkpoint_%d.pth.tar' % args.inference_version)
        # if os.path.exists(model_path):
        # 	cnn_model = torch.load(model_path)



        avg_val_acc=0.0
        avg_val_roc=0.0

        best_val_acc_global = 0.0
        best_val_roc_global = 0.0

        for i in range(5):
            if args.model == "parallel":
                model = ParallelCNN(max_len, drop_rate=args.drop_rate)
            elif args.model == 'cnn':
                model = SimpleCNN(max_len, drop_rate=args.drop_rate)
            elif args.model == 'siamese':
                model = SiameseCNN(max_len,drop_rate=args.drop_rate)
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

            pre_losses = [1e18] * 3
            best_val_acc = 0.0
            best_val_roc = 0.0

            X_train=[]
            y_train=[]
            for j in range(5):
                if(j!=i):
                    X_train.extend(fold_list[j][0])
                    y_train.extend(fold_list[j][1])
            X_train=np.array(X_train)
            y_train=np.array(y_train)
            X_val=fold_list[i][0]
            y_val=fold_list[i][1]

            for epoch in range(1, args.num_epochs+1):
                start_time = time.time()
                #print(len(X_train)) 
                train_acc, train_loss,train_roc = train_epoch(model, X_train, y_train, optimizer)
                X_train, y_train = shuffle(X_train, y_train, 1)
                val_acc, val_loss, val_roc = valid_epoch(model, X_val, y_val)

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_val_roc = val_roc
                    best_epoch = epoch
                    # test_acc, test_loss = valid_epoch(model, X_test, y_test)
                    if best_val_acc >= best_val_acc_global:
                        best_val_acc_global=best_val_acc
                        best_val_roc_global=best_val_roc
                        with open(os.path.join(args.train_dir, 'checkpoint_{}.pth.tar'.format(args.name)), 'wb') as fout:
                            torch.save(model, fout)
                # 	with open(os.path.join(args.train_dir, 'checkpoint_0.pth.tar'), 'wb') as fout:
                # 		torch.save(model, fout)

                epoch_time = time.time() - start_time
                print("Epoch " + str(epoch) + " of " + str(args.num_epochs) + " took " + str(epoch_time) + "s")
                print("  learning rate:                 " + str(optimizer.param_groups[0]['lr']))
                print("  training loss:                 " + str(train_loss))
                print("  training accuracy:             " + str(train_acc))
                print("  training roc:                  " + str(train_roc))
                print("  validation loss:               " + str(val_loss))
                print("  validation accuracy:           " + str(val_acc))
                print("  validation roc:                " + str(val_roc))
                print("  best epoch:                    " + str(best_epoch))
                print("  best validation accuracy:      " + str(best_val_acc))
                print("  best validation auc            " + str(best_val_roc))
                if train_loss > max(pre_losses):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9995
                pre_losses = pre_losses[1:] + [train_loss]

            avg_val_acc += best_val_acc
            avg_val_roc += best_val_roc
        print("average acc:   "+str(avg_val_acc/5))
        print("average roc:   "+str(avg_val_roc/5))


    else:
        print ("Predicting",args.data_dir)
        if args.model == 'parallel':
            model = ParallelCNN(max_len,drop_rate=0.5)
        else:
            model = SimpleCNN(max_len,drop_rate=0.5)
        model.to(device)
        model_path = os.path.join(args.train_dir, 'checkpoint_%s.pth.tar' % args.name)
        if os.path.exists(model_path):
            model = torch.load(model_path)
        start=time.time()

        if args.savepred is not None:
            fout=open(args.savepred,'w')
        try:
            for line in open(args.data_dir):
                if line[0]=='>':
                    if args.savepred is not None:
                        fout.write(line)
                        continue
                elif ('n' in line or 'N' in line):
                    if args.savepred is not None:
                        fout.write('Error!\n')
                else:
                    line=line.strip('\n').strip('\r')
                    testX=np.array(one_hot_features(line.strip('\n').strip('\r')))
                    pred = inference(model, testX)[0]
                    if args.savepred is not None:
                        fout.write('%f\n'%float(pred[1]))
        finally:
            if args.savepred is not None:
                fout.close()



        # count = 0
        # for i in range(len(X_test)):
        # 	# test_image = X_test[i].reshape((1, 3, 32, 32))
        # 	result = inference(model, i)[1]
