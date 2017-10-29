from __future__ import print_function
import numpy as np
import pickle

class Hyperparams:
    batch_size = 64
    embed_dim = 300
    hidden_dim = 1000
    ctxlen = 100 # For inference
    maxlen = 400 # For training
    vallen = 640 # For validation

# Map each character with the indicated index
def load_vocab():
    vocab = 'E,-0123456789'
    digit2idx = {digit:idx for idx, digit in enumerate(vocab)}
    idx2digit = {idx:digit for idx, digit in enumerate(vocab)}
    
    return digit2idx, idx2digit 

def create_train_data():
    digit2idx, idx2digit = load_vocab()

    train_lines = [line.split('"')[1] for line in open('../data/train.csv', 'r').read().splitlines()[1:]]
    test_lines = [line.split('"')[1] for line in open('../data/test.csv', 'r').read().splitlines()[1:]]
            
    xs0, xs1, xs2, xs3 = [], [], [], []
    ys0, ys1, ys2, ys3 = [], [], [], []
    for i, line in enumerate(train_lines + test_lines):
        digits = line[-Hyperparams.maxlen:] 

        # Numbers consisting of more than five digits are excluded
        isvalid = True
        for num in digits.split(","):
            if len(num) > 5:
                isvalid = False
                break
        if not isvalid:
            continue
        
        x = [digit2idx[digit] for digit in digits]
        y = [digit2idx[digit] for digit in (digits[1:] + ",")]

        # Data Bucketing
        if len(x) <= 100:
            x += [0] * (100 - len(x)) # Zero postpadding
            y += [0] * (100 - len(y)) # Zero postpadding
            xs0.append(x); ys0.append(y)
        elif len(x) <= 200:
            x += [0] * (200 - len(x)) # Zero postpadding
            y += [0] * (200 - len(y)) # Zero postpadding
            xs1.append(x); ys1.append(y)
        elif len(x) <= 300:
            x += [0] * (300 - len(x)) # Zero postpadding
            y += [0] * (300 - len(y)) # Zero postpadding
            xs2.append(x); ys2.append(y)
        else:
            x += [0] * (400 - len(x)) # Zero postpadding
            y += [0] * (400 - len(y)) # Zero postpadding
            xs3.append(x); ys3.append(y)

    X = [np.array(xs0), np.array(xs1), np.array(xs2), np.array(xs3)]   
    Y = [np.array(ys0), np.array(ys1), np.array(ys2), np.array(ys3)]

    pickle.dump((X, Y), open('../data/train.pkl', 'wb'))
    
def load_train_data(num):
    X, Y = pickle.load(open('../data/train.pkl', 'rb'))
    return X[num], Y[num]

def create_val_data():
    digit2idx, idx2digit = load_vocab()

    lines = [line.split('"')[1] for line in open('../data/train.csv', 'r').read().splitlines()[1:]][-Hyperparams.vallen:]

    xs, ys = [], []
    for line in lines:  
        digits = line[:line.rfind(",")+1][-Hyperparams.ctxlen:]
        x = [digit2idx[digit] for digit in digits]
        x = [0] * (Hyperparams.ctxlen - len(x)) + x # Zero prepadding
        xs.append(x)
        
        ys.append(line[line.rfind(",")+1:]) # Ground truth

    X = np.array(xs)
                 
    pickle.dump((X, ys), open('../data/val.pkl', 'wb'))

def load_val_data():
    X, Y = pickle.load(open('../data/val.pkl', 'rb'))
    return X, Y
    
def create_test_data():
    digit2idx, idx2digit = load_vocab()

    ids = [line.split(',')[0] for line in open('../data/test.csv', 'r').read().splitlines()[1:]]
    lines = [line.split('"')[1] + "," for line in open('../data/test.csv', 'r').read().splitlines()[1:]]
    xs = []
    for line in lines:  
        x = [digit2idx[digit] for digit in line[-Hyperparams.ctxlen:]]
        x = [0] * (Hyperparams.ctxlen - len(x)) + x # Zero prepadding
        
        xs.append(x)
         
    X = np.array(xs, np.int32)
    
    pickle.dump((X, ids), open('../data/test.pkl', 'wb'))
 
def load_test_data():
    X, ids = pickle.load(open('../data/test.pkl', 'rb'))
    return X, ids

if __name__ == "__main__":
    create_train_data()
    create_val_data()
    create_test_data()
    print("Done")