import argparse
import numpy as np
import os
import torch
from scipy.io import loadmat
import pickle

# -------------------- Argumentos --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--P', type=int, default=10, help='history steps')
parser.add_argument('--Q', type=int, default=1, help='prediction steps')
parser.add_argument('--save_path', type=str, default='./CUSTOM_DATA/', help='path to save processed files')
parser.add_argument('--se_file', type=str, default='./CUSTOM_DATA/SE(CUSTOM).txt', help='Node2Vec spatial embedding file')
args = parser.parse_args()

P = args.P
Q = args.Q
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

# -------------------- Cargar datos --------------------
data = loadmat('traffic_dataset.mat')
Y_train_raw = data['tra_Y_tr']  # [36, 1261]
Y_test_raw = data['tra_Y_te']  # [36, 840]
adj_mat = data['tra_adj_mat']  # [36, 36]

# -------------------- Construir dataset X e Y (basado solo en flujo de tráfico) --------------------
def build_dataset_from_volume(Y_raw, P, Q):
    total = Y_raw.shape[1]
    samples = total - P - Q + 1
    X = np.array([Y_raw[:, i:i+P].T for i in range(samples)])       # [samples, P, N]
    Y = np.array([Y_raw[:, i+P:i+P+Q].T for i in range(samples)])   # [samples, Q, N]
    return X, Y

X_train, Y_train = build_dataset_from_volume(Y_train_raw, P, Q)
X_test, Y_test = build_dataset_from_volume(Y_test_raw, P, Q)

# -------------------- Normalización --------------------
mean = X_train.mean()
std = X_train.std()
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# -------------------- Temporal Embedding --------------------
def build_TE(num_samples, P, Q):
    TE = []
    for i in range(num_samples):
        dow = i % 7
        tod = i % 96
        step = [[dow, tod] for _ in range(P + Q)]
        TE.append(step)
    return np.array(TE, dtype=np.int32)

TE_train = build_TE(X_train.shape[0], P, Q)
TE_test = build_TE(X_test.shape[0], P, Q)

# -------------------- Cargar SE --------------------
def load_SE_from_txt(path, num_nodes, dims):
    SE = np.zeros((num_nodes, dims))
    with open(path, 'r') as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            idx = int(parts[0])
            vec = np.array([float(x) for x in parts[1:]])
            SE[idx] = vec
    return torch.FloatTensor(SE)

SE = load_SE_from_txt(args.se_file, num_nodes=36, dims=64)

# -------------------- Convertir a tensores --------------------
trainX = torch.FloatTensor(X_train)       # [samples, P, N]
trainY = torch.FloatTensor(Y_train)       # [samples, Q, N]
trainTE = torch.LongTensor(TE_train)      # [samples, P+Q, 2]
testX  = torch.FloatTensor(X_test)
testY  = torch.FloatTensor(Y_test)
testTE = torch.LongTensor(TE_test)

# -------------------- Dividir test para validación --------------------
val_split = int(0.2 * testX.shape[0])
valX, valTE, valY = testX[:val_split], testTE[:val_split], testY[:val_split]
testX, testTE, testY = testX[val_split:], testTE[val_split:], testY[val_split:]

# -------------------- Guardar --------------------
torch.save((trainX, trainTE, trainY), os.path.join(save_path, 'train.pt'))
torch.save((valX, valTE, valY), os.path.join(save_path, 'val.pt'))
torch.save((testX, testTE, testY), os.path.join(save_path, 'test.pt'))
torch.save(SE, os.path.join(save_path, 'SE.pt'))
torch.save({'mean': mean, 'std': std}, os.path.join(save_path, 'norm.pt'))

# Guardar adj_mx con dummy ids
adj_np = adj_mat.astype(np.float32)
dummy_ids = list(range(adj_np.shape[0]))
dummy_map = {i: i for i in dummy_ids}
with open(os.path.join(save_path, 'adj_mx.pkl'), 'wb') as f:
    pickle.dump((dummy_ids, dummy_map, adj_np), f)

# -------------------- Guardar edgelist (formato: i j weight) --------------------
edge_list_file = os.path.join(save_path, 'Adj(CUSTOM).txt')
with open(edge_list_file, 'w') as f:
    N = adj_mat.shape[0]
    for i in range(N):
        for j in range(N):
            if adj_mat[i, j] > 0:
                f.write(f"{i} {j} {adj_mat[i, j]}\n")
print(f"Adjacency list guardada en: {edge_list_file}")

print(f"Datos guardados en {save_path} con P={P}, Q={Q} (usando solo flujo de tráfico)")