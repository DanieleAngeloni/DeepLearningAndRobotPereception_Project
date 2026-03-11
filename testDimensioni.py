import torch
from CHALLENGE.dataset import DepthDataset

DATA_DIR = "C:\\Users\\angel\\PycharmProjects\\pythonProjectDeep\\ProgettoDeepLearning\\CHALLENGE\\DepthEstimationUnreal"

# Inizializza il dataset (puoi cambiare TRAIN in VAL o TEST)
dataset = DepthDataset(data_dir=DATA_DIR, train=DepthDataset.TRAIN)

# Carica un sample
rgb, depth = dataset[0]

# Stampa le dimensioni
print("RGB shape:", rgb.shape)
print("Depth shape:", depth.shape)


import numpy as np

file_path = r"C:\Users\angel\PycharmProjects\pythonProjectDeep\ProgettoDeepLearning\CHALLENGE\DepthEstimationUnreal\depth\train\0.npy"

# Carica il file .npy
dati = np.load(file_path)

# Mostra il contenuto (o una parte, se è grande)
print(dati)

# Se vuoi mostrare solo un ritaglio centrale per leggere meglio
print("Valori centrali:")
print(dati[60:85, 100:130])