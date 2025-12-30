import pandas as pd

features = pd.read_csv("data/elliptic_txs_features.csv")
classes = pd.read_csv("data/elliptic_txs_classes.csv")
edges = pd.read_csv("data/elliptic_txs_edgelist.csv")

print("FEATURES SHAPE:", features.shape)
print("CLASSES SHAPE:", classes.shape)
print("EDGES SHAPE:", edges.shape)
