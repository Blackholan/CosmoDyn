import pickle

with open(
    "PotsGSat462710N1.pkl",
    "rb",
) as f:
    pots = pickle.load(f)

print(type(pots))
print(pots.keys())