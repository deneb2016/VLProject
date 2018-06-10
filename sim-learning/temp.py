import pickle
import numpy as np

bbb = pickle.load(open("labels.pickle", "rb"))

label_size = 80
same_label_set = []
sample_label = np.zeros(label_size)
print(sample_label)
sample_label[3] = 1

eq = np.equal(sample_label, sample_label)
print(eq.astype(int))
print(np.prod(eq.astype(int)))

for idx, label in bbb.items():
    eq = np.equal(label, sample_label)
    indicator = np.prod(eq.astype(int))
    if (indicator == 1):
        same_label_set.append(idx)

print(len(same_label_set))
sel = np.random.choice(same_label_set, 1, replace=False)
print(sel)
print(sel[0])
