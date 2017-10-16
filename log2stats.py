import sys
import numpy as np

test, advers = [], []  
t_count, a_count = 0, 0

for line in sys.stdin:

    if not line.startswith("Test accuracy"):
        continue

    acc = float(line.split(":")[1])

    if line.startswith("Test accuracy on adversarial"):
        advers.append(100*acc)
        a_count += 1
        continue

    if line.startswith("Test accuracy on legitimate"):
        test.append(100*acc)
        t_count += 1
        continue


assert t_count == a_count

print(np.mean(test), np.std(test), np.min(test), np.max(test))
print(np.mean(advers), np.std(advers), np.min(advers), np.max(advers))

