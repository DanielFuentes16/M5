import matplotlib.pyplot as plt

with open('loss.txt') as f:
    lines = f.readlines()
if len(lines) is 0:
    print('No lines in file')
    exit()

losses = []
iters = []

for line in lines:
    splitStr = line.split()
    iters.append(int(splitStr[0]))
    losses.append(float(splitStr[1]))

plt.plot(iters, losses)
plt.title("Model loss")
plt.xlabel("Iteration")
plt.ylabel("Total loss")
plt.savefig("loss.png")
