import matplotlib.pyplot as plt
import random
from torchvision import transforms

#plotting rondom images from dataset
def class_plot(data):
    inv_normalize = transforms.Normalize(
        mean=[-0.4302 / 0.2361, -0.4575 / 0.2347, -0.4539 / 0.2432],
        std=[1 / 0.2361, 1 / 0.2347, 1 / 0.2432]
    )
    classes = data.classes
    encoder = {}
    for i in range(len(classes)):
        encoder[i] = classes[i]

    fig,axes = plt.subplots(figsize=(14, 10), nrows = 6, ncols=3)
    for ax in axes.flatten():
        a = random.randint(0,len(data))
        (image,label) = data[a]
        label = int(label)
        l = encoder[label]
        image = inv_normalize(image)
        image = image.numpy().transpose(1,2,0)
        im = ax.imshow(image)
        ax.set_title(l)
        ax.axis('off')
    plt.show()
