from matplotlib import pyplot as plt
import numpy as np

G_losses = np.loadtxt('./checkpoints/MNIST/G_losses.txt')
D_losses = np.loadtxt('./checkpoints/MNIST/D_losses.txt')

plt.style.use('ggplot')
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.grid(False)
plt.legend()
plt.savefig('./checkpoints/MNIST/G&D_losses.png')
plt.show()

G_losses = np.loadtxt('./checkpoints/CIFAR10/G_losses.txt')
D_losses = np.loadtxt('./checkpoints/CIFAR10/D_losses.txt')

plt.style.use('ggplot')
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.grid(False)
plt.legend()
plt.savefig('./checkpoints/CIFAR10/G&D_losses.png')
plt.show()
