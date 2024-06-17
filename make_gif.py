import imageio
from glob import glob

path1_list = sorted(glob('./imgs/'+"cf*.png"))
duration = 2
loop = 0

pics_list = []
for image_name in path1_list:
    im = imageio.v3.imread(image_name)
    pics_list.append(im)
imageio.mimsave('./checkpoints/CIFAR10/cf.gif', pics_list, 'GIF', fps=0.8, loop=loop)

path2_list = sorted(glob('./imgs/'+"mn*.png"))
pics_list = []
for image_name in path2_list:
    im = imageio.v3.imread(image_name)
    pics_list.append(im)
imageio.mimsave('./checkpoints/MNIST/mn.gif', pics_list, 'GIF', fps=0.8, loop=loop)
