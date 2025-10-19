import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import os

#%% Reading files
folderPath = "images"
files = os.listdir(folderPath)
imageList = sorted([img for img in files if img.endswith('.png')])
print(imageList)

#%% Reading images
for i in range(len(imageList)):
    I = np.array(Image.open(os.path.join(folderPath,imageList[i])))
    plt.figure()
    plt.imshow(I)
    plt.show()

#%% Simple average
I_avg = np.zeros(I.shape)
scale = 0

for i in range(len(imageList)):
    I_avg += (I.mean()/255.0)*(I/255.0)
    scale += (I.mean()/255.0)

I_avg /= scale.sum()

plt.figure()
plt.imshow(I_avg)
plt.show()