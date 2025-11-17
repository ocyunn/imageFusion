import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import PIL.Image as Image

#%% Reading files

folderPath = "images"
files = os.listdir(folderPath)
imageList = sorted([img for img in files if img.endswith('.png')])
exposureTime = 1/np.array([0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
B = np.log(exposureTime)

#%%

def gsolve(Z, B, l, w):
    (dim0, dim1) = Z.shape
    X = np.zeros((dim0*dim1+n-1,n+dim0))
    y = np.zeros((dim0*dim1+n-1, 1))
    k = 0

    for i in range(dim0):
        for j in range(dim1):
            wij = w[Z[i,j]]
            X[k, Z[i,j]] = wij
            X[k, n+i] = -wij
            y[k] = wij*B[j]
            k += 1

    for i in range(n-2):
        X[k+i,i] = l*w[i]
        X[k+i,i+1] = -2*l*w[i]
        X[k+i,i+2] = l*w[i]

    X[-1,int((zmin+zmax)/2)] = 1

    # Solve the system
    W, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    g = W[:n].flatten()
    logE = W[n:].flatten()

    return g, logE

def irradiance(Z, B, g, w):
    logIrradianceMap = np.zeros(Z.shape[0])
    for i in range(Z.shape[0]):
            z = Z[i, :].astype(int)
            wj = w[z]
            logIrradianceMap[i] = np.sum(wj * (g[z] - B)) / (wj.sum())

    return logIrradianceMap

#%% Settings
# Max and min intensity
zmin = 0
zmax = 255

# Bit depth
n = 256

# Smoothness parameter
lb = 100

# Number of brackets
N = len(exposureTime)

# Number of sampled points for fitting
npoints = 100

#%% Creating placeholders
Z = np.zeros((npoints, N), dtype = int)
I = np.array(Image.open(os.path.join(folderPath,imageList[0])))[:,:,0]
(dim0, dim1) = I.shape
Istack = np.zeros((dim0*dim1, N))
logIrradianceMap = np.zeros((dim0, dim1, 3))

#%% Sampling randomly
np.random.seed(21)
sample_idx = np.random.randint(low = 0, high = I.shape[0]*I.shape[1], size = (npoints,))

#%% Creating weight function
w = np.arange(0, 256)
w = (w - zmin) * (w <= (zmax + zmin) / 2) + (zmax - w) * (w > (zmax + zmin) / 2)

#%% Running the algorithm on each channel

for channel in range(3):
    for i in range(N):
        I = np.array(Image.open(os.path.join(folderPath,imageList[i])))[:,:,channel].flatten()
        Z[:,i] = I[sample_idx]
        Istack[:,i] = I.copy()
    #% Exposure fusion
    g, logE = gsolve(Z,B,lb,w)
    logIrradianceMap[:, :, channel] = (irradiance(Istack, B, g, w)).reshape(dim0, dim1)

    plt.figure(); plt.plot(g, 'o'); plt.show()

#%% Plotting (log) irradiance maps
channelTitle = ["Red", "Green", "Blue"]
fig = plt.figure(figsize = (6,3), dpi = 300)

for i in range(3):
    ax = fig.add_subplot(1,3,i+1)
    cm = ax.imshow(logIrradianceMap[:, :, i], cmap="RdBu_r", vmin = -6, vmax = 6)
    ax.axis("off")
    ax.set_title(channelTitle[i])

plt.tight_layout()
plt.show()

# fig.savefig("figures/logIrradianceMaps.png")

#%% Plotting RGB log irradiance map
gamma = 1/2.2
irradianceMap = np.exp(logIrradianceMap)**gamma
irradianceMap /= (irradianceMap+1)
irradianceMap *= 255
irradianceMap = np.uint8(irradianceMap)

fig = plt.figure(figsize = (3,3), dpi = 300)
ax = fig.add_subplot(111)
ax.imshow(irradianceMap)
ax.axis("off")
plt.tight_layout()
plt.show()

hdrImage = Image.fromarray(irradianceMap)
hdrImage.save("figures/hdrImage.jpg")