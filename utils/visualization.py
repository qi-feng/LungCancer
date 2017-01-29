import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#import SimpleITK as sitk

import seaborn as sns
try:
    sns.set_style("ticks")
    # sns.set_style({"axes.axisbelow": False})
    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
except:
    print("sns problem")

try:
    from mayavi import mlab
except:
    print("Can't import mayavi")


def plot_projections(slices1):
    projectionZ = np.sum(slices1, axis=0)
    projectionX = np.sum(slices1, axis=1)
    projectionY = np.sum(slices1, axis=2)

    plt.imshow(projectionZ, cmap=plt.cm.bone)
    plt.colorbar()
    plt.show()

    plt.imshow(projectionX, cmap=plt.cm.bone)
    plt.show()

    plt.imshow(projectionY, cmap=plt.cm.bone)
    plt.show()

def plot_all_slices(slices1, ncols = 8):
    num_scans1 = slices1.shape[0]
    fig, axes = plt.subplots(num_scans1 // ncols, ncols, sharex='all', sharey='all',
                             figsize=(ncols * 2, num_scans1 // ncols * 2))
    for i in range(num_scans1 // ncols * ncols):
        axes[i // ncols, i % ncols].axis('off')
        axes[i // ncols, i % ncols].imshow(slices1[i], cmap=plt.cm.bone)

    # plt.tight_layout()
    plt.show()


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

