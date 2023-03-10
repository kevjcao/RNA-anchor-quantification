#!/usr/bin/env python3

# Install cellpose if needed
# import pip
# pip.main(['install', 'cellpose'])

import matplotlib.pyplot as plt
from cellpose import models, io, plot, utils
import tkinter as tk
from tkinter import filedialog
import numpy as np
import re
import os
from pathlib import Path

# locate directory for batch processing
root = tk.Tk()
root.withdraw()
root.directory = filedialog.askdirectory()

stack_dir: Path = Path(root.directory)
for file in stack_dir.glob('*.tif'):
    image_path = root.directory + '/' + file.name
    # create folder to hold cellpose outputs
    cellpose_output = root.directory + '/cellpose_output'
    if not os.path.exists(cellpose_output):
        os.makedirs(cellpose_output)

    # read tif into cellpose; channels=[0, 0] for grayscale image
    model = models.Cellpose(gpu=False, model_type='cyto')
    img = io.imread(image_path)
    masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0, 0],
                                             flow_threshold=0.4, do_3D=False)
    # save cellpose segmentation as _seg.npy
    io.masks_flows_to_seg(img, masks, flows, diams, image_path)

    # save segmentation as .txt for imageJ; outlines also used for quantification
    base = os.path.splitext(image_path)[0]
    outlines = utils.outlines_list(masks)
    io.outlines_to_text(base, outlines)

    # plot segmentation, mask, and pose outputs
    fig1 = plt.figure(figsize=(12, 5))
    plot.show_segmentation(fig1, img, masks, flows[0], channels=[0, 0])
    plt.tight_layout()
    seg_file = file.name.replace(".tif", ".png")
    plt.savefig(root.directory + '/cellpose_output/' + seg_file)
    plt.show()

    # obtain ROIs for each segmented cell; plot red overlay on original image
    np_seg = re.sub('.tif$', '_seg.npy', image_path)
    dat = np.load(np_seg, allow_pickle=True).item()
    outlines = utils.outlines_list(dat['masks'])

    fig2 = plt.figure(figsize=(5, 5))
    plt.imshow(dat['img'])
    for o in outlines:
        plt.plot(o[:, 0], o[:, 1], linewidth=0.5, color='r')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # measure mvg of ROIs
    rois_n = np.zeros(len(outlines))
    for i in range(1, len(outlines) + 1):
        f_mask = np.isin(masks, i)
        f_slice = f_mask * img
        f_mean = f_slice[f_slice != 0].mean()
        rois_n[i - 1] = f_mean
    roi_mvg = file.name.replace(".tif", ".csv")
    np.savetxt((root.directory + '/cellpose_output/' + roi_mvg), rois_n, delimiter=",")





