#!/usr/bin/env python3

# Install cellpose if needed
# import pip
# pip.main(['install', 'cellpose'])

import matplotlib.pyplot as plt
import cellpose
from cellpose import models, io, plot, utils
import tkinter as tk
from tkinter import filedialog
import numpy as np
import re
import os
from pathlib import Path
from glob import glob

# locate directory for batch processing
root = tk.Tk()
root.withdraw()
root.directory = filedialog.askdirectory()

## choose model; uncomment for standard models
model_path = filedialog.askopenfilename()
root_model = os.path.split(model_path)
cellpose.io.add_model(model_path)

stack_dir: Path = Path(root.directory)

for file in stack_dir.glob('*.tif'):
    image_path = root.directory + '/' + file.name
    # create folder to hold cellpose outputs
    cellpose_output = root.directory + '/cellpose_output'
    if not os.path.exists(cellpose_output):
        os.makedirs(cellpose_output)

    # read tif into cellpose; channels=[0, 0] for grayscale image
    model = models.CellposeModel(gpu=False, model_type=root_model[1])
    # model = models.Cellpose(gpu=False, model_type='cyto2')
    img = io.imread(image_path)

    ## for custom models omit diam; use following line:
    masks, flows, styles = model.eval(img, diameter=None, channels=[0, 0], flow_threshold=None, do_3D=False)

    ## for standard models, use following lines:
    # masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0, 0], flow_threshold=None, do_3D=False)
    # save cellpose segmentation as _seg.npy
    # io.masks_flows_to_seg(img, masks, flows, diams, image_path)

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

    ## may not be necessary
    # obtain ROIs for each segmented cell; plot red overlay on original image
    # np_seg = re.sub('.tif$', '_seg.npy', image_path)
    # dat = np.load(np_seg, allow_pickle=True).item()
    # outlines = utils.outlines_list(dat['masks'])

    fig2 = plt.figure(figsize=(5, 5))
    plt.imshow(img)
    # plt.imshow(dat['img'])
    for o in outlines:
        plt.plot(o[:, 0], o[:, 1], linewidth=0.5, color='r')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # measure mvg of ROIs
    rois_n = np.zeros(len(outlines))
    for i in range(1, len(outlines) + 1):
        f_mask = np.isin(masks, i)
        # create a new image where mask is applied to original image
        f_slice = f_mask * img
        f_mean = f_slice[f_slice != 0].mean()
        rois_n[i - 1] = f_mean
    roi_mvg = file.name.replace(".tif", ".csv")
    np.savetxt((root.directory + '/cellpose_output/' + roi_mvg), rois_n, delimiter=",")

# for file in os.listdir(cellpose_output):
#
# cellpose_output_dir: Path = Path(cellpose_output)
# with open("concat_test_1.csv", "w") as outfile:
#     for count, file in enumerate(cellpose_output_dir.glob('*.csv')):
#         with open(file) as in_file:
#             header = next(in_file)
#             if count == 0:
#                 outfile.write(header)
#                 line = next(in_file)
#             if not line.startswith("\n"):
#                 line = line + "\n"
#             outfile.write(line)
