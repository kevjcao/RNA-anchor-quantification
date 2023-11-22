#!/usr/bin/env python3

## Install cellpose if needed
# import pip
# pip.main(['install', 'cellpose'])

import matplotlib.pyplot as plt
import cellpose
from cellpose import models, io, plot, utils
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import seaborn as sns
import os
from pathlib import Path

# locate directory for batch processing
root = tk.Tk()
root.withdraw()
root.directory = filedialog.askdirectory()

# create folder to hold cellpose outputs
cellpose_output = root.directory + '/cellpose_output'
if not os.path.exists(cellpose_output):
    os.makedirs(cellpose_output)

# choose model; comment out if using standard models
# model_path = filedialog.askopenfilename()
# root_model = os.path.split(model_path)
# cellpose.io.add_model(model_path)

stack_dir: Path = Path(root.directory)
for file in stack_dir.glob('*.tif'):
    image_path = root.directory + '/' + file.name
    # read tif into cellpose; channels=[0, 0] for grayscale image; use models.Cellpose for standard models
    # model = models.CellposeModel(gpu=False, model_type=root_model[1])
    model = models.Cellpose(gpu=False, model_type='cyto2')
    img = io.imread(image_path)

    ## for custom models use following line:
    # masks, flows, styles = model.eval(img, diameter=None, channels=[0, 0], flow_threshold=None, do_3D=False)

    ## for standard models use following lines; save cellpose segmentation as _seg.npy:
    masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0, 0], flow_threshold=None, do_3D=False)
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

    fig2 = plt.figure(figsize=(5, 5))
    plt.imshow(img)
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

#create a new file that places all the csv files into one
compilation_df = []
filenames = []
for cp_output in os.listdir(cellpose_output):
    if cp_output.endswith(".csv"):
        try:
            df = pd.read_csv(os.path.join(cellpose_output, cp_output), header=None)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame([0])
            df_column = df.iloc[:, 0:1]
            compilation_df.append(df)
            filenames.append(os.path.splitext(cp_output)[0])
        else:
            df_column = df.iloc[:, 0:1]
            compilation_df.append(df_column)
            filenames.append(os.path.splitext(cp_output)[0])

#save the compiled csv file
compilation_df = pd.concat(compilation_df, axis=1)
compilation_df.columns = filenames
compilation_df.to_csv(os.path.join(cellpose_output, 'compiled_cellpose_outputs.csv'), index=False)

plt.figure(figsize=(10, 6))
sns.violinplot(data=compilation_df, inner='quartile', color='skyblue')
sns.swarmplot(data=compilation_df, size=4, palette='dark:black')
plt.title("Cellpose outputs, raw data")
plt.xlabel("file name")
if len(compilation_df.columns) > 5:
    plt.xticks(rotation=90)
plt.ylabel("mean grey value (au)")
plt.show()