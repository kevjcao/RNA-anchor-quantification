#!/usr/bin/env python3

## Install cellpose if needed
# import pip
# pip.main(['install', 'cellpose'])
# import cellpose

import matplotlib.pyplot as plt
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

    # for custom models use following line:
    # masks, flows, styles = model.eval(img, diameter=None, channels=[0, 0], flow_threshold=None, do_3D=False)

    # for standard models use following lines; save cellpose segmentation as _seg.npy:
    masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0, 0], flow_threshold=None, do_3D=False)
    io.masks_flows_to_seg(img, masks, flows, diams, image_path)

    # save segmentation as .txt for imageJ; outlines also used for quantification
    base = os.path.splitext(image_path)[0]
    outlines = utils.outlines_list(masks)
    io.outlines_to_text(base, outlines)

    # plot segmentation, mask, and pose outputs
    fig1 = plt.figure(figsize=(12, 5), dpi=300)
    plot.show_segmentation(fig1, img, masks, flows[0], channels=[0, 0])
    plt.tight_layout()
    seg_file = file.name.replace(".tif", ".png")
    plt.savefig(root.directory + '/cellpose_output/' + seg_file)
    plt.show()

    fig2 = plt.figure(figsize=(5, 5), dpi=300)
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

# create a new file that places all the csv files into one
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

# save the compiled csv file
compilation_df = pd.concat(compilation_df, axis=1)
compilation_df.columns = filenames
compilation_df.to_excel(os.path.join(cellpose_output, 'compiled_cellpose_outputs.xlsx'), index=False)
df_stats = compilation_df.describe()

# plot the compiled csv files w/ mean & std
fig3, ax = plt.subplots(figsize=(10, 10), dpi=300)
sns.violinplot(data=compilation_df, inner='quartile', color='xkcd:pale green')
sns.swarmplot(data=compilation_df, size=2, palette=['grey'])
for i, column in enumerate(compilation_df.columns):
    col_mean = compilation_df[column].mean()
    col_std = compilation_df[column].std()
    ax.scatter(i, col_mean, s=10, color='red', label='Mean' if i == 0 else '',  zorder=3)
    if col_mean > 0:
        ax.text(i + 0.15, col_mean, r"$\mu = $" + str(round(col_mean, 2)), fontsize=10, va="center",
                bbox=dict(facecolor="white", edgecolor="black", pad=2), zorder=3)
        ax.errorbar(i, col_mean, yerr=col_std, color='black', fmt=' ', capsize=2, label='Std Dev' if i == 0 else '',
                    zorder=3)
    else:
        ax.errorbar(i, col_mean, yerr=0, color='black', fmt=' ', capsize=2, label='Std Dev' if i == 0 else '',
                    zorder=3)
    unique_ints = compilation_df[column].count()
    if unique_ints > 1:
        ax.text(i, ax.get_ylim()[0], f'n={unique_ints}', fontsize=10, ha='center', va='bottom', color='black')
    else:
        ax.text(i, ax.get_ylim()[0], f'no cells', fontsize=10, ha='center', va='bottom', color='black')
ax.set_title("Cellpose outputs, raw data")
ax.set_ylabel("mean grey value (au)")
ax.set_xlabel("file name")
if len(compilation_df.columns) > 5:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.show()
