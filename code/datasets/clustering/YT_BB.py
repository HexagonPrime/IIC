import os
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data.dataset import Dataset 
from PIL import Image

"""
CLAIM: The code from this file was written by Cai Shengqu based on
'https://github.com/utkuozbulak/pytorch-custom-dataset-examples'.
"""

class YT_BB(Dataset):
    """
    Arguments:
        root: CSV files path.
        transform: desired transformation.
        frame: which frame to take.
        crop: whether crop by the bounding box.
        partition: one of 'train', 'test' and 'train+test', refers to which partition to use.
    """
    
    def __init__(self, root, transform, frame, crop, partition):
	self.root = root
        # Avoid exceeding 10 frames.
        frame = frame % 19
        if frame > 9:
            frame = 18 - frame
        csv_path_train = root + '/frame' + str(frame) + '_train' + '.csv'
        csv_path_test = root + '/frame' + str(frame) + '_test' + '.csv'
        self.transform = transform
        self.crop = crop
        print 'Crop: ' + str(self.crop)
        print 'Frame: ' + str(frame)

        if partition == 'train':
            tmp_df = pd.DataFrame.from_csv(csv_path_train, header=None, index_col=False)
        elif partition == 'test':
            tmp_df = pd.DataFrame.from_csv(csv_path_test, header=None, index_col=False)
        elif partition == 'train+test':
            tmp_df_train = pd.DataFrame.from_csv(csv_path_train, header=None, index_col=False)
            tmp_df_test = pd.DataFrame.from_csv(csv_path_test, header=None, index_col=False)
            tmp_df = pd.concat([tmp_df_train, tmp_df_test])
        else:
            assert(False)

        col_names = ['segment_id', 'class_id', 'path', 'timestamp',\
                     'object_presence', 'xmin', 'xmax', 'ymin', 'ymax']
        tmp_df.columns = col_names

        self.image_arr = np.asarray(tmp_df.iloc[:, 2])
        self.label_arr = np.asarray(tmp_df.iloc[:, 1])
        self.xmin_arr = np.asarray(tmp_df.iloc[:, 5])
        self.xmax_arr = np.asarray(tmp_df.iloc[:, 6])
        self.ymin_arr = np.asarray(tmp_df.iloc[:, 7])
        self.ymax_arr = np.asarray(tmp_df.iloc[:, 8])
        print len(self.image_arr)

    def __getitem__(self, index):
        # Get output image.
        img = Image.open(self.root + self.image_arr[index])
        label = self.label_arr[index]
	
        # Crop frames by bounding boxes. It is kept for the use of original YT_BB,
        # redundant for cropped yt_bb_small.
        if self.crop:
	    width, height = img.size
	    left = int(width * self.xmin_arr[index])
	    top = int(height * self.ymin_arr[index])
	    right = int(width * self.xmax_arr[index])
	    bottom = int(height * self.ymax_arr[index])
	    img = img.crop((left, top, right, bottom))
	    img = transforms.Resize([32,32])(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_arr)
