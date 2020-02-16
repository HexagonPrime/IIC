import os
import torch
import pandas as pd
from torch.utils.data.dataset import Dataset 

class YT_BB(Dataset):
    """
    Arguments:
        root: CSV files path
        transform: desired transformation
        frame: which frame to take
        crop: whether crop by the bounding box
    """
    
    def __init__(self, root, transform, frame, crop):
        self.csv_path = root + '/yt_bb.csv'
        self.transform = transform
        self.frame = frame
        self.crop = crop

        tmp_df = pd.DataFrame.from_csv(csv_path, header=None, index_col=False)
        col_names = ['segment_id', 'class_id', 'path', 'timestamp', 'object_presence', 'xmin', 'xmax', 'ymin', 'ymax']
        tmp_df.columns = col_names
        
        # Get list of unique video segment files
        self.vids = tmp_df['segment_id'].unique()
        self.dataset = tmp_df.groupby('segment_id')

    def __getitem__(self, index):
        this_segment_id = self.vids[index]
        this_group = self.dataset.get_group(this_segment_id)

        # Select the specific frame
        this_frame = self.frame

        # Choose the last frame if the required frame does not exist
        if this_frame > len(this_group)-1:
            this_frame = len(this_group)-1
        
        this_row = this_group.iloc[[this_frame]]

        # Get output image
        img = Image.open(this_row['path'])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = this_row['class_id']
        return img, label

    def __len__(self):
        return len(self.vids)