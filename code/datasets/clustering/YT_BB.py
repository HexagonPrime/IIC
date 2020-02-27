import os
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data.dataset import Dataset 
from PIL import Image

class YT_BB(Dataset):
    """
    Arguments:
        root: CSV files path
        transform: desired transformation
        frame: which frame to take
        crop: whether crop by the bounding box
    """
    
    def __init__(self, root, transform, frame, crop):
	self.root = root
        self.csv_path = root + '/yt_bb.csv'
        self.transform = transform
        self.crop = crop

        tmp_df = pd.DataFrame.from_csv(self.csv_path, header=None, index_col=False)
        col_names = ['segment_id', 'class_id', 'path', 'timestamp', 'object_presence', 'xmin', 'xmax', 'ymin', 'ymax']
        tmp_df.columns = col_names
        
        # Get list of unique video segment files
        groups = tmp_df.groupby('segment_id')
        self.dataset = []
        included = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for name, group in groups:
            # Circular if this frame required exceeds frames the segment has.
            # this_frame = frame % len(group)
            # Choose the last frame if the required frame does not exist
            if this_frame > len(this_group)-1:
                this_frame = len(this_group)-1
            print 'Frame: ' + str(this_frame)
            this_row = group.iloc[[this_frame]]
            this_class = this_row['class_id'].iat[0]
            if included[this_class] < 1000:
                self.dataset.append(this_row)
                included[this_class] = included[this_class] + 1
        print 'Dataset size: ' + str(len(self.dataset))

    def __getitem__(self, index):
        this_row = self.dataset[index]

        # Get output image.
        img = Image.open(self.root + this_row['path'].iat[0])
	
        if self.crop:
	    width, height = img.size
# 	    print width
# 	    print height
	    left = int(width * this_row['xmin'].iat[0])
	    top = int(height * this_row['ymin'].iat[0])
	    right = int(width * this_row['xmax'].iat[0])
	    bottom = int(height * this_row['ymax'].iat[0])
# 	    print left
# 	    print top
# 	    print right
# 	    print bottom
	    img = img.crop((left, top, right, bottom))
	    #new_width, new_height = img.size
	    img = transforms.Resize([32,32])(img)
# 	    print new_width
# 	    print new_height
# 	    img.show()
	
	#img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = this_row['class_id'].iat[0]
        return img, label

    def __len__(self):
        return len(self.dataset)
