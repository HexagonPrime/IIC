import sys
from datetime import datetime

import torch
import torchvision
from torch.utils.data import ConcatDataset

from code.datasets.clustering.truncated_dataset import TruncatedDataset
from code.utils.cluster.transforms import sobel_make_transforms, \
  greyscale_make_transforms
from code.utils.semisup.dataset import TenCropAndFinish
from .general import reorder_train_deterministic

from code.datasets.clustering.YT_BB import YT_BB
from code.datasets.clustering.YT_BB_5 import YT_BB_5


# Used by sobel and greyscale clustering twohead scripts -----------------------

def cluster_twohead_create_YT_BB_dataloaders(config):
  assert (config.mode == "IID")
  assert (config.twohead)

  if config.dataset == "YT_BB":
    config.train_partitions_head_A = config.train_partition
    config.train_partitions_head_B = config.train_partitions_head_A

    config.mapping_assignment_partitions = config.assignment_partition
    config.mapping_test_partitions = config.test_partition

    dataset_class = YT_BB  #TODO YT_BB custom class

    # datasets produce either 2 or 5 channel images based on config.include_rgb
    tf1, tf2, tf3 = sobel_make_transforms(config)
  else:
    assert (False)

  print("Making datasets with YT_BB")
  sys.stdout.flush()

  dataloaders_head_A = \
    _create_dataloaders(config, dataset_class, tf1, tf2,
                        partition=config.train_partitions_head_A,
                       )

  dataloaders_head_B = \
    _create_dataloaders(config, dataset_class, tf1, tf2,
                        partition=config.train_partitions_head_B,
                       )

  mapping_assignment_dataloader = \
    _create_mapping_loader(config, dataset_class, tf3,
                           partition=config.mapping_assignment_partitions
                           )

  mapping_test_dataloader = \
    _create_mapping_loader(config, dataset_class, tf3,
                           partition=config.mapping_test_partitions
                          )

  return dataloaders_head_A, dataloaders_head_B, \
         mapping_assignment_dataloader, mapping_test_dataloader


# Data creation helpers --------------------------------------------------------

def _create_dataloaders(config, dataset_class, tf1, tf2,
                        partition, shuffle=False):
  curr_frame = int(config.base_frame)
  train_imgs_list = []
  for i in xrange(config.base_num):
    train_imgs_curr = dataset_class(root=config.dataset_root,
                                    transform=tf1,
                                    frame=curr_frame + config.base_interval * i,
                                    crop=config.crop_by_bb,
                                    partition=partition)
    train_imgs_list.append(train_imgs_curr)

  train_imgs = ConcatDataset(train_imgs_list)
  train_dataloader = torch.utils.data.DataLoader(train_imgs,
                                                 batch_size=config.dataloader_batch_sz,
                                                 shuffle=shuffle,
                                                 num_workers=0,
                                                 drop_last=False)

  if not shuffle:
    assert (isinstance(train_dataloader.sampler,
                       torch.utils.data.sampler.SequentialSampler))
  dataloaders = [train_dataloader]

  for d_i in xrange(config.num_dataloaders):
    print 'Include increment: ' + str(config.frame_increment)
    print 'd_i: ' + str(d_i)
    if config.frame_increment:
      curr_frame = curr_frame + config.interval
    else:
      if d_i == 0:
        curr_frame = curr_frame + config.interval
    print("Creating auxiliary dataloader ind %d out of %d time %s" %
          (d_i, config.num_dataloaders, datetime.now()))
    sys.stdout.flush()

    train_tf_imgs_list = []
    # for each base train dataset, create corresponding transformed dataset,
    # then concat together.
    for i in xrange(config.base_num):
      this_base_frame = config.base_frame + config.base_interval * i
      this_tf_frame = (curr_frame + config.base_interval * i) % 19
      if this_tf_frame > 9:
        this_tf_frame = this_base_frame - (this_tf_frame - 9)
      train_imgs_tf_curr = dataset_class(root=config.dataset_root,
                                         transform=tf2,
	                                 frame=this_tf_frame,
                                         crop=config.crop_by_bb,
                                         partition=partition)

      train_tf_imgs_list.append(train_imgs_tf_curr)

    train_imgs_tf = ConcatDataset(train_tf_imgs_list)
    train_tf_dataloader = \
      torch.utils.data.DataLoader(train_imgs_tf,
                                  batch_size=config.dataloader_batch_sz,
                                  shuffle=shuffle,
                                  num_workers=0,
                                  drop_last=False)

    if not shuffle:
      assert (isinstance(train_tf_dataloader.sampler,
                         torch.utils.data.sampler.SequentialSampler))
    # Verify the length for each dataloader is the same.
    assert (len(train_dataloader) == len(train_tf_dataloader))
    dataloaders.append(train_tf_dataloader)

  num_train_batches = len(dataloaders[0])
  print("Length of datasets vector %d" % len(dataloaders))
  print("Number of batches per epoch: %d" % num_train_batches)
  sys.stdout.flush()

  return dataloaders


def _create_mapping_loader(config, dataset_class, tf3, partition, 
                           truncate=False, truncate_pc=None,
                           tencrop=False,
                           shuffle=False):
  if truncate:
    print("Note: creating mapping loader with truncate == True")

  if tencrop:
    assert (tf3 is None)

  imgs_list = []
  if config.test_on_all_frame:
    for i in xrange(10):
      imgs_curr = dataset_class(root=config.dataset_root,
                                transform=tf3,
                                frame=i,
                                crop=config.crop_by_bb,
                                partition=partition)
  
      if truncate:
        print("shrinking dataset from %d" % len(imgs_curr))
        imgs_curr = TruncatedDataset(imgs_curr, pc=truncate_pc)
        print("... to %d" % len(imgs_curr))

      if tencrop:
        imgs_curr = TenCropAndFinish(imgs_curr, input_sz=config.input_sz,
                                 include_rgb=config.include_rgb)

      imgs_list.append(imgs_curr)
  else:
    for i in xrange(config.base_num):
      imgs_curr = dataset_class(root=config.dataset_root,
                                transform=tf3,
                                frame=config.base_frame + config.base_interval * i,
                                crop=config.crop_by_bb,
                                partition=partition)
  
      if truncate:
        print("shrinking dataset from %d" % len(imgs_curr))
        imgs_curr = TruncatedDataset(imgs_curr, pc=truncate_pc)
        print("... to %d" % len(imgs_curr))

      if tencrop:
        imgs_curr = TenCropAndFinish(imgs_curr, input_sz=config.input_sz,
                                   include_rgb=config.include_rgb)
  
      imgs_list.append(imgs_curr)

  imgs = ConcatDataset(imgs_list)
  dataloader = torch.utils.data.DataLoader(imgs,
                                           batch_size=config.batch_sz,
                                           # full batch
                                           shuffle=shuffle,
                                           num_workers=0,
                                           drop_last=False)

  if not shuffle:
    assert (isinstance(dataloader.sampler,
                       torch.utils.data.sampler.SequentialSampler))
  return dataloader
