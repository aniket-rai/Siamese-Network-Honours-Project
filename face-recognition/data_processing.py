import os
import numpy as np
import matplotlib.pyplot as plt

from torchvision.io import read_image

# -----------------------------------
# MASTER DATASET - LFW
# -----------------------------------
class LFW:
  def __init__(self):
    self.img_dir = "lfw"
    self.target_transform = _label_target_transform
    self.img_labels = _get_image_labels(self.img_dir);

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, _get_image_path(self.img_labels[idx]))
    
    image = read_image(img_path).permute(1,2,0)
    label = self.target_transform(self.img_labels[idx])

    return image, label

# -----------------------------------
# MASTER DATASET - LFW TRAIN
# -----------------------------------
class LFW_Train:
  def __init__(self):
    self.img_dir = "lfw"
    self.target_transform = _label_target_transform
    self.img_labels = _generate_data(0);

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path_0 = os.path.join(self.img_dir, _get_image_path(self.img_labels[idx][0]))
    img_path_1 = os.path.join(self.img_dir, _get_image_path(self.img_labels[idx][1]))

    image1 = read_image(img_path_0).permute(1,2,0)
    image2 = read_image(img_path_1).permute(1,2,0)

    label = self.img_labels[idx][2]

    return image1, image2, label

# -----------------------------------
# MASTER DATASET - LFW TEST
# -----------------------------------
class LFW_Test:
  def __init__(self):
    self.img_dir = "lfw"
    self.target_transform = _label_target_transform
    self.img_labels = _generate_data(1)

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, _get_image_path(self.img_labels[idx]))
    
    image1 = read_image(img_path[0]).permute(1,2,0)
    image2 = read_image(img_path[1]).permute(1,2,0)
    label = self.img_labels[idx][2]

    return image1, image2, label

# -------------------------------------
# UTILITY FUNCTIONS FOR THE LFW CLASSES
# -------------------------------------
def _get_image_labels(img_dir):
  image_labels = list()

  for (dirpath, dirnames, filenames) in os.walk(img_dir):
    for files in filenames:
      image_labels.append(files)
  
  return image_labels


def _get_image_path(label):
  label = label[:-9] + "/" + label
  return label


def _label_target_transform(label):
  label = label[:-9]
  names = label.split("_")
  string = ""
  
  for name in names:
    string += name

  return string 


# -----------------------------------
# TEST AND TRAIN UTILITY FUNCTIONS
# -----------------------------------
def _generate_data(train_or_test):
  if (train_or_test == 0):
    file_name = "lfw/train.txt"
  elif (train_or_test == 1):
    file_name = "lfw/test.txt"
  else:
    print("Invalid input. Use 0 for train or 1 for test.")
    return None
  
  with open(file_name) as f:
    data_points = int(f.readline())
    data = f.readlines()
    positive_pairs = data[:data_points]
    negative_pairs = data[data_points:(data_points*2)]

  test_data = []

  length = 4
  padding = "0"

  for pair in positive_pairs:      
    pair = pair[:-1].split("\t")
    
    name = pair[0]
    pos_1 = f"{pair[1]:{padding}>{length}}"
    pos_2 = f"{pair[2]:{padding}>{length}}"

    data = [f"{name}_{pos_1}.jpg", f"{name}_{pos_2}.jpg", True]

    test_data.append(data)
      

  for pair in negative_pairs:
    pair = pair[:-1].split("\t")

    name_1 = pair[0]
    name_2 = pair[2]
    pos_1 = f"{pair[1]:{padding}>{length}}"
    pos_2 = f"{pair[3]:{padding}>{length}}"
    
    data = [f"{name_1}_{pos_1}.jpg", f"{name_2}_{pos_2}.jpg", False]

    test_data.append(data)
  
  return test_data
