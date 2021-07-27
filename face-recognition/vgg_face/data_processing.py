import cv2
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image

def preprocess(img_path):
  img = cv2.imread(img_path)
  original_image = img.copy()

  # detected face region
  # region = detect_face(img)

  resize_factor = min(224/img.shape[0], 224/img.shape[1])
  new_size = (int(img.shape[1]*resize_factor), int(img.shape[0]*resize_factor))
  img = cv2.resize(img, new_size)

  img = np.expand_dims(image.img_to_array(img), axis=0)
  img /= 255 # normalise image

  return img #, region

def generate_data(mode):
  train_file_name = "lfw\\train.txt"
  test_file_name = "lfw\\test.txt"
  all_file_name = "lfw\\all.txt"

  if (mode == "train"):
    file_name = train_file_name
  elif (mode == "test"):
    file_name = test_file_name
  elif (mode == "all"):
    file_name = all_file_name
  else:
    raise NameError("Invalid mode selected - Choose one of train, test, all")

  generated_data = []
  
  with open(file_name) as f:
    data_points = int(f.readline())
    data = f.readlines()
    positive_pairs = data[:data_points]
    negative_pairs = data[data_points:(data_points*2)]


  length = 4
  padding = "0"

  for pair in positive_pairs:      
    pair = pair[:-1].split("\t")
    
    name = pair[0]
    pos_1 = f"{pair[1]:{padding}>{length}}"
    pos_2 = f"{pair[2]:{padding}>{length}}"

    data = [f"{name}_{pos_1}.jpg", f"{name}_{pos_2}.jpg", True]
    data[0] = _get_image_path(data[0])
    data[1] = _get_image_path(data[1])

    generated_data.append(data)
      

  for pair in negative_pairs:
    pair = pair[:-1].split("\t")

    name_1 = pair[0]
    name_2 = pair[2]
    pos_1 = f"{pair[1]:{padding}>{length}}"
    pos_2 = f"{pair[3]:{padding}>{length}}"
    
    data = [f"{name_1}_{pos_1}.jpg", f"{name_2}_{pos_2}.jpg", False]
    data[0] = _get_image_path(data[0])
    data[1] = _get_image_path(data[1])

    generated_data.append(data)
  
  return generated_data

def _get_image_path(label):
  img_dir = "lfw_cropped"
  # img_dir = "lfw"
  
  label = img_dir + "\\" + label[:-9] + "\\" + label
  return label

def resize(img, width):
  orig_width = img.shape[0]
  ratio = width / orig_width
  
  height = int(img.shape[1] * ratio)
  image = cv2.resize(img, (height, width))

  return image
