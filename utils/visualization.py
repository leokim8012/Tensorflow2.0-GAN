
import matplotlib.pyplot as plt
import numpy as np
import os

import glob
from PIL import Image
def generate_and_save_images(model, filename, test_input, path=os.getcwd() + '/outputs/' ):
  # `training`이 False : (배치정규화를 포함하여) 모든 층들이 추론 모드로 실행됨
  print((model))
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
  plt.suptitle(filename, y=0.95 , size=10, weight=3)

  plt.savefig(path + filename +'.png', dpi=128)


def generate_gif_from_images(
    filename,
    image_path=os.getcwd() + '/outputs/png/', 
    gif_path=os.getcwd() + '/outputs/gif/'
  ):
  fp_in = image_path + '*.png'
  fp_out = gif_path + filename + '.gif'

  img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in), key=os.path.getctime)]
  img.save(fp=fp_out, format='GIF', append_images=imgs,
          save_all=True, duration=100, loop=0)