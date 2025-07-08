import pandas as pd
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import xml.etree.ElementTree as ET

def __get_box_co_ords(tree):
    root = tree.getroot()
    bnd_box = root.find('object').find('bndbox')
    x_min = int(bnd_box.find('xmin').text)
    y_min = int(bnd_box.find('ymin').text)
    x_max = int(bnd_box.find('xmax').text)
    y_max = int(bnd_box.find('ymax').text)

    return (x_min,y_min),(x_max,y_max)