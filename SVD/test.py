import cv2
import numpy as np
from SVD import SVD
import matplotlib.pyplot as plt

path = "D:\\Code\\Python\\Mini_Project\\SVD\\Data\\flower.jpg"

img = cv2.imread(path, 0)
matx = img.astype(np.float64)
print("Original matrix:\n", matx)
print("Shape: ", matx.shape)
print("Rank:", np.linalg.matrix_rank(matx))

svd_b = SVD(matx)
U, lamda, Vt = svd_b.svd()
