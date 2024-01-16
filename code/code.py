import cv2
import os
import numpy as np
import matplotlib.pyplot as plt 
import scipy
from skimage import io, color
from sklearn.cluster import KMeans
from pathlib import Path


class CartoonEffect:

    def __init__(self, img_name, img_path, sigma_value=1.2, k_value=1.2, threshold=5, color_space="lab", clusters_num=4):
        self.__img_name = img_name
        self.__img_path = img_path
        self.__sigma_value = sigma_value
        self.__k_value = k_value
        self.__threshold = threshold
        self.__color_space = color_space
        self.__clusters_num = clusters_num
        self.__img = io.imread(img_path)
        self.__result_directory = os.path.join(Path(self.__img_path).resolve().parents[1],'result_images')

        if not os.path.exists(self.__result_directory):
            os.makedirs(self.__result_directory)

        plt.rcParams['figure.figsize'] = (15.0, 12.0)
        plt.axis('off')

    def __image_smoothing(self):
        sigma = (self.__sigma_value, self.__sigma_value, 0)

        smoothed_img = scipy.ndimage.gaussian_filter(input = self.__img, sigma = sigma)
        #smoothed_img_median = scipy.ndimage.median_filter(input = self.__img, size = size)

        if not os.path.exists(os.path.join(self.__result_directory, "smoothing")):
            os.makedirs(os.path.join(self.__result_directory, "smoothing"))

        plt.imsave(os.path.join(self.__result_directory, "smoothing",
                                self.__img_name + "_sigma=" +
                                str(self.__sigma_value) +"_smoothing.jpg"), smoothed_img)

        return smoothed_img

    def __edge_detection(self):
        sigma_1 = self.__sigma_value
        sigma_2 = self.__sigma_value * self.__k_value

        img_with_gaussian_1 = scipy.ndimage.gaussian_filter(input = self.__img, sigma = sigma_1)
        img_with_gaussian_2 = scipy.ndimage.gaussian_filter(input = self.__img, sigma = sigma_2)

        DoG = np.subtract(img_with_gaussian_1, img_with_gaussian_2)
        DoG_gray = cv2.cvtColor(DoG, cv2.COLOR_RGB2GRAY)
        edges = np.where(DoG_gray <= self.__threshold, 1, 0)

        if not os.path.exists(os.path.join(self.__result_directory, "edges")):
            os.makedirs(os.path.join(self.__result_directory, "edges"))

        plt.imsave(os.path.join(self.__result_directory, "edges",
                                self.__img_name + "_sigma=" +
                                str(self.__sigma_value) + "_k=" +
                                str(self.__k_value) + "_threshold=" +
                                str(self.__threshold) + "_edges.jpg"), edges, cmap="gray")

        return edges

    def __image_quantization(self, smoothed_img):
        if self.__color_space == "lab":
            converted_image = color.rgb2lab(smoothed_img)
        elif self.__color_space == "hsv":
            converted_image = color.rgb2hsv(smoothed_img)

        last_channel = converted_image[:, :, 0]

        last_channel_2d = last_channel.reshape((-1, 1))

        kmeans = KMeans(n_clusters=self.__clusters_num)
        kmeans.fit(last_channel_2d)
        quantized_labels = kmeans.predict(last_channel_2d)

        quantized_l_channel = kmeans.cluster_centers_[quantized_labels]
        quantized_l_channel = quantized_l_channel.reshape(last_channel.shape)

        converted_image_quantized = converted_image.copy()
        converted_image_quantized[:, :, 0] = quantized_l_channel

        if self.__color_space == "lab":
            quantized_rgb_image = color.lab2rgb(converted_image_quantized)
        elif self.__color_space == "hsv":
            quantized_rgb_image = color.hsv2rgb(converted_image_quantized)

        if not os.path.exists(os.path.join(self.__result_directory, "quantized")):
            os.makedirs(os.path.join(self.__result_directory, "quantized"))

        plt.imsave(os.path.join(self.__result_directory, "quantized",
                                self.__img_name + "_sigma=" +
                                str(self.__sigma_value) + "_colorSpace=" +
                                self.__color_space+ "_clusterNum=" +
                                str(self.__clusters_num) + "_quantized.jpg"), quantized_rgb_image)

        return quantized_rgb_image

    def combining_edge_and_quantized_image(self):

        smoothed_img = self.__image_smoothing()
        edges = self.__edge_detection()
        quantized_img = self.__image_quantization(smoothed_img=smoothed_img)

        inverse = 1- edges
        final_img = np.multiply(quantized_img,inverse[...,None])

        if not os.path.exists(os.path.join(self.__result_directory, "final")):
            os.makedirs(os.path.join(self.__result_directory, "final"))

        plt.imsave(os.path.join(self.__result_directory, "final",
                                self.__img_name + "_sigma=" +
                                str(self.__sigma_value) + "_k=" +
                                str(self.__k_value) + "_threshold=" +
                                str(self.__threshold) + "_colorSpace=" +
                                self.__color_space+ "_clusterNum=" +
                                str(self.__clusters_num) + "_final.jpg"), final_img)

img_name = "img1.jpg" #img2.jpg, img3.jpg, img4.jpg, img5.jpg, img6.jpg, img7.jpg
img_dir_path = os.path.join(Path(__file__).resolve().parents[1], "data")
img_path = os.path.join(img_dir_path, img_name)

filter = "gaussian"
sigma_value = 1.2
k_value = 1.2
threshold = 5
color_space = "lab" #hsv
clusters_num = 4



cartoonEffect_Gaussian = CartoonEffect(img_name = img_name,
                                    img_path=img_path,
                                    sigma_value=sigma_value, 
                                    k_value=k_value, 
                                    threshold=threshold, 
                                    color_space=color_space, 
                                    clusters_num=clusters_num)


cartoonEffect_Gaussian.combining_edge_and_quantized_image()