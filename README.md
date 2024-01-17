## Introduction

Image filtering is one of the most fundamental tasks in Image Processing. It has been used for very different purposes such as image smoothing and edge detection. The main aim of image smoothing is to remove the high-frequency component of the given image and to obtain the low-frequency component. Smoothing is carried out by convolving an image with a low-pass filter like the Gaussian filter kernel. The goal of edge detection is to determine the pixels where the brightness values are changed abruptly. Sobel and Prewitt filters are some of the examples of edge detection filters.

Many image editing tools like Photoshop let the user perform some special filters on the images for various artistic effects. For example, one can obtain cartoon-like images, pencil drawing images, etc. Real-Time Video Abstraction by Winnemoller et al.[1] is one such study where the cartoon-like outputs are obtained as follows: A color image is first smoothed by a non-linear image filter, and then the edges are extracted. After that, the smoothed image is quantized to represent the image with less number of colors for the cartoon effect. Lastly, the edges and the quantized image are combined. You may refer to the article for the details and to understand how those processes are performed.

## Description of Given Algorithms and Chosen Methods

### Smoothing
Gaussian Filter as its name implies, has the shape of the function ‘Gaussian distribution’ to define the weights inside the kernel, which are used to compute the weighted average of the neighboring points (pixels) in an image.
<p align="center"> 
<img src=https://github.com/akifozgur/cartoon-effect-to-colorful-images/blob/main/img/gaussian.png>
</p>
In other words, each value in the Gaussian filter is from the zero mean Gaussian distribution. One thing we need to keep in mind is that the kernel size is dependent of standard deviation 𝛔 of Gaussian function:
<p align="center"> 
<img src=https://github.com/akifozgur/cartoon-effect-to-colorful-images/blob/main/img/sigma.png>
</p>
Median Filter is one of Non-linear filters, which is also used for smoothing. Its basic idea is to replace each pixel by the median of its neighboring pixels.
By doing so, it removes some spikes introduced by noise: especially impulse and salt & pepper noise. This is because stand- alone noise pixels with extreme intensities like black and white cannot survive after median filtering. Another advantage of median filter is that it does not introduce new pixel values since it only re-use existing pixel values from window.
In this project I used the recommended scipy library to apply gaussian and median filters.

### Edge Detection
  Difference of Gaussian (DoG) is a technique used for edge
detection in image processing. It's created by taking the difference
between Gaussian filters to emphasize differences between pixels in
an image and highlight edges.
  Firstly, Gaussian filters at different scales (with different standard
deviations) are created. Each Gaussian filter helps smooth and flatten
the image. Then, the difference between these Gaussian filters at
different scales forms the DoG filter. This difference highlights edges in
the image. The DoG filter is used as an edge detection method to
identify edges in the image. It emphasizes significant changes in pixel
values, particularly sudden transitions found at edges. The resulting
DoG image is subjected to a thresholding process to determine edges.
<p align="center"> 
<img src=https://github.com/akifozgur/cartoon-effect-to-colorful-images/blob/main/img/edge.png>
</p>

### Image Quantization
Image quantization is a process used in digital image processing
to reduce the number of distinct colors or levels of brightness in an
image while preserving its visual quality to a reasonable extent. It
involves representing the image with a reduced color palette or fewer
bits per pixel, thereby reducing the overall file size.
The process of image quantization typically involves these steps:
Color Space Conversion: The image is often initially converted
from a high-color space (such as RGB with millions of colors) to a
lower-color space, like an indexed color space (such as paletted
images or indexed color modes), which allows a limited number of
colors to be used.
Note: In this project, I converted the RGB images to HSV and Lab
images.
Color Reduction: The main technique in quantization is reducing
the number of distinct colors in the image. This can be done using
various algorithms like:

- Uniform Quantization: Dividing the color space into a fixed
number of equally sized regions and assigning representative
colors for each region.
- Median Cut Algorithm: Dividing the color space into boxes
and averaging the colors within each box to represent the box.
- K-Means Clustering: Grouping similar colors together by
iteratively assigning pixels to centroids and adjusting centroids to
minimize the distance between pixels and centroids.

Note: In this project, I used K-Means Clustering as Color Reduction.

Palette Creation: After reducing the colors, a palette is generated
that contains the representative colors chosen during the quantization
process.

Assigning Indices: Each pixel in the image is then mapped or
associated with an index in the generated palette, effectively replacing
the original colors with indices pointing to colors in the palette.

Reconstruction: Finally, the image is reconstructed using the
reduced color palette. This reconstructed image often looks very
similar to the original but with a smaller file size due to the reduced
color information.

### Combining Edge and Quantized Image

I took the inverse of the binary image I extracted using the
Difference of Gaussian algorithm and multiplied it elementary-wise
with the quantized image. In this way, I obtained images with clearer
edges and more cartoon-like appearance.

<p align="center"> 
<img src=https://github.com/akifozgur/cartoon-effect-to-colorful-images/blob/main/img/result.png>
</p>



[1] Holger Winnem ̈oller, Sven C Olsen, and Bruce Gooch. Real-time video abstrac- tion. In ACM Transactions On Graphics (TOG), volume 25, pages 1221-1226. ACM, 2006.
