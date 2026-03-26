import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def histogramEqual():
    img = cv.imread('demoimages/badquality.jpg', cv.IMREAD_GRAYSCALE)

   

    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdfNorm = cdf * float(hist.max()) / cdf.max()

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.plot(hist, label="Histogram")
    plt.plot(cdfNorm, color='r', label="CDF")
    plt.xlabel("Intensity")
    plt.ylabel("no of pixels")
    plt.title("Histogram & CDF")
    plt.legend()

    plt.show()

    equImg = cv.equalizeHist(img)

    equhist = cv.calcHist([equImg], [0], None, [256], [0, 256])
    equcdf = equhist.cumsum()
    equcdfNorm = equcdf * float(equhist.max()) / equcdf.max()

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(equImg, cmap='gray')
    plt.title("Equalized Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.plot(equhist, label="Histogram")
    plt.plot(equcdfNorm, color='r', label="CDF")
    plt.xlabel("Intensity")
    plt.ylabel("no of pixels")
    plt.title("Histogram & CDF")
    plt.legend()

    plt.show()

    claheObj = cv.createCLAHE(clipLimit=5, tileGridSize=(8,8))
    claheImg = claheObj.apply(img)
    clahehist = cv.calcHist([claheImg], [0], None, [256], [0, 256])
    clahecdf = clahehist.cumsum()
    clahecdfNorm = clahecdf * float(clahehist.max()) / clahecdf.max()

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(claheImg, cmap='gray')
    plt.title("CLAHE Image")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.plot(clahehist, label="Histogram")
    plt.plot(clahecdfNorm, color='r', label="CDF")
    plt.xlabel("Intensity")
    plt.ylabel("no of pixels")
    plt.title("Histogram & CDF")
    plt.legend()

    plt.show()

histogramEqual() 