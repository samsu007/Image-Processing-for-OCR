# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 20:05:53 2021

@author: samsu
"""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt



st.title("Image processing for OCR using OpenCV")

file = st.file_uploader("Upload Image")


def resize_image(img, flag, MAX_PIX):
    """
    Resize an RGB numpy array of an image, either along the height or the width, and keep its aspect ratio. Show restult.
    """
    h, w, c = img.shape

    if flag == 'h':
        dsize = (int((MAX_PIX * w) / h), int(MAX_PIX))
    else:
        dsize = (int(MAX_PIX), int((MAX_PIX * h) / w))

    img_resized = cv2.resize(
        src=img,
        dsize=dsize,
        interpolation=cv2.INTER_CUBIC,
    )

    h, w, c = img_resized.shape
    print(f'Image shape: {h}H x {w}W x {c}C')

    return img_resized

def apply_morphology(img, method):
    """
    Apply a morphological operation, either opening (i.e. erosion followed by dilation) or closing (i.e. dilation followed by erosion). Show result.
    """
    if method == 'open':
        op = cv2.MORPH_OPEN
    elif method == 'close':
        op = cv2.MORPH_CLOSE

    img_morphology = cv2.morphologyEx(
        src=img,
        op=op,
        kernel=np.ones((5, 5), np.uint8),
    )

    return img_morphology

def apply_adaptive_threshold(img, method):
    """
    Apply adaptive thresholding, either Gaussian (threshold value is the weighted sum of neighbourhood values where weights are a Gaussian window) or mean (threshold value is the mean of neighbourhood area). Show result.
    """
    img = cv2.cvtColor(
        src=img,
        code=cv2.COLOR_RGB2GRAY,
    )

    if method == 'gaussian':
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    elif method == 'mean':
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C

    img_adaptive = cv2.adaptiveThreshold(
        src=img,
        maxValue=255,
        adaptiveMethod=adaptive_method,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    return img_adaptive

def apply_sobel(img, direction):
    print("0",img)


    img = cv2.cvtColor(
        src=img,
        code=cv2.COLOR_RGB2GRAY,
    )

    print("1")

    if direction == 'h':
        dx, dy = 0, 1
    
        print("2")


    elif direction == 'v':
        dx, dy = 1, 0

        print("3")

    img_sobel = cv2.Sobel(
        src=img,
        ddepth=cv2.CV_64F,
        dx=dx,
        dy=dy,
        ksize=5,
    )
    print("4",img_sobel)


    return img_sobel

def apply_laplacian(img):

    img = cv2.cvtColor(
        src=img,
        code=cv2.COLOR_RGB2GRAY,
    )

    img_laplacian = np.uint8(
        np.absolute(
            cv2.Laplacian(
                src=img,
                ddepth=cv2.CV_64F,
            )
        )
    )


    return img_laplacian



if file is not None:
    image = Image.open(file)

    image = np.array(image)

    if st.sidebar.checkbox("Show Image"):
        st.write("Original Image : ")
        original_image = st.image(image)

    if st.sidebar.checkbox("Show Crop Image"):
        ymin = st.sidebar.slider(
            "ymin", min_value=0, value=50, max_value=image.shape[0])
        ymax = st.sidebar.slider(
            "ymax", min_value=0, value=-50, max_value=image.shape[0])
        xmin = st.sidebar.slider(
            "xmin", min_value=0, value=50, max_value=image.shape[1])
        xmax = st.sidebar.slider(
            "xmax", min_value=0, value=-50, max_value=image.shape[1])
        # print(int(min),int(ymax),int(xmin),int(xmax))

        st.write(image.shape)

        image = image[
            int(ymin): int(ymax),
            int(xmin): int(xmax),
        ]

        h, w, c = image.shape
        st.write("Cropped Image : ")
        cropped_image = st.image(image)

        st.write("Shape of Cropped Image : ", (h, w, c))

    if st.sidebar.checkbox("Apply Gaussian blurring"):

        image = cv2.GaussianBlur(
            src=image,
            ksize=(5, 5),
            sigmaX=0,
            sigmaY=0,
        )
        st.write("Applying Blur On Image : ")
        st.image(image)

    st.sidebar.subheader("Border : ")

    if st.sidebar.checkbox("White"):

        image = cv2.copyMakeBorder(
            src=image,
            top=10,
            bottom=10,
            left=10,
            right=10,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        st.write("Show Bordered Image : ")
        bordered_image = st.image(image)
        

    if st.sidebar.checkbox("Black"):

        image = cv2.copyMakeBorder(
            src=image,
            top=10,
            bottom=10,
            left=10,
            right=10,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        st.write("Show Bordered Image : ")
        bordered_image = st.image(image)

    st.sidebar.subheader("Resize : ")

    if st.sidebar.checkbox("Height Base Resized"):
        MAX_PIX = st.text_input("Enter Max Pixel to Resize (based on height - Eg: 10,20,...) : ")
        height = image.shape[1]
        if st.button("Height Resized"):
            image = resize_image(image, 'h',int(MAX_PIX))
            st.write("Resized Image based on Height")
            st.image(image)
     
    if st.sidebar.checkbox("Width Base Resized"):
        MAX_PIX = st.text_input("Enter Max Pixel to Resize (based on width - Eg: 10,20,...) : ")
        width = image.shape[0]
        if st.button("Width Resized"):
            image = resize_image(image, 'w',int(MAX_PIX))
            st.write("Resized Image based on Width")
            st.image(image)

    
    
    st.sidebar.subheader("Morphological operations : ")

    if st.sidebar.checkbox("Opening - erosion followed by dilation using 5 x 5 kernel"):
        image = apply_morphology(image,'open')
        st.write("Morphology Based on Opening Method : ")
        st.image(image)
     
    if st.sidebar.checkbox("Closed - dilation followed by Erosion using 5 x 5 kernel"):
        image = apply_morphology(image,'close')
        st.write("Morphology Based on Closing Method : ")
        st.image(image)

    st.sidebar.subheader("Adaptive thresholding : ")
    
    threshold =  st.sidebar.radio("Types of Thresholding : ",("None","Gaussian","Mean"),index=0)

    if threshold == "Gaussian":
        image = apply_adaptive_threshold(image,'gaussian')
        st.write("Apply Thresholding Based on Gaussian Method : ")
        st.image(image)

    if threshold == "Mean":
        image = apply_adaptive_threshold(image,'mean')
        st.write("Apply Thresholding Based on Mean Method : ")
        st.image(image)
     
    st.sidebar.subheader("Sobel filter : ")

    sobel_filter =  st.sidebar.radio("Types of Filter : ",("None","Horizontal","Vertical"),index=0)

    if sobel_filter == "Horizontal":
        st.write("Sobel Filter on Image - Detect Horizontal Edges")
        image = apply_sobel(image,'h')
        fig, ax = plt.subplots()
        ax.imshow(image,cmap = 'gray')
        plt.axis("off")
        st.pyplot(fig)

    if sobel_filter == "Vertical":
        st.write("Sobel Filter on Image - Detect Vertical Edges")
        image = apply_sobel(image,'v')
        fig, ax = plt.subplots()
        ax.imshow(image,cmap = 'gray')
        plt.axis("off")
        st.pyplot(fig)
    
    st.sidebar.subheader("Laplacian filter : ")

    if st.sidebar.checkbox("Apply Filter"):
        image = apply_laplacian(image)
        st.write("Laplacian Filter on Image : ")
        st.image(image)
    
    st.sidebar.subheader("Encoding Image : ")


    if st.sidebar.checkbox("Encoding Image (optional)"):
        _, buf = cv2.imencode(
            ext=".jpg",
            img=image,
        )

        data = buf.tostring()

        image = cv2.imdecode(
            buf=buf,
            flags=cv2.IMREAD_UNCHANGED,
        )
    
        # st.write(data)
        st.image(image)



