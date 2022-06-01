import math

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    if img_l.shape != img_r.shape:
        return "Images must be in the same size!"

    disparity_map = np.zeros_like(img_l)  # ans array

    img_l_height = img_l.shape[0]
    img_l_width = img_l.shape[1]

    for i in range(k_size, img_l_height-k_size):
        for j in range(k_size, img_l_width-k_size):
            left_kernel = img_l[i-k_size: i+k_size+1, j-k_size: j+k_size+1]  # left image kernel
            min_error = np.inf
            for x in range(disp_range[0], disp_range[1]):  # going through every possible range
                if j-k_size-x >= 0:
                    # getting the right image kernel by shifting it left x coordinates
                    left_kernel_shift = img_r[i-k_size: i+k_size+1, j-k_size-x: j+k_size+1-x]

                if left_kernel_shift.shape == left_kernel.shape:
                    check = np.sum(np.power(left_kernel-left_kernel_shift, 2))  # formula we have been given
                    if check < min_error:
                        min_error = check
                        disparity_map[i, j] = x

    return disparity_map


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: int, k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    if img_l.shape != img_r.shape:
        return "Images must be in the same size!"

    img_l_height = img_l.shape[0]
    img_l_width = img_l.shape[1]

    disparity_map = np.zeros_like(img_l)
    for i in range(k_size, img_l_height - k_size):
        for j in range(k_size, img_l_width - k_size):
            left_kernel = img_l[i-k_size:i+k_size+1, j-k_size:j+k_size+1]  # left image kernel
            max_response = -np.inf
            for x in range(disp_range[0], disp_range[1]):
                if j - k_size - x >= 0:
                    # getting the right image kernel by shifting it left x coordinates
                    left_kernel_shift = img_r[i-k_size:i+k_size+1, j-k_size-x:j+k_size+1-x]
                if left_kernel_shift.shape == left_kernel.shape:
                    # formula we have been taught
                    ncc = np.sum(np.multiply(left_kernel, left_kernel_shift))
                    ncc /= math.sqrt(np.sum(np.power(left_kernel, 2)) * np.sum(np.power(left_kernel_shift, 2)))
                    if ncc > max_response:
                        disparity_map[i, j] = x
                        max_response = ncc
    return disparity_map


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """
    if src_pnt.shape[0] < 4 or dst_pnt.shape[0] < 4:
        return "Only 4 points and above allowed"
    homography = np.zeros((8, 9))
    index = 0
    for i in range(0, src_pnt.shape[0]):
        # source points
        x1 = src_pnt[i, 0]
        y1 = src_pnt[i, 1]
        # destination points
        x2 = dst_pnt[i, 0]
        y2 = dst_pnt[i, 1]
        # creating an array for homography
        homography[index] = np.array([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2, -x2])
        homography[index+1] = np.array([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
        # incrementing the index
        index += 2

    vh = np.linalg.svd(homography)[2]  # getting the vh
    h = vh[vh.shape[0]-1, :] / vh[vh.shape[0]-1, vh.shape[1]-1]  # getting the h and normalizing
    h = h.reshape((3, 3))
    pred = np.zeros((4, 2))
    for j in range(0, src_pnt.shape[0]):
        temp = np.array([src_pnt[j, 0], src_pnt[j, 1], 1]).reshape((3, 1))  # getting the source coordinates data
        change_in_coor = h @ temp  # dot product to receive the change in the coordinates
        pred[j, 0] = change_in_coor[0, 0] / change_in_coor[2, 0]
        pred[j, 1] = change_in_coor[1, 0] / change_in_coor[2, 0]

    error = np.sqrt(np.sum((pred - dst_pnt)) ** 2)
    return h, error


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    ##### Your Code Here ######

    src_p = []
    fig2 = plt.figure()

    # same as onclick_1, but with the source image instead of the dest
    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        src_p.append([x, y])

        if len(src_p) == 4:
            plt.close()
        plt.show()

    # display image 2, same operations as display image 1
    cid = fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)

    homography = cv.findHomography(src_p, dst_p)[0]  # using cv to find the accurate homography

    source_height = src_img.shape[0]
    source_width = src_img.shape[1]
    warp = np.zeros_like(dst_img)
    # boundaries = np.zeros_like(dst_img)
    for i in range(source_height):
        for j in range(source_width):
            temp = np.array([j, i, 1])
            new_coor = np.dot(homography, temp)  # dot product for calculating the coordinates
            y = int(new_coor[0] / new_coor[new_coor.shape[0]-1])  # getting the new y
            x = int(new_coor[1] / new_coor[new_coor.shape[0]-1])  # getting the new x
            warp[x, y] = src_img[i, j]  # setting that the new coordinate will present the source in the original i and j
            # if warp[x, y] == :
            #     boundaries[x, y] = True
            # else:
            #     boundaries[x, y] = False
    boundaries = warp == 0  # getting where we want to "paste"
    outcome = dst_img * boundaries + (1 - boundaries) * warp  # using a formula we have been taught
    plt.imshow(outcome)
    plt.show()
