from distutils import dist

import imutils as imutils
import numpy as np
import cv2
import math
from imutils import perspective
from scipy.spatial import distance

pixelsPerMetric = None
width = 20

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(1500 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()

def openImage(path):
    img = cv2.imread(path, 0)
    return img


def toBinary(img: np.ndarray, treshold: int):
    rows, cols = img.shape
    for i in range(rows):
        for j in range(cols):
            if img[i, j] > treshold:
                img[i, j] = 0
            else:
                img[i, j] = 255
    return img
    # img = cv2.threshold(img, 33, 255, cv2.THRESH_BINARY)[1]
    # return img

def deleteNeuron(img: np.ndarray, mask: np.ndarray):
    # mask = cv2.blur(mask, (11, 11))
    # mask = cv2.blur(mask, (5, 5))
    #cv2.imshow("mask", mask)
    #cv2.waitKey(0)
    rows, cols = mask.shape
    result = np.zeros([rows, cols], int)
    for i in range(0, rows):
        for j in range(0, cols):
            if mask[i, j] == 0:
                result[i, j] = 255
                # result[i-10:i+10, j-10:j+10] = random.randint(33, 37)
            else:
                result[i, j] = img[i, j]
    result = result.astype(np.uint8)
    return result


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def method2(binary, contours):
    # getting ROIs with findContours
    print("number of countor = " + str(len(contours)))
    mask = np.zeros(binary.shape)
    pixelsPerMetric = None
    list_length = []
    list_width = []
    for (i, c) in enumerate(contours):
        if cv2.contourArea(c) < 2:
            continue
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        print("Box "+str(i)+": \n", box)
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        print("perspective "+str(i)+": ", box.astype("int"))
        print("")
        cv2.drawContours(mask, [box.astype("int")], 0, 255, 2)
        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(mask, (int(x), int(y)), 5, 255, -1)

    #compute mid-point:
        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(mask, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(mask, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(mask, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(mask, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(mask, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
        cv2.line(mask, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)
        # compute the Euclidean distance between the midpoints
        dA = distance.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = distance.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            print("dB = ", dB)
            pixelsPerMetric = dB / 0.15

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        print("length = ", dimB)
        print("width = ", dimA)


        # draw the object sizes on the image
        cv2.putText(mask, "{:.1f}in".format(dimB),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
        cv2.putText(mask, "{:.1f}in".format(dimA),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
        # measuring the length of a contour in pixels:
        #contourLength = cv2.arcLength(c, True)
        list_length.append(max(dimA, dimB))
        list_width.append(min(dimA, dimB))
        #cv2.putText(mask, str(contourLength),
                    #(int(50 + 10), int(50)), cv2.FONT_HERSHEY_SIMPLEX,
                    #0.65, (0, 255, 255), 2)
        # show the output image
        #cv2.imshow("Image", mask)
        #cv2.waitKey(0)

    avg_length = sum(list_length) / len(list_length)
    avg_width = sum(list_width) / len(list_width)
    print("Method2 - avg length: ", avg_length*300, "avg width: ", avg_width*300)
    return avg_length, avg_width


def props_for_contours(contours, ary):
    """Calculate bounding box & the number of set pixels for each contour."""
    c_info = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        c_im = np.zeros(ary.shape)
        cv2.rectangle(ary, (x, y), (x + w, y + h), (0, 255, 0), 2)
        c_info.append({
            'x1': x,
            'y1': y,
            'x2': x + w - 1,
            'y2': y + h - 1,
            'sum': np.sum(ary * (c_im > 0))/255
        })
    #cv2.imshow("123.jpg", ary)
    #cv2.waitKey(0)
    return c_info


def find_dots(img, lines):
    rows, cols = img.shape
    cont = {}
    countLines = 0
    for line in lines:
        print("Line " + str(countLines) + " : ")
        rho, theta = line[0]
        for i in range(0, rows-1):
            for j in range(0, cols-1):
                if img[i][j] == 255 and (math.sin(theta) != 0):
                    right = (i)*math.cos(theta) + (j)*math.sin(theta)
                    print("right: ", right)
                    print("rho: ", rho)
                    if abs(rho - right) <= 10:
                        print("find point!!!!!!!!!!!!")
                        if countLines not in cont.keys():
                            cont[countLines] = []
                            cont[countLines].append((i, j))
                        else:
                            cont[countLines].append((i, j))
                    else:
                        print("Nope..")
        countLines=countLines+1
    return cont

def main():
    img = openImage("im1.tif")
    # crop_img = img[0:0 + 400, 0:0 + 400]
    #crop_img = img

    mask = toBinary(img, 150)
    img_erosion = cv2.erode(mask, np.ones((17, 17), np.uint8), iterations=1)
    # img_dilation = cv2.dilate(img, np.ones((5,5), np.uint8), iterations = 1)
    cv2.imwrite("mask_after_erosion.jpg", img_erosion)

    img = openImage("im1.tif")
    img_binary = toBinary(img, 45)
    cv2.imwrite("img_to_binary.jpg", img_binary)

    result = deleteNeuron(img_binary, img_erosion)
    cv2.imwrite("after_deletion.jpg", result)
    result = cv2.resize(result, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    # change colors
    result = cv2.bitwise_not(result)

    cv2.imwrite("after_change_color.jpg", result)

    #lines = cv2.HoughLines(result, 4, np.pi / 180, 50) #lines=None, minLineLength=None, maxLineGap=None)
    #lines = cv2.HoughLinesP(crop_img, 1, np.pi / 180, 50, 2, 19)
    contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    method2(result, contours)


    # ret, labels = cv2.connectedComponents(result, connectivity=8)

#main()