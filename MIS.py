import imutils
import numpy as np
import cv2
from matplotlib.pyplot import *
from decimal import *
import math
from imutils import perspective
from scipy.spatial import distance
import sympy as sp
import scipy
from scipy import integrate
from numpy import zeros, polyfit
from numpy import pi
import random
from PIL import Image
import matplotlib


def openImage(path):
    img = cv2.imread(path, 0)
    return img

def toBinary(img: np.ndarray, treshold: int):
    """
    consider the appropriate binary image as follows if the number smaller than the threshold is turned white and large becomes black
    :param img: 
    :param treshold: 
    :return: binary image
    """""
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
    """
    We will go over the original image and by the mask image we will lower the neurons.
    If the mask are white pixel we will change it to black and if white pixel leave unchanged.
    So according to the mask that found the neurons we will remove from the image the neurons and get image with oxons only.
    :param img:
    :param mask:Image only with neurons
    :return:image without neurons
    """
    # mask = cv2.blur(mask, (11, 11))
    # mask = cv2.blur(mask, (5, 5))
    rows, cols = mask.shape
    result = np.zeros([rows, cols], int)
    for i in range(0, rows):
        for j in range(0, cols):
            if mask[i, j] == 0:
                result[i, j] = 255
            else:
                result[i, j] = img[i, j]
    result = result.astype(np.uint8)
    return result

def Polynom(contour):
    """
    for connected commonent finds its polynomial.
    :param contour:
    :return:polynomial look like: [a, b, c] This means: ax^2 +bx+c
    """
    mx = []
    my = []
    for p in contour:
        x, y = p[0]
        mx.append(x)
        my.append(y)
    p1 = np.polyfit(mx, my, 2)
    return p1
def findLengthOfOxons(contours,pol):
    """
    ************fix*******
    Finds the length of the curve on the polynomial from the max point to the min point for all the polynomials of the image
    Calculate the length of the curve according to the curve finding formula by integral of:
  Root containing one and polynomial.
    :param pol:
    :return:
    """
    counter = Decimal(0)
    getcontext().prec = 28
    for i in range(len(pol)):
        y, x, w, h = cv2.boundingRect(contours[i])
        xmin = x
        xmax = x+h
        mx = []
        my = []
        for p in contours[i]:
            x, y = p[0]
            mx.append(x)
            my.append(y)
        #parameter_optimal, cov = scipy.optimize.curve_fit(func, mx, my, p0=pol[i])
        counter = Decimal(counter) + Decimal((integral_calc(pol[i], xmax) - integral_calc(pol[i], xmin)))

    # parameter_optimal, cov = scipy.optimize.curve_fit(func, mx, my, p0=pol[124])
    # print("paramater =", parameter_optimal)
    # y = func(mx, *parameter_optimal)
    # print("y:", y)
    # plot(mx, my, 'o')
    # plot(mx, y, 'r--')
    # show()
    return counter

def func(x,a,b,c):
    fn = lambda x, a, b, c: (a*x**2+b*x+c)
    den = integrate.quad(fn, 0.0, b/c)[0]
    num = np.asarray([integrate.quad(fn, _x/c, b/c)[0] for _x in x])
    return num/den

def integral_calc(pol, xmax):
    """
    Calculate the length of the curve according to the curve finding formula by integral of:
  Root containing one and polynomial.
    :param pol:
    :param xmax:
    :return: The length of the curve
    """
    a = pol[0]
    b = pol[1]
    ax_b = a * xmax + b
    lan = np.abs(ax_b + math.sqrt(1 + math.pow(ax_b, 2)))
    return 1 / a * ((1 / 2 * ax_b) * math.sqrt(1 + math.pow(ax_b, 2)) + 1 / 2*math.log(np.abs(lan), math.e))

def findThickOfExons(img,contours,pol):
    counter = 0
    sum = 0
    for i in range(len(pol)):
        y, x, w, h = cv2.boundingRect(contours[i])
        x1 = x
        x2 = x+h
        for j in range(x1, x2, 3):
            "find the rationa equation for the point"
            y = pol[i][0] * j ** 2 + pol[i][1] * j + pol[i][2]  # ax1^2 +bx1 +c
            tm, tn = findTangent(y, j, pol[i])
            rm, rn = findRationa(y, j, tm)
            img = img.copy()
            ir, ic =img.shape
            result = np.zeros((ir+10, ic+10))
            result[:img.shape[0], :img.shape[1]] = img
            j = int(round(j))
            y = int(round(y))
            for r in range(j-5, j+5):
                for c in range(y-5, y+5):
                    if r > 580:
                        r = 580
                    if r < 0:
                        r = abs(r)
                    if c < 0:
                        c = abs(c)
                    if c > 438:
                        c = 438
                    if result[r][c] == 255 and rn == c - rm*r:
                        sum += 1
            counter += 1
    return sum/counter

def findTangent(y1,x1,pol):
    m =pol[0]*x1+pol[1]# m = f'(x)=ax1+b
    n=y1-m*x1
    return (m , n)

def findRationa(y,x,tm):
    rm = -1/tm
    rn = y+(1/tm)*x
    return rm, rn

def findThickAndLength(binary, contours):
    """

    :param binary:
    :return:
    """

    pol = []
    for contour in contours:
        if cv2.contourArea(contour) < 3:
            continue
        pol.append(Polynom(contour))
    length = findLengthOfOxons(contours, pol)
    print("length:", length)
    thick = findThickOfExons(binary, contours, pol)
    print("thick:", thick)
    return length, thick

    print(leng)
    cv2.imshow("please work!", mask)
    cv2.waitKey(0)


def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


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
                        if countLines not in cont.keys():
                            cont[countLines] = []
                            cont[countLines].append((i, j))
                        else:
                            cont[countLines].append((i, j))
                    else:
                        print("Nope..")
        countLines=countLines+1
    return cont

def start(path = "im1.tif", binaryTresh = 45, maskTresh = 150):
    img = openImage(path)
    mask = toBinary(img, maskTresh)
    img_erosion = cv2.erode(mask, np.ones((17, 17), np.uint8), iterations=1)
    # img_dilation = cv2.dilate(img, np.ones((5,5), np.uint8), iterations = 1)
    cv2.imwrite("mask_after_erosion.jpg", img_erosion)
    img = openImage("im1.tif")
    img_binary = toBinary(img, binaryTresh)
    cv2.imwrite("img_to_binary.jpg", img_binary)
    result = deleteNeuron(img_binary, img_erosion)
    cv2.imwrite("after_deletion.jpg", result)
    result = cv2.resize(result, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    "change colors"
    result = cv2.bitwise_not(result)
    cv2.imwrite("after_change_color.jpg", result)
    contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    length1, width1 = method1(result, contours)
    length2, width2 = method2(result, contours)
    return length1, width1, length2, width2


def method1(result, contours):
    # lines = cv2.HoughLines(result, 4, np.pi / 180, 10)  # lines=None, minLineLength=None, maxLineGap=None)
    length, width = findThickAndLength(result, contours)
    print("Method1 - avg length: ", length, "avg width: ", width)
    return length, width

def method2(binary, contours):
    # getting ROIs with findContours
    #contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    print("number of countor[0] = " + str(len(contours[0])))
    print("number of countor = " + str(len(contours)))
    #c_im = np.zeros(binary.shape)
    #cv2.drawContours(c_im, contours[105], -1, 255, 1)
    mask = np.zeros(binary.shape)
    pixelsPerMetric = None
    list_length = []
    list_width = []
    for (i, c) in enumerate(contours):
        # check if the size of the contour is bigger then 1 pixel.
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
        # show the output image of each contour
        #cv2.imshow("Image", mask)
        #cv2.waitKey(0)

    #calculate avg length and width of exons
    avg_length = sum(list_length) / len(list_length)
    avg_width = sum(list_width) / len(list_width)
    #cv2.imshow("check all sizes", mask)
    #cv2.waitKey(0)
    print("Method2 - avg length: ", avg_length*300, "avg width: ", avg_width*300)
    return avg_length, avg_width


#def main():
#    start()
#if __name__ == '__main__':
#    main()