import cv2 as cv
from random import *
import numpy as np

import re
import os

from os import listdir
from os.path import isfile, join

captures = dict()

f = [f for f in listdir(".") if isfile(join(".", f))]
for f in [f for f in listdir(".") if isfile(join(".", f))]:
    m = re.match("capture_([^_]+)_(\d+).jpg", f)
    if m is not None:
        # print(m[1], m[2])
        captures[m[1]] = int(m[2])

print(captures)

"""
EVENT_MOUSEMOVE     = 0,
EVENT_LBUTTONDOWN   = 1,
EVENT_RBUTTONDOWN   = 2,
EVENT_MBUTTONDOWN   = 3,
EVENT_LBUTTONUP     = 4,
EVENT_RBUTTONUP     = 5,
EVENT_MBUTTONUP     = 6,
EVENT_LBUTTONDBLCLK = 7,
EVENT_RBUTTONDBLCLK = 8,
EVENT_MBUTTONDBLCLK = 9,
EVENT_MOUSEWHEEL    = 10,
EVENT_MOUSEHWHEEL   = 11,
"""

# les couleurs de base
R = (0, 0, 255)
G = (0, 255, 0)
B = (255, 0, 0)
M = (255, 0, 255)
C = (255, 255, 0)
Y = (0, 255, 255)
BL = (0, 0, 0)
W = (255, 255, 255)

HERE = os.path.normpath(os.path.dirname(__file__)).replace("\\", "/")
TOP = os.path.dirname(HERE)
DATA = TOP + "/AnumbyFormes"
FOND = TOP + "/AnumbyVehicule"

print(HERE, DATA)

forms = ["Rond", "Square", "Triangle", "Star5", "Star4", "Eclair", "Coeur", "Lune", "fond"]
#           0       1          2          3        4        5         6        7       8

# count existing forms


start = None
end = None
def on_click(event, x, y, p1, p2):
    global frame, start, end

    if event == cv.EVENT_LBUTTONDOWN:
        # print("mouse down", x, y)
        start = (x, y)
    elif event == cv.EVENT_MOUSEMOVE:
        # print("mouse move", x, y)
        pass
    elif event == cv.EVENT_LBUTTONUP:
        if start is not None:
            print("mouse up", start, end)
            end = (x, y)


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

figure_id = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cv.imshow("frame", frame)

    quit = False
    cv.namedWindow('frame')
    cv.setMouseCallback('frame', on_click)

    k = cv.waitKey(0)

    zero = 48

    """
    On peut sélectionner 
    - soit une figure, et dans ce cas on donne son numéro [0..7]
    - soit le fond et dans ce cas on attribut le numéro "8"
    """

    print(">>> k =", k, start, end)

    if k >= zero + 0 and k <= zero + 8:
        # c'est une figure valable
        if start is not None:
            x1 = start[0]
            y1 = start[1]
            x2 = end[0]
            y2 = end[1]
            w = x2 - x1
            h = y2 - y1
            w = min(w, h)
            h = w
            figure = np.zeros((h, w, 3), np.float64)
            figure = frame[y1: y1+h, x1:x1+w, :]
            try:
                resized = cv.resize(figure, (80, 80), interpolation=cv.INTER_AREA)
            except:
                continue

            fig = forms[k - zero]
            if fig in captures:
                figure_id = captures[fig] + 1
            else:
                figure_id = 0
            captures[fig] = figure_id
            name = "capture_{}_{:03d}.jpg".format(fig, figure_id)
            print("mouse up", start, end, name)
            cv.imwrite("capture_{}_{:03d}.jpg".format(forms[k - zero], figure_id), resized)
            cv.imshow(name, figure)
            print(captures)
            k = cv.waitKey(0)
            cv.destroyWindow(name)

    elif k == 27:
        quit = True
        break
    elif k == 113:
        quit = True
        break

    start = None
    end = None

"""
images = []
for form in forms:
    image = cv.imread('capture_{}.jpg'.format(form))
    resized = cv.resize(image, (80, 80), interpolation=cv.INTER_AREA)
    # src_gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

    cv.imwrite(TOP + "/dataset/{}/RawImages{}.jpg".format(form, form), resized)

    # cv.imshow(form, resized)
    # cv.waitKey()

    images.append(image)
"""


