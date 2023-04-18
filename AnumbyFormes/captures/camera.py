import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
import os
import re


# les couleurs de base
R = (0, 0, 255)
G = (0, 255, 0)
B = (255, 0, 0)
M = (255, 0, 255)
C = (255, 255, 0)
Y = (0, 255, 255)
BL = (0, 0, 0)
W = (255, 255, 255)
# COLOR_X = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))


# conversion Radians -> degrés
def rad2deg(angle):
    return 180 * angle / np.pi


# conversion degrés -> Radians
def deg2rad(angle):
    return np.pi * angle / 180


# get aspect ratio from a contour
def AspectRatio(cnt):
    try:
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = float(w) / h
    except:
        aspect_ratio = 0
    return aspect_ratio


# get area from a contour
def Area(cnt):
    try:
        area = cv.contourArea(cnt)
    except:
        area = 0
        rect_area = 0
    return area


# get enclosing rectangle corners from a contour
def ExtremePoints(cnt):
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    return [leftmost, topmost, rightmost, bottommost, leftmost]


def BoundingRectangle(cnt):
    # Calcul de l'angle de rotation de l'image détectée
    # 'corners' contient les coins [L, T, R, B]
    #     T
    #   L   R
    #     B

    corners = ExtremePoints(cnt)

    # on calcul l'angle et les points de chaque facette du contour, et le centre du carré
    # et on va moyenner les résultats obtenus pour estimer les valeurs
    xc = 0
    yc = 0
    alpha = 0
    radius = 0
    for i, corner in enumerate(range(4)):
        # coordonnées [(x1, y1), (x2, y2)] de la facette
        x1 = corners[corner][0]
        y1 = corners[corner][1]
        x2 = corners[corner + 1][0]
        y2 = corners[corner + 1][1]

        xc += x1
        yc += y1

        radius += np.sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1))
        if int(x2) != int(x1):
            t = (y2 - y1) / (x2 - x1)
            a = rad2deg(np.arctan(t))
        else:
            a = 90.

        a -= (i % 2) * 90.

        # print(i, "AspectRatio> ", i, a, radius, x1, y1)

        alpha += a

    alpha = alpha / 4.0
    xc = xc / 4.0
    yc = yc / 4.0
    radius = radius / 4.0
    # print(i, "AspectRatio> moyenne", alpha, radius, x1, y1)

    xcorners = [int(xc + radius * np.cos(deg2rad(alpha + 45 + 90 * corner))) for corner in range(5)]
    ycorners = [int(yc + radius * np.sin(deg2rad(alpha + 45 + 90 * corner))) for corner in range(5)]

    return xcorners, ycorners, (xc, yc), alpha, radius


def form_find_figures(src, model, forms, data_min, data_max, pattern):
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    shape = pattern[0].shape
    max_area = (shape[0] - 1) * (shape[1] - 1)

    images = 0

    low = 80
    high = 255
    ret, thresh = cv.threshold(src_gray, low, high, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    new_figure = None
    for i, cnt in enumerate(contours):
        x, y, w, h = cv.boundingRect(cnt)

        # on attend un contour carré pour les figures
        ratio = AspectRatio(cnt)

        if ratio > 1.02: continue
        if ratio < 0.98: continue

        area = int(Area(cnt))
        if area <= 5000: continue

        # print(i, "ratio=", ratio, "area=", area, "w, h=", w, h, "x, y=", x, y)

        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.intp(box)
        # cv.drawContours(src, [box], 0, G, 2)

        xc = int(x + w/2.)
        yc = int(y + h/2.)

        extract = np.zeros((h*2, w*2, 3), np.float64)
        extract = src[yc - h: yc + h, xc - w: xc + w, :]

        try:
            # cv.imshow("extract", extract)
            resized = cv.resize(extract, (80, 80), interpolation=cv.INTER_AREA)
            cv.imshow("resized", resized)

            a = np.zeros_like(pattern)
            for r in range(a.shape[1]):
                for c in range(a.shape[2]):
                    for i in range(3):
                        a[0, r, c, 0] += resized[r, c, i]
            a = a / 3
            a = a / data_max

            # print("shapes=", extract.shape, a.shape, a.dtype)

            result = model(a)
            r = np.zeros(8)
            for k in range(8):
                r[k] = result[0, k]
            a_test = np.argmax(r)

            print("prédiction=", forms[a_test])

        except:
            pass

    return 0



HERE = os.path.normpath(os.path.dirname(__file__)).replace("\\", "/")
TOP = os.path.dirname(HERE)
DATA = TOP
LOG = DATA + "/log/"

save_dir = DATA + "/run/models/best_model.h5"

print("DATA=", DATA, "TOP=", TOP, "save_dir=", save_dir)

forms = ["Rond", "Square", "Triangle", "Star5", "Star4", "Eclair", "Coeur", "Lune"]

model = keras.models.load_model(save_dir)
pattern = np.load(DATA + "/dataset/pattern.npy", allow_pickle=True)

with open(DATA + "/run/minmax.txt", "r") as f:
    line = f.readline()
    m = re.match("min=(\d+.\d+) max=(\d.\d+)", line)
    # print(line, m[1], m[2])
    try:
        data_min = float(m[1])
        data_max = float(m[2])
    except:
        print("pas de configuration minmax")
        exit()

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame

    # print("frame.shape", frame.shape)

    form_find_figures(frame, model, forms, data_min, data_max, pattern)
    cv.imshow('frame', frame)

    """
    a = np.zeros_like(pattern)
    for r in range(a.shape[1]):
        for c in range(a.shape[2]):
            a[0, r, c, 0] = extract[0, r, c, 0]

    print("shapes=", extract.shape, a.shape, a.dtype)

    result = model(a)
    r = np.zeros(8)
    for k in range(8):
        r[k] = result[0, k]
    a_test = np.argmax(r)

    print("prédiction=", r, forms[a_test])
    """

    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
