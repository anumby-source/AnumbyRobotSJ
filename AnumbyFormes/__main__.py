import argparse

import cv2 as cv
from random import *
import numpy as np

import re
import os
from os import listdir
from os.path import isfile, join

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import pwk
import datetime
import matplotlib.pyplot as plt

HERE = os.path.normpath(os.path.dirname(__file__)).replace("\\", "/")
DATA = HERE
LOG = DATA + "/log/"

print("DATA=", DATA)

R = (0, 0, 255)
G = (0, 255, 0)
B = (255, 0, 0)
M = (255, 0, 255)
C = (255, 255, 0)
Y = (0, 255, 255)
Black = (0, 0, 0)
White = (255, 255, 255)


"""
Insertion d'une entrée dans un DataFrame pandas
"""
def insert(df, row):
    insert_loc = df.index.max()

    if pd.isna(insert_loc):
        df.loc[0] = row
    else:
        df.loc[insert_loc + 1] = row


def rad2deg(alpha):
    return 180 * alpha / np.pi


def deg2rad(alpha):
    return np.pi * alpha / 180


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




class Figures(object):
    def __init__(self):
        self.image = None

        self.draw_forms = [self.drawRond, self.drawSquare, self.drawTriangle, self.drawStar5,
                           self.drawStar4, self.drawEclair, self.drawCoeur, self.drawLune,
                           self.drawHexagone, self.drawPentagone, self.drawLogo, self.drawD]
        self.forms = ["Rond", "Square", "Triangle", "Star5", "Star4", "Eclair", "Coeur", "Lune",
                      "Hexagone", "Pentagone", "Logo", "D"]
        self.line_width = 1

    def set_zoom(self, cell, space, margin, line):
        # Au centre, nous avons "cell" qui contient la forme elle-même
        #
        #  [ margin | space | cell | space | margin ]
        #
        self.cell = cell

        self.space = space
        self.margin = margin
        self.line_width = line

        self.cell2 = int(self.cell / 2)
        self.cell4 = int(self.cell2 / 2)

        self.figure_size = self.margin + self.space + self.cell + self.space + self.margin

        print("set_zoom> margin=", self.margin, "space=", self.space, "cell=", self.cell, "size=", self.figure_size, "line_width=", self.line_width)


    def set_image(self, form_number):
        w = form_number * self.figure_size
        h = self.figure_size
        image1 = np.ones((h, w, 3)) * 255

        self.image = np.ones((h, w, 3)) * 255

    def drawFrame(self, x, y):

        """
        # Total
        corner1 = (x, y)
        corner2 = (x + self.figure_size - 1 , y + self.figure_size - 1 )
        self.canvas.create_rectangle(corner1,
                                     corner2,
                                     fill="white",
                                     width = self.line_width + 1)
        """

        A = self.margin
        corner1 = (x + A, y + A)
        corner2 = (x + self.figure_size - A - 1 , y + self.figure_size - A - 1 )
        cv.rectangle(self.image, corner1, corner2, Black, self.line_width)

        B = self.margin + self.space

        return x + B, y + B


    def drawPolygone(self, pointes, x, y):
        x, y = self.drawFrame(x, y)
        radius = self.cell2
        pts = []
        for dalpha in range(pointes + 1):
            alpha = dalpha * 2*np.pi/pointes
            r = radius

            px = int(x + self.cell2 + r * np.cos(alpha - np.pi/2))
            py = int(y + self.cell2 + r * np.sin(alpha - np.pi/2))

            # console.log("small=", small, "alpha=", alpha, "px = ", px, "py = ", py)

            pts.append((px, py))

        for i in range(pointes + 1):
            cv.line(self.image, pts[i], pts[(i + 1) % (pointes + 1)], Black, self.line_width)


    def drawStar(self, pointes, x, y):
        x, y = self.drawFrame(x, y)

        radius = self.cell * 0.15

        pts = []
        small = False

        n = 2*pointes
        for dalpha in range(n + 1):
            alpha = dalpha * np.pi/pointes
            r = 0

            if small:
                r = radius
                small = False
            else:
                r = self.cell2
                small = True

            px = int(x + self.cell2 + r * np.cos(alpha - np.pi/2))
            py = int(y + self.cell2 + r * np.sin(alpha - np.pi/2))

            # console.log("small=", small, "alpha=", alpha, "px = ", px, "py = ", py)

            pts.append((px, py))

        for i in range(n + 1):
            cv.line(self.image, pts[i], pts[(i + 1) % (n + 1)], Black, self.line_width)


    def drawRond(self, x, y):
        # print("rond")
        x, y = self.drawFrame(x, y)

        cv.circle(self.image, (int(x) + self.cell2, int(y) + self.cell2), self.cell2, Black, self.line_width)


    def drawSquare(self, x, y):
        # print("square")
        self.drawPolygone(4, x, y)


    def drawTriangle(self, x, y):
        # print("triangle")
        self.drawPolygone(3, x, y)


    def drawStar5(self, x, y):
        # print("star5")
        self.drawStar(5, x, y)


    def drawStar4(self, x, y):
        # print("star4")
        self.drawStar(4, x, y)


    def drawHexagone(self, x, y):
        # print("hexagone")
        self.drawPolygone(6, x, y)


    def drawPentagone(self, x, y):
        # print("pentagone")
        self.drawPolygone(5, x, y)


    def drawLogo(self, x, y):
        # print("logo")
        x, y = self.drawFrame(x, y)
        pointes = 4
        radius = self.cell2
        pts = []

        cx = x + self.cell2
        cy = y + self.cell2

        dalpha = np.pi/10
        dr = self.cell * 0.18

        n = pointes
        for nalpha in range(n):
            alpha = nalpha * 2*np.pi/pointes + np.pi/4
            r = radius

            px = int(cx + r * np.cos(alpha - dalpha))
            py = int(cy + r * np.sin(alpha - dalpha))
            pts.append((px, py))

            px = int(cx + (r - dr) * np.cos(alpha))
            py = int(cy + (r - dr) * np.sin(alpha))
            pts.append((px, py))

            px = int(cx + r * np.cos(alpha + dalpha))
            py = int(cy + r * np.sin(alpha + dalpha))
            pts.append((px, py))

        for i in range(n):
            cv.line(self.image, pts[i], pts[(i + 1) % n], Black, self.line_width)



    def drawCoeur(self, x, y):
        # print("coeur")

        x, y = self.drawFrame(x, y)

        radius = self.cell4

        c1x = x + self.cell4
        c1y = y + self.cell4

        alpha1 = np.pi * 0.77
        start1 = rad2deg(alpha1)
        end1 = rad2deg(np.pi * 2)
        p12x = int(c1x + radius * np.cos(alpha1))
        p12y = int(c1y + radius * np.sin(alpha1))

        cv.ellipse(self.image, (c1x, c1y), (self.cell4, self.cell4), 0, start1, end1, Black, self.line_width)

        c2x = x + self.cell2 + self.cell4
        c2y = c1y

        alpha2 = np.pi * 2.23
        start2 = rad2deg(np.pi)
        end2 = rad2deg(alpha2)

        # print(start1, start2, extent1)

        p21x = int(c2x + radius * np.cos(alpha2))
        p21y = int(c2y + radius * np.sin(alpha2))

        cv.ellipse(self.image, (c2x, c2y), (self.cell4, self.cell4), 0, start2, end2, Black, self.line_width)

        cv.line(self.image, (p12x, p12y), (x + self.cell2, y + self.cell), Black, self.line_width)
        cv.line(self.image, (x + self.cell2, y + self.cell), (p21x, p21y), Black, self.line_width)


    def drawEclair(self, x, y):
        # print("éclair")

        x, y = self.drawFrame(x, y)

        #self.canvas.create_line(x, y + self.cell*0.2, x + self.cell, y + self.cell*0.8, fill="green")
        #self.canvas.create_line(x, y + self.cell*0.55, x + self.cell, y, fill="green")

        pts = []
        pts.append((int(x), int(y + self.cell*0.2)))                 # 1
        pts.append((int(x + self.cell*0.305), int(y + self.cell*0.38)))   # 2
        pts.append((int(x + self.cell*0.22), int(y + self.cell*0.43)))    # 3
        pts.append((int(x + self.cell*0.53), int(y + self.cell*0.63)))    # 4
        pts.append((int(x + self.cell*0.44), int(y + self.cell*0.69)))    # 5

        pts.append((int(x + self.cell), int(y + self.cell)))              # 6

        pts.append((int(x + self.cell*0.595), int(y + self.cell*0.60)))    # 7
        pts.append((int(x + self.cell*0.67), int(y + self.cell*0.55)))   # 8
        pts.append((int(x + self.cell*0.43), int(y + self.cell*0.31)))     # 9
        pts.append((int(x + self.cell*0.515), int(y + self.cell*0.265)))    # 10
        pts.append((int(x + self.cell*0.35), int(y + self.cell*0.01)))     # 11
        pts.append((int(x), int(y + self.cell*0.2)))                  # 1

        n = len(pts)
        for i in range(n):
            cv.line(self.image, pts[i], pts[(i + 1) % n], Black, self.line_width)


    def drawLune(self, x, y):
        # print("lune")

        x, y = self.drawFrame(x, y)

        first = True

        def intersection(x1, y1, r1, x0, r0):
            """
            y0 = y1
            C1 => r0^2 = (x0 - x)^2 + (y0 - y)^2
            C2 => r1^2 = (x1 - x)^2 + (y0 - y)^2

            r0^2 = x0^2 + x^2 - 2*x0*x + y0^2 + y^2 - 2*y0*y
            r1^2 = x1^2 + x^2 - 2*x1*x + y0^2 + y^2 - 2*y0*y

            r1^2 - r0^2 = x1^2 + x^2 - 2*x1*x + y0^2 + y^2 - 2*y0*y - (x0^2 + x^2 - 2*x0*x + y0^2 + y^2 - 2*y0*y)
            r1^2 - r0^2 = x1^2 + x^2 - 2*x1*x + y0^2 + y^2 - 2*y0*y - x0^2 - x^2 + 2*x0*x - y0^2 - y^2 + 2*y0*y
            r1^2 - r0^2 = (x1^2 - x0^2) - 2*x1*x + 2*x0*x + x^2 + (y0^2 - y0^2) - 2*y0*y + 2*y0*y + y^2 - x^2 - y^2
            r1^2 - r0^2 = (x1^2 - x0^2) - 2*x1*x + 2*x0*x + x^2 + y^2 - x^2 - y^2
            r1^2 - r0^2 - (x1^2 - x0^2) - 2*x*(x1 - x0) = 0

            x = (r1^2 - r0^2 - x1^2 + x0^2) / 2*(x1 - x0)

            C1 => r1^2 = (x1 - x)^2 + (y1 - y)^2
            0 = (x1 - x)^2 + (y1 - y)^2 - r1^2
            0 = x1^2 + x^2 + 2*x1*x + y1^2 + y^2 - 2*y1*y - r1^2
            0 = y^2 - 2*y1*y + (x1^2 + x^2 - 2*x1*x + y1^2 - r1^2)
            """
            y0 = y1
            x = (r1*r1 - r0*r0 - x1*x1 + x0*x0) / (2*(x0 - x1))
            A = 1
            B = -2*y1
            C = x1*x1 + x*x - 2*x1*x + y1*y1 - r1*r1
            D = B*B - 4*A*C

            # print("r1, r0, x1, x0=", r1, r0, x1, x0, "r1^2, r0^2, x1^2, x0^2=", r1*r1, r0*r0, x1*x1, x0*x0, "n=", (r1*r1 - r0*r0 - x1*x1 + x0*x0), "d=", 2*(x0 - x1), "x=", x)
            # print("intersection= x1, y1, r1, x0, r0, x = ", x1, y1, r1, x0, r0, x, " A=", A, "B=", B, "C=", C, "D=", D)

            y1, y2 = 0, 0
            try:
                f = lambda e: (-B + e * np.sqrt(D))/2*A
                y1 = f(1)
                y2 = f(-1)

                # print("intersection= A", x, y1, y2)
            except:
                pass

            return x, y1, y2


        radius1 = int(self.cell2)
        c1x = int(x + radius1)
        c1y = int(y + radius1)

        radius2 = int(self.cell2 * 0.8)
        c2x = int(c1x + self.cell2*0.6)
        c2y = int(c1y)

        x, y1, y2 = intersection(c1x, c1y, radius1, c2x, radius2)

        # coord1 = c1x - radius1, c1y - radius1, c1x + radius1, c1y + radius1
        alpha1 = np.arccos((x - c1x) / radius1)
        start1 = rad2deg(alpha1)
        end1 = rad2deg(2*np.pi - alpha1)

        cv.ellipse(self.image, (c1x, c1y), (radius1, radius1), 0, start1, end1, Black, self.line_width)

        # coord2 = c2x - radius2, c2y - radius2, c2x + radius2, c2y + radius2
        alpha2 = np.arccos((x - c2x) / radius2)
        start2 = rad2deg(alpha2)
        end2 = rad2deg(2*np.pi - alpha2)

        cv.ellipse(self.image, (c2x, c2y), (radius2, radius2), 0, start2, end2, Black, self.line_width)


    def drawD(self, x, y):
        # print("d")

        x, y = self.drawFrame(x, y)

        cv.line(self.image, (int(x + self.cell2), int(y)), (int(x), int(y)), Black, self.line_width)
        cv.line(self.image, (int(x), int(y)), (int(x), int(y + self.cell)), Black, self.line_width)
        cv.line(self.image, (int(x), int(y + self.cell)), (int(x + self.cell2), int(y + self.cell)), Black, self.line_width)

        coord = x, y, x + self.cell, y + self.cell

        cv.ellipse(self.image, (x + self.cell2, y + self.cell2), (self.cell2, self.cell2), 0, 3*90, 5*90, Black, self.line_width)



    def drawAll(self, y, form_number=None):
        for x, drawer in enumerate(self.draw_forms):
            if form_number is not None and x >= form_number: break
            # print(self.forms[x], x * self.cell + self.margin, y)
            drawer(x * (self.figure_size), y)


    def prepare_source_images(self, form_number=None, rebuild_forme=None):
        if form_number is None: form_number = len(self.forms)

        # y = 5
        # self.prepareAll(y, form_number)

        images = []

        y = 1
        for form, drawer in enumerate(self.draw_forms):
            if form >= form_number: break
            if rebuild_forme is not None and rebuild_forme != form:
                continue

            print("prepare_source_images> form=", form)

            self.set_image(1)

            drawer(1, y)


            os.makedirs(DATA + "./dataset/{}".format(self.forms[form]), mode=0o750, exist_ok=True)
            filename = DATA + "./dataset/{}/RawImages{}.jpg".format(self.forms[form], self.forms[form])

            # cv.imshow(filename, self.image)
            # cv.waitKey(0)
            status = cv.imwrite(filename, self.image)
            # cv.imwrite("BlackImages{}.jpg".format(self.forms[form]), black)

            shape = self.image.shape
            print("image shape = ", shape)
            data = np.zeros([shape[0], shape[1]])
            for i in range(3):
                data[:, :] += self.image[:, :, i]

            data /= 3.0

            images.append(data)

        return images

    def load_source_images(self, form_number=None, rebuild_forme=None):
        if form_number is None: form_number = len(self.forms)

        images = []

        for nform, form in enumerate(self.forms):
            if nform >= form_number: break

            if rebuild_forme is not None and rebuild_forme != nform:
                continue

            print("load_source_images> nform=", nform)

            filename = DATA + "/dataset/{}/RawImages{}.jpg".format(self.forms[nform], self.forms[nform])
            img = cv.imread(filename, cv.IMREAD_COLOR)
            data = np.zeros([img.shape[0], img.shape[1]])
            for i in range(3):
                data[:, :] += img[:, :, i]

            data /= 3.0

            # print("load_source_images> ", data.shape)

            images.append(data)

        return images


def change_perpective(image):
    def f(x, y, width, height):
        sigma = 1.3
        xx = x + gauss(mu=float(0), sigma=sigma)
        if xx < 0: xx = 0
        if xx >= width: xx = width - 1
        yy = y + gauss(mu=float(0), sigma=sigma)
        if yy < 0: yy = 0
        if yy >= height: yy = height - 1
        # print(x, y, xx, yy)
        return xx, yy

    def setmin(v, vmin):
        if vmin is None or v < vmin: return v
        return vmin

    def setmax(v, vmax):
        if vmax is None or v > vmax: return v
        return vmax

    # on va étendre l'image pour accepter la déformation causée par la transformation
    extend = 1
    full_extend = extend*2 + 1

    # on installe l'image à transformer au centre de l'image étendue
    width, height = image.shape
    img = np.ones((full_extend * height, full_extend * width, 3)) * 255.
    for c in range(3):
        img[extend*width:(extend+1)*width, extend*height:(extend+1)*height, c] = image[:, :]


    # pour construire la matrice de transformation, on dessine 4 points qui sont les 4 coins d'un carré autour de la figure
    offset = 10
    pts1 = np.array([[-offset, -offset],
                     [width+offset, -offset],
                     [-offset, height+offset],
                     [width+offset, height+offset]], np.float32)

    R = (0, 0, 255)
    G = (0, 255, 0)
    B = (255, 0, 0)
    M = (255, 0, 255)
    colors = [R, G, B, M]
    for x in range(0, 4):
        cv.circle(img, (extend * width + int(pts1[x][0]), extend * height + int(pts1[x][1])), 3, colors[x], cv.FILLED)

    # cv.imshow("original image", img)

    # pour définir la transformation, on déplace aléatoirement ces 4 points autour de leur position initiale
    pts2 = np.zeros_like(pts1)

    for x in range(0, 4):
        pts2[x][0], pts2[x][1] = f(pts1[x][0], pts1[x][1], width=full_extend*width, height=full_extend*height)

    # application de la transformation de perspective. On garde la taille de l'image transformée identique
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    width = full_extend*width
    height = full_extend*height
    img2 = cv.warpPerspective(img, matrix, (width, height))

    # lors de la transformation, l'image fait aparaître des zones noires correspondant aux limites de l'image d'origine
    # Pour éliminer ces zones noires, on repère les 4 points de référence (qui sont colorés) dans l'image transformée
    # comme on sait que le carré de référence entoure exactement la figure , on peut découper la zone de référence
    # et la déplacer dans une nouvelle image complètement blanche
    xmin = None
    xmax = None
    ymin = None
    ymax = None
    for x in range(width):
        for y in range(height):
            r = img2[y, x, 0]
            g = img2[y, x, 1]
            b = img2[y, x, 2]
            t = False
            if (r == 0 and g == 0 and b == 255):
                t = True
                # print("R", x, y, r, g, b)
            elif (r == 0 and g == 255 and b == 0):
                t = True
                # print("G", x, y, r, g, b)
            elif (r == 255 and g == 0 and b == 0):
                t = True
                # print("B", x, y, r, g, b)
            elif (r == 255 and g == 0 and b == 255):
                t = True
                # print("M", x, y, r, g, b)
            if t:
                # on efface les points de référence
                img2[y, x, 0] = 255
                img2[y, x, 1] = 255
                img2[y, x, 2] = 255
                # on met à jour les limites de la zone de référence
                xmin = setmin(x, xmin)
                xmax = setmax(x, xmax)
                ymin = setmin(y, ymin)
                ymax = setmax(y, ymax)

    # print(height, width, img2.shape, ymin, ymax, xmin, xmax)
    # nouvelle image
    img_finale = np.ones((height, width, 3)) * 255.
    # insertion de la zone de référence transformée qui contient la figure
    img_finale[ymin:ymax, xmin:xmax, :] = img2[ymin:ymax, xmin:xmax, :]

    # img_finale = np.zeros((60, 60, 3))
    # img_finale[:, :, :] = img2[20:80, 20:80, :]

    return img_finale


def transformation_model(height, width):
    """
    Crop(height, width)
    Flip
    Translation(height_factor, width_factor)
    Rotation
    Zoom(height_factor)
    Height(factor)
    Width(factor)
    Contrast(factor)
    """
    # scale = 1.5

    # general range of transformation
    transformation_range = 0.08
    return tf.keras.Sequential([
        keras.layers.RandomZoom(-0.5, fill_mode="nearest"),
        keras.layers.RandomRotation(transformation_range),
        # keras.layers.RandomTranslation(transformation_range, transformation_range, fill_mode="nearest"),
        # keras.layers.RandomCrop(int(height*scale), int(width*scale)),
        # keras.layers.RandomContrast(0.5),
    ])

def add_noise(image):
    height, width = image.shape[:2]

    """
    n taches de bruit [0, n_max]
    pour chaque tache, on ajoute m pixels de bruit [0..m_max]
    la largeur de la tache vaut sigma [0 .. sigma_max]
    """
    n_max = 5
    m_max = 10
    sigma_max = 30

    n = int(randrange(0, n_max))
    for i in range(n):
        xs = []
        ys = []
        x0 = int(randrange(0, width))
        y0 = int(randrange(0, height))
        sigma = int(randrange(0, sigma_max))
        m = int(randrange(0, m_max))
        for j in range(m):
            x = int(gauss(mu=float(x0), sigma=sigma))
            y = int(gauss(mu=float(y0), sigma=sigma))

            # b = int(randrange(0, 255))
            # g = int(randrange(0, 255))
            # r = int(randrange(0, 255))

            if x >= 0 and x < width and y >= 0 and y < height:
                # print("noise", "tache=", i, "pixel=", j, "wh=", width, height, "xy=", x, y, "bgr=", r, g, r)
                image[y, x, 0] = 0
                image[y, x, 1] = 0
                image[y, x, 2] = 0

    """
    cv.imshow("noised image", image)
    k = cv.waitKey(0)
    if k == 27 or k == 113:
        exit()
    """

    return image


def transformation(image):
    # print("transformation> ", type(image), image.shape)

    height, width = image.shape[:2]

    mode = "tf"

    transformed_image = transformation_model(height, width)

    """
    On voudrait ajouter quelques pixels parasites sur les images produites
    """

    data = np.zeros([height, width, 3], np.float32)
    data[:, :, 0] = image[:, :, 0]
    data[:, :, 1] = image[:, :, 1]
    data[:, :, 2] = image[:, :, 2]

    X = np.zeros([height, width, 3], np.float32)
    # data = add_noise(data)

    img = transformed_image(data).numpy()
    # print(img.shape, type(img))

    img_finale = np.zeros([img.shape[0], img.shape[1]])
    for j in range(3):
        img_finale[:, :] += img[:, :, j]

    img_finale /= 3

    return img_finale


def build_data(figures, data_size, images):
    # on sauvegarde les data non normlisées

    """
    def f(x):
        sigma = 5
        x = gauss(mu=float(x), sigma=sigma)
        if x < 0: x = 0.
        if x > 255: x = 255.
        return x
    """

    captures = dict()

    captured_images = []

    path = DATA + "/captures/"
    for f in [f for f in listdir(path) if isfile(join(path, f))]:
        m = re.match("capture_([^_]+)_(\d+).jpg", f)
        if m is not None:
            # print(m[1], m[2])
            name = m[1]
            captures[name] = int(m[2])
            img = cv.imread(path + f)
            if name == "fond":
                cl = 8
            else:
                cl = figures.forms.index(name)
            captured_images.append((img, cl))

    print(captures)
    print([(i[0].shape, i[1]) for i in captured_images])

    # vf = np.vectorize(f)

    # shape = images[0].shape
    shape = captured_images[0][0].shape

    print("build_data> ", shape)
    image_number = len(captured_images)

    # sélectionne la proportion de données dans le training set et le test set
    frac = 0.85
    first = True
    data_id = 0

    for i in range(data_size):
        for captured_image in captured_images:
            raw_img = captured_image[0]
            image_id = captured_image[1]

            if data_id % 1000 == 0: print("generate data> data_id = ", data_id)
            # print(raw_img.shape)

            # random transformations
            data = transformation(raw_img)

            # visualisation de l'image finale
            show = False
            if show:
                print("forme", image_id, "data_id=", data_id)
                cv.imshow("input image", raw_img)
                cv.imshow("output image", data)
                k = cv.waitKey(0)
                if k == 27 or k == 113:
                    exit()

            if first:
                shape = data.shape
                x_data = np.zeros([data_size * image_number, shape[0], shape[1], 1])
                y_data = np.zeros([data_size * image_number])
                first = False

            x_data[data_id, :, :, 0] = data[:, :]
            y_data[data_id] = image_id

            data_id += 1

            # print("build_data> p={} x_data={}".format(p, x_data[p, :, :, 0]))
            # print("build_data> p={} y_data={}".format(p, y_data[p]))


    # x_data = x_data.reshape(-1, shape[0], shape[1], 1)

    data_number = data_id

    p = np.random.permutation(len(x_data))

    x_data = x_data[p]
    y_data = y_data[p]

    index = int(frac*data_size*image_number)

    print("build_data> x_data.shape=", x_data.shape, "index=", index, "generated data=", data_number)

    # print("----------------------------------------------------------------------------------------------")
    # print("build_data> x_data={}".format(x_data[:, :, :, :]))
    # print("build_data> y_data={}".format(y_data[:]))
    # print("----------------------------------------------------------------------------------------------")

    x_train = x_data[:index, :, :, :]
    y_train = y_data[:index]
    x_test = x_data[index:, :, :, :]
    y_test = y_data[index:]
    print("build_data> x_train : ", x_train.shape)
    print("build_data> y_train : ", y_train.shape)
    print("build_data> x_test : ", x_test.shape)
    print("build_data> y_test : ", y_test.shape)

    os.makedirs(DATA + "/data/", mode=0o750, exist_ok=True)
    np.save(DATA + "/data/x_train.npy", x_train, allow_pickle=True)
    np.save(DATA + "/data/y_train.npy", y_train, allow_pickle=True)
    np.save(DATA + "/data/x_test.npy", x_test, allow_pickle=True)
    np.save(DATA + "/data/y_test.npy", y_test, allow_pickle=True)

    np.save(DATA + "/dataset/pattern.npy", x_train[0:1], allow_pickle=True)

    return x_train, y_train, x_test, y_test

"""
relecture de mean/std par rapport à un modèle déjà entraîné
"""
def get_mean_std():
    with open("./run/models/mean_std.txt", "r") as f:
        lines = f.readlines()

    m = re.match("(\d+.\d+)", lines[0])
    mean = float(m.group(1))
    m = re.match("(\d+.\d+)", lines[1])
    std = float(m.group(1))

    return mean, std

"""
relecture de mean/std par rapport à un modèle déjà entraîné
"""
def get_xmax():
    with open("./run/models/xmax.txt", "r") as f:
        lines = f.readlines()

    m = re.match("(\d+.\d+)", lines[0])
    xmax = float(m.group(1))

    return xmax

def load_data():
    x_train = np.load(DATA + "/data/x_train.npy", allow_pickle=True)
    y_train = np.load(DATA + "/data/y_train.npy", allow_pickle=True)
    x_test = np.load(DATA + "/data/x_test.npy", allow_pickle=True)
    y_test = np.load(DATA + "/data/y_test.npy", allow_pickle=True)

    return x_train, y_train, x_test, y_test

def model_chat_gpt():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(80, 80, 1)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(9, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Print the model summary

def build_model_v1(shape, form_number):
    model = keras.models.Sequential()

    model.add(keras.layers.Input((shape[1], shape[2], 1)))

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    # model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    # model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(form_number + 1, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def build_model_v2(shape):
    print("build_model_v2> shape=", shape)

    shape = (shape[1], shape[2])

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape, name="InputLayer"))
    model.add(keras.layers.Dense(50, activation="relu", name="Dense_n1"))
    model.add(keras.layers.Dense(50, activation="relu", name="Dense_n2"))
    model.add(keras.layers.Dense(50, activation="relu", name="Dense_n3"))
    model.add(keras.layers.Dense(1, name="Output"))

    model.compile(optimizer="rmsprop",
                  loss="mse",
                  metrics=["mae", "mse"])

    return model


def do_run(figures, form_number, data_size, rebuild_data, rebuild_model, rebuild_forme=None):
    os.makedirs("./data", mode=0o750, exist_ok=True)

    if rebuild_data:
        rebuild_model = True

        images = figures.load_source_images(form_number=form_number, rebuild_forme=rebuild_forme)

        # print("run> ", type(images), images)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Generating data from images")
        x_train, y_train, x_test, y_test = build_data(figures, data_size, images)
    else:
        x_train, y_train, x_test, y_test = load_data()

    print("run> x_train : ", x_train.shape)
    print("run> y_train : ", y_train.shape)
    print("run> x_test : ", x_test.shape)
    print("run> y_test : ", y_test.shape)
    print("run> x_train : ", y_train[10:20, ])

    """
    def plot_images(x, y=None, indices='all', columns=12, x_size=1, y_size=1,
                    colorbar=False, y_pred=None, cm='binary', norm=None, y_padding=0.35, spines_alpha=1,
                    fontsize=20, interpolation='lanczos', save_as='auto'):
    """

    pwk.plot_images(x=x_train, y=y_train, indices=range(10*12), fontsize=8, save_as=LOG + "Data.jpg")

    print('Before normalization : Min={}, max={}'.format(x_train.min(), x_train.max()))

    xmin = x_train.min()
    xmax = x_train.max()

    x_train = x_train / xmax
    x_test = x_test / xmax

    print('After normalization  : Min={}, max={}'.format(x_train.min(), x_train.max()))

    os.makedirs(DATA + "/run/models", mode=0o750, exist_ok=True)
    save_dir = DATA + "/run/models/best_model.h5"

    with open(DATA + "/run/minmax.txt", "w+") as f:
        f.write("min={} max={}".format(xmin, xmax))

    if rebuild_model:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Build model")

        model = build_model_v1(x_train.shape, form_number)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Start training")
        batch_size  = 128
        epochs      = 32

        savemodel_callback = keras.callbacks.ModelCheckpoint(filepath=save_dir, verbose=0, save_best_only=True)

        from tensorflow.keras.utils import to_categorical

        # Convert the target data to one-hot encoded format
        y_train = to_categorical(y_train, num_classes=9)
        y_test = to_categorical(y_test, num_classes=9)

        fit_verbosity = 1
        history = model.fit(x_train, y_train,
                            batch_size      = batch_size,
                            epochs          = epochs,
                            verbose         = fit_verbosity,
                            validation_data = (x_test, y_test),
                            callbacks = [savemodel_callback])

        pwk.plot_history(history, figsize=(form_number, 4), save_where=LOG)

    else:
        model = keras.models.load_model(save_dir)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Evaluate model")
    score = model.evaluate(x_test, y_test, verbose=0)

    print(f'Test loss     : {score[0]:4.4f}')
    print(f'Test accuracy : {score[1]:4.4f}')

    return model, x_train, y_train, x_test, y_test

def find_figures(src):
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    shape = src.shape
    max_area = (shape[0] - 1) * (shape[1] - 1)
    # print(shape, max_area)
    # src_gray = cv.blur(src_gray, (3, 3))
    ret, thresh = cv.threshold(src_gray, 200, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        x, y, w, h = cv.boundingRect(cnt)

        # on attend un contour carré pour les figures
        ratio = AspectRatio(cnt)
        if ratio > 1.02: continue
        if ratio < 0.98: continue

        area = int(Area(cnt))
        # La surface est évidemment dépendante du facteur de grandissement
        # sans grandissement on obtient une surface de l'ordre de 2500
        # il faudra construire un étalonnage en fonction du grandissement image réelle

        if area == max_area: continue
        ratio = (np.sqrt(area) - 1) / shape[0]
        # print("Area=", area, ratio)

        return ratio

        break
    return 0


def handle_arguments():
    argParser = argparse.ArgumentParser()

    # argParser.add_argument("-formes", type=int, default=8, help="nombre de formes pour l'entrainement")

    # argParser.add_argument("-cell", type=int, default=100, help="taille de la figure elle_même")
    # argParser.add_argument("-space", type=int, default=30, help="espace autour de la figure")
    # argParser.add_argument("-margin", type=int, default=40, help="marge entre figures")
    # argParser.add_argument("-line", type=int, default=2, help="line width")

    # argParser.add_argument("-data", "--data_size", type=int, default=100, help="nombre de données for traning")

    group1 = argParser.add_mutually_exclusive_group()
    group1.add_argument("-figures", action="store_true", help="build figures")
    group1.add_argument("-run", action="store_true", help="run")

    argParser.add_argument("-f", "--figure", type=int, default=None, help="figure to build")

    # par défaut: c'est l'action "load" qui est vraie
    argParser.add_argument("-build_data", action="store_true", help="build data")
    argParser.add_argument("-build_model", action="store_true", help="build model")

    args = argParser.parse_args()

    return args.figures, args.figure, \
           args.build_data, args.build_model, \
           args.run


# ===================================================================================================================


def main():
    figures = Figures()

    with open(DATA + "/dataset/figure.conf", "r") as f:
        line = f.readline()
        #             cell=60 pace=20 margin=30 line=10 data_size=5000 form_number=8
        m = re.match("cell=(\d+) space=(\d+) margin=(\d+) line=(\d+) data_size=(\d+) form_number=(\d+)", line)
        print(line, m[1], m[2], m[3], m[4])
        try:
            cell = int(m[1])
            space = int(m[2])
            margin = int(m[3])
            line = int(m[4])

            data_size = int(m[5])
            form_number = int(m[6])
            figures.set_zoom(cell, space, margin, line)
        except:
            print("pas de configuration pour les figures")
            exit()

    build_figures, figure, build_data, build_model, run = handle_arguments()

    # ============ generlal parameters=================
    os.makedirs(DATA + "./dataset", mode=0o750, exist_ok=True)
    os.makedirs(DATA + "./data", mode=0o750, exist_ok=True)

    if build_figures:
        if figure is None:
            print("Formes> Rebuild all figures")
            figures.prepare_source_images(form_number=form_number)
        else:
            print("Formes> Rebuild figure # {}".format(figure))
            figures.prepare_source_images(form_number=form_number, rebuild_forme=figure)

        return
    elif run:

        print("Formes> Run with build_data={} build_model={}".format(build_data, build_model))
    else:
        print("Formes> no action")
        return

    model, x_train, y_train, x_test, y_test = do_run(figures,
                                                     form_number=form_number,
                                                     data_size=data_size,
                                                     rebuild_data=build_data,
                                                     rebuild_model=build_model)

    # pwk.plot_images(x_train, y_train, [27], x_size=5, y_size=5, colorbar=True, save_as=LOG + '01-one-digit')
    # pwk.plot_images(x_train, y_train, range(0,64), columns=8, save_as=LOG + '02-many-digits')

    print("len(x_test)=", len(x_test))
    tests = len(x_test)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Predictions")

    test_global = True
    if test_global:
        fraction = 20
        batch = int(tests/fraction)
        n = 0
        for i in range(fraction):
            n1 = i*batch
            n2 = n1 + batch - 1
            result = model(x_test[n1:n2-1])
            y_pred = np.argmax(result, axis=-1)
            pwk.plot_images(x_test[n1:n2-1], y_test[n1:n2-1], indices=range(8 * 8), columns=8, x_size=1, y_size=1, y_pred=y_pred,
                            save_as=LOG + 'Data.jpg')
            pwk.plot_images(x_test[n1:n2-1], y_pred, indices=range(8 * 8), columns=8, x_size=1, y_size=1, y_pred=y_pred,
                            save_as=LOG + 'Predictions.jpg')
            # errors = [i for i in range(len(x_test)-1) if y_pred[i] != y_test[i]]
            # print("errors", errors)
            # errors=errors[:min(24,len(errors))]
            # pwk.plot_images(x_test, y_test, errors[:15], columns=8, x_size=2, y_size=2, y_pred=y_pred, save_as=LOG + 'Errors')

            pwk.plot_confusion_matrix(y_test[n1:n2-1], y_pred, range(8), normalize=True, save_as=LOG + 'Confusion.jpg')
    else:
        test_number = 1
        errors = 0
        ok = 0
        scores1 = []
        scores2 = []
        scores3 = []
        scores4 = []
        for i in range(tests):
            # n = randint(0, tests)
            # print("n=", n)
            A = x_test[i:i+1, :, :, :][0]

            img = np.zeros((A.shape[0], A.shape[1], 3), np.uint8)

            for c in range(80):
                for r in range(80):
                    e = A[c, r, 0]
                    if e < 0.99:
                        img[c, r, 0] = 0
                        img[c, r, 1] = 0
                        img[c, r, 2] = 0
                    else:
                        img[c, r, 0] = 255
                        img[c, r, 1] = 255
                        img[c, r, 2] = 255



            # print(img.shape)

            ratio = find_figures(img)

            # cv.imshow("A", img)
            # cv.waitKey(0)

            B = int(y_test[i:i + 1][0])

            now = datetime.datetime.now()
            # print("data.shape", x_test[i:i + 1, :, :, :].shape)
            result = model(x_test[i:i + 1, :, :, :])
            t = datetime.datetime.now() - now
            r = np.zeros(8)
            for k in range(8):
                r[k] = result[0, k]

            predmax = np.argmax(r)

            if predmax == B:
                scores1.append(ratio)

            if False:

                if predmax != B:
                    # le max ne correspond pas à la théorie
                    # print("i=", i, "A", A.shape, B, "pred=", predmax, "durée=", t, "score=", r)
                    scores1.append(r[B])         # résultat pour la théorie
                    scores2.append(r[predmax])   # résultat max
                else:
                    print("conforme", B, figures.forms[B])
                    scores3.append(r[predmax])   # résultat conforme à la théorie

                if r[predmax] > 0.99 and predmax != B:
                    scores4.append(r[predmax])   # résultat non conforme à la théorie



        plt.figure(figsize=[4, 4])
        plt.title("ratio")
        plt.plot(range(len(scores1)), scores1)
        plt.show()

        """
        plt.figure(figsize=[4, 4])
        plt.title("non conforme")
        plt.scatter(scores1, scores2)
        plt.figure(figsize=[4, 4])
        plt.title("conforme")
        plt.plot(range(len(scores3)), scores3)
        if len(scores4) > 0:
            plt.figure(figsize=[4, 4])
            plt.title("non conforme seuil")
            plt.plot(range(len(scores4)), scores4)
        plt.show()
        """

    return

if __name__ == "__main__":
    main()
