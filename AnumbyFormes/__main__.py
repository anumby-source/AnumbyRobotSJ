import sys
import argparse

import tkinter as tk
from tkinter import *
from PIL import ImageGrab
from PIL import Image
import cv2 as cv
from random import *
import numpy as np

import re
import os

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import pwk

HERE = os.path.normpath(os.path.dirname(__file__)).replace("\\", "/")
DATA = HERE

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


class Figures(object):
    def __init__(self):
        self.top = tk.Tk()
        self.top.overrideredirect(1)  # FRAMELESS CANVAS WINDOW

        self.draw_forms = [self.drawRond, self.drawSquare, self.drawTriangle, self.drawStar5,
                           self.drawStar4, self.drawEclair, self.drawCoeur, self.drawLune,
                           self.drawHexagone, self.drawPentagone, self.drawLogo, self.drawD]
        self.forms = ["Rond", "Square", "Triangle", "Star5",
                      "Star4", "Eclair", "Coeur", "Lune",
                      "Hexagone", "Pentagone", "Logo", "D"]
        self.line_width = 2

    def run(self):
        self.top.mainloop()

    def set_zoom(self, c):
        self.cell = c
        self.cell2 = self.cell / 2
        self.cell4 = self.cell2 / 2
        self.margin = 10


    def set_canvas(self, form_number):
        self.canvas = tk.Canvas(self.top, bg="white",
                                height=3 * (self.cell + self.margin) + self.margin,
                                width=form_number * (self.cell + 2*self.margin) + self.margin)
        self.canvas.pack()


    def drawGrille(self):
        for row in range(11):
            self.canvas.create_line(self.margin, row*self.cell/10 + self.margin,
                                    self.margin + self.cell, row*self.cell/10 + self.margin,
                                    fill="red")
            for col in range(11):
                self.canvas.create_line(col*self.cell/10 + self.margin, self.margin,
                                        col*self.cell/10 + self.margin, self.margin + self.cell,
                                        fill="red")


    def drawFrame(self, x, y):
        offset = 3
        corner1 = (x + offset, y + offset)
        corner2 = (x + self.cell - 1 - offset + 2*self.margin, y + self.cell - 1 - offset + 2*self.margin)
        self.canvas.create_rectangle(corner1,
                                     corner2,
                                     outline="black",
                                     width = self.line_width + 1)
        return x + self.margin, y + self.margin


    def drawPolygone(self, pointes, x, y):
        x, y = self.drawFrame(x, y)
        radius = self.cell2
        pts = []
        for dalpha in range(pointes + 1):
            alpha = dalpha * 2*np.pi/pointes
            r = radius

            px = x + self.cell2 + r * np.cos(alpha - np.pi/2)
            py = y + self.cell2 + r * np.sin(alpha - np.pi/2)

            # console.log("small=", small, "alpha=", alpha, "px = ", px, "py = ", py)

            pts.append((px, py))

        self.canvas.create_polygon(pts, fill="white", outline="black", width=self.line_width)


    def drawStar(self, pointes, x, y):
        x, y = self.drawFrame(x, y)

        radius = self.cell * 0.15

        pts = []
        small = False
        for dalpha in range(2*pointes + 1):
            alpha = dalpha * np.pi/pointes
            r = 0

            if small:
                r = radius
                small = False
            else:
                r = self.cell2
                small = True

            px = x + self.cell2 + r * np.cos(alpha - np.pi/2)
            py = y + self.cell2 + r * np.sin(alpha - np.pi/2)

            # console.log("small=", small, "alpha=", alpha, "px = ", px, "py = ", py)

            pts.append((px, py))

        self.canvas.create_polygon(pts, fill="white", outline="black", width=self.line_width)


    def drawRond(self, x, y):
        # print("rond")
        x, y = self.drawFrame(x, y)
        self.canvas.create_oval(x, y, x + self.cell, y + self.cell, fill="white", outline="black", width=self.line_width)


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

        for nalpha in range(pointes):
            alpha = nalpha * 2*np.pi/pointes + np.pi/4
            r = radius

            px = cx + r * np.cos(alpha - dalpha)
            py = cy + r * np.sin(alpha - dalpha)
            pts.append((px, py))

            px = cx + (r - dr) * np.cos(alpha)
            py = cy + (r - dr) * np.sin(alpha)
            pts.append((px, py))

            px = cx + r * np.cos(alpha + dalpha)
            py = cy + r * np.sin(alpha + dalpha)
            pts.append((px, py))

        self.canvas.create_polygon(pts, fill="white", outline="black", width=self.line_width)


    def drawCoeur(self, x, y):
        # print("coeur")

        x, y = self.drawFrame(x, y)

        radius = self.cell4

        c1x = x + self.cell4
        c1y = y + self.cell4

        start1 = 0
        extent1 = np.pi * 1.23
        p12x = c1x + radius * np.cos(start1 + extent1)
        p12y = c1y - radius * np.sin(start1 + extent1)

        self.canvas.create_arc(c1x - self.cell4, c1y - self.cell4,
                          c1x + self.cell4, c1y + self.cell4,
                          start=rad2deg(start1),
                          extent=rad2deg(extent1), style=ARC, width=self.line_width)

        c2x = x + self.cell2 + self.cell4
        c2y = c1y

        start2 = np.pi - (start1 + extent1)
        extent2 = extent1

        # print(start1, start2, extent1)

        p21x = c2x + radius * np.cos(start2)
        p21y = c2y - radius * np.sin(start2)

        self.canvas.create_arc(c2x - self.cell4, c2y - self.cell4,
                               c2x + self.cell4, c2y + self.cell4,
                               start=rad2deg(start2),
                               extent=rad2deg(extent2), style=ARC, width=self.line_width)

        self.canvas.create_line(p12x, p12y, x + self.cell2, y + self.cell, fill="black", width=self.line_width)
        self.canvas.create_line(x + self.cell2, y + self.cell, p21x, p21y, fill="black", width=self.line_width)


    def drawEclair(self, x, y):
        # print("éclair")

        x, y = self.drawFrame(x, y)

        #self.canvas.create_line(x, y + self.cell*0.2, x + self.cell, y + self.cell*0.8, fill="green")
        #self.canvas.create_line(x, y + self.cell*0.55, x + self.cell, y, fill="green")

        pts = []
        pts.append((x, y + self.cell*0.2))                 # 1
        pts.append((x + self.cell*0.305, y + self.cell*0.38))   # 2
        pts.append((x + self.cell*0.22, y + self.cell*0.43))    # 3
        pts.append((x + self.cell*0.53, y + self.cell*0.63))    # 4
        pts.append((x + self.cell*0.44, y + self.cell*0.69))    # 5

        pts.append((x + self.cell, y + self.cell))              # 6

        pts.append((x + self.cell*0.595, y + self.cell*0.60))    # 7
        pts.append((x + self.cell*0.67, y + self.cell*0.55))   # 8
        pts.append((x + self.cell*0.43, y + self.cell*0.31))     # 9
        pts.append((x + self.cell*0.515, y + self.cell*0.265))    # 10
        pts.append((x + self.cell*0.35, y + self.cell*0.01))     # 11
        pts.append((x, y + self.cell*0.2))                  # 1
        self.canvas.create_polygon(pts, fill="white", outline="black", width=self.line_width)


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


        radius1 = self.cell2
        c1x = x + radius1
        c1y = y + radius1

        radius2 = self.cell2 * 0.8
        c2x = c1x + self.cell2*0.6
        c2y = c1y

        x, y1, y2 = intersection(c1x, c1y, radius1, c2x, radius2)

        alpha = np.arccos((x - c1x) / radius1)
        espace1 = rad2deg(alpha)

        coord1 = c1x - radius1, c1y - radius1, c1x + radius1, c1y + radius1
        self.canvas.create_arc(coord1, outline="black",
                               start=espace1, extent=(360. - 2 * espace1), style=ARC,
                               width=self.line_width)

        coord2 = c2x - radius2, c2y - radius2, c2x + radius2, c2y + radius2
        alpha = np.arccos((x - c2x) / radius2)
        espace2 = rad2deg(alpha)
        self.canvas.create_arc(coord2, outline="black",
                               start=espace2, extent=(360. - 2 * espace2), style=ARC,
                               width=self.line_width)


    def drawD(self, x, y):
        # print("d")

        x, y = self.drawFrame(x, y)

        self.canvas.create_line(x + self.cell2, y, x, y, fill="black", width=self.line_width)
        self.canvas.create_line(x, y, x, y + self.cell, fill="black", width=self.line_width)
        self.canvas.create_line(x, y + self.cell, x + self.cell2, y + self.cell, fill="black", width=self.line_width)
        coord = x, y, x + self.cell, y + self.cell
        self.canvas.create_arc(coord, outline="black", start=-90, extent=180, style=ARC, width=self.line_width)


    def drawAll(self, y, form_number=None):
        for x, drawer in enumerate(self.draw_forms):
            if form_number is not None and x >= form_number: break
            # print(self.forms[x], x * self.cell + self.margin, y)
            drawer(self.margin + x * (self.cell + 2*self.margin), y)


    def prepare_source_images(self, zoom, form_number=None, rebuild_forme=None):
        if form_number is None: form_number = len(self.forms)

        self.set_zoom(zoom)
        self.set_canvas(form_number)

        self.canvas.delete("all")

        # self.drawGrille()

        y = self.margin
        self.drawAll(y, form_number)

        images = []

        y = self.margin
        for form, drawer in enumerate(self.draw_forms):
            if form >= form_number: break
            if rebuild_forme is not None and rebuild_forme != form:
                continue

            print("prepare_source_images> form=", form)

            self.top.update()
            X = self.margin + form * (self.cell + 2*self.margin)
            Y = y
            img = ImageGrab.grab((X - 1,
                                  Y - 1,
                                  X + self.cell + 2 + 2*self.margin,
                                  Y + self.cell + 2 + 2*self.margin))

            pix = np.array(img.getdata())

            cvimg = pix.reshape((img.size[0], img.size[1], 3)).astype(np.float32)

            """
            black = np.zeros((img.size[0], img.size[1]), np.float32)
            print(cvimg.shape, black.shape)

            for c in range(2):
                black += cvimg[:,:,0]
            black /= 3.0
            # black = black.astype(np.int8)

            bbb = (black < 255.0)*1*black
            """

            # print(cvimg[0:3, 0:3, :])
            os.makedirs("./dataset/{}".format(self.forms[form]), mode=0o750, exist_ok=True)
            filename = "./dataset/{}/RawImages{}.jpg".format(self.forms[form], self.forms[form])
            cv.waitKey()
            cv.imshow(filename, cvimg)
            cv.waitKey()
            cv.imwrite(filename, cvimg)
            # cv.imwrite("BlackImages{}.jpg".format(self.forms[form]), black)

            data = np.zeros([img.size[0], img.size[1]])
            for i in range(3):
                data[:, :] += cvimg[:, :, i]

            data /= 3.0

            images.append(data)

        return images

    def load_source_images(self, zoom, form_number=None, rebuild_forme=None):
        if form_number is None: form_number = len(self.forms)

        images = []

        for nform, form in enumerate(self.forms):
            if nform >= form_number: break

            if rebuild_forme is not None and rebuild_forme != nform:
                continue

            print("load_source_images> nform=", nform)

            filename = DATA + "/dataset/{}/RawImages{}.jpg".format(self.forms[nform], self.forms[nform])
            img = Image.open(filename)
            npimg = np.array(img)
            data = np.zeros([img.size[0], img.size[1]])
            for i in range(3):
                data[:, :] += npimg[:, :, i]

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
    scale = 0.5
    return tf.keras.Sequential([
        keras.layers.RandomZoom((0.0, 0.7), fill_mode="nearest"),
        keras.layers.RandomRotation(0.5),
        keras.layers.RandomTranslation(0.5, 0.5, fill_mode="nearest"),
        # keras.layers.RandomCrop(int(height*scale), int(width*scale)),
        # keras.layers.RandomContrast(0.5),
    ])


def transformation(image):
    # print("transformation> ", type(image), image.shape)

    height, width = image.shape[:2]

    mode = "tf"

    transformed_image = transformation_model(height, width)

    data = np.zeros([height, width, 3], np.float32)
    data[:, :, 0] = image[:, :]
    data[:, :, 1] = image[:, :]
    data[:, :, 2] = image[:, :]

    img = transformed_image(data).numpy()
    # print(img.shape, type(img))

    img_finale = np.zeros([img.shape[0], img.shape[1]])
    for j in range(3):
        img_finale[:, :] += img[:, :, j]

    img_finale /= 3

    return img_finale


def build_data(data_size, images):
    # on sauvegarde les data non normlisées

    def f(x):
        sigma = 5
        x = gauss(mu=float(x), sigma=sigma)
        if x < 0: x = 0.
        if x > 255: x = 255.
        return x

    vf = np.vectorize(f)

    # print("build_data> ", images)
    shape = images[0].shape
    image_number = len(images)
    frac = 0.85
    first = True
    data_id = 0

    for i in range(data_size):
        for image_id, raw_img in enumerate(images):
            if data_id % 1000 == 0: print("generate data> data_id = ", data_id)
            # print(raw_img.shape)

            # Change rotation
            data = transformation(raw_img)

            """
            # visualisation de l'image finale
            print("forme", image_id, "data_id=", data_id)
            cv.imshow("input image", raw_img)
            cv.imshow("output image", data)
            cv.waitKey(0)
            """

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


def build_model_v1(shape, form_number):
    model = keras.models.Sequential()

    model.add(keras.layers.Input((shape[1], shape[2], 1)))

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(form_number, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  # loss='mse',
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


def do_run(figures, form_number, zoom, data_size, version, rebuild_forms, rebuild_data, rebuild_model, rebuild_forme=None):
    figures.set_zoom(zoom)

    os.makedirs("./data", mode=0o750, exist_ok=True)

    if rebuild_data:
        rebuild_model = True

        if rebuild_forms:
            images = figures.prepare_source_images(zoom=zoom, form_number= form_number, rebuild_forme=rebuild_forme)
        else:
            images = figures.load_source_images(zoom=zoom, form_number=form_number, rebuild_forme=rebuild_forme)

        # print("run> ", type(images), images)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Generating data from images")
        x_train, y_train, x_test, y_test = build_data(data_size, images)
    else:
        x_train, y_train, x_test, y_test = load_data()

    print("run> x_train : ", x_train.shape)
    print("run> y_train : ", y_train.shape)
    print("run> x_test : ", x_test.shape)
    print("run> y_test : ", y_test.shape)
    print("run> x_train : ", y_train[10:20, ])

    print('Before normalization : Min={}, max={}'.format(x_train.min(), x_train.max()))

    xmax = x_train.max()

    x_train = x_train / xmax
    x_test = x_test / xmax

    print('After normalization  : Min={}, max={}'.format(x_train.min(), x_train.max()))

    os.makedirs(DATA + "/run/models", mode=0o750, exist_ok=True)
    save_dir = DATA + "/run/models/best_model.h5"

    if rebuild_model:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Build model")

        if version == "v1":
            model = build_model_v1(x_train.shape, form_number)
        else:
            model = build_model_v2(x_train.shape)

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Start training")
        batch_size  = 512
        epochs      =  16

        savemodel_callback = keras.callbacks.ModelCheckpoint(filepath=save_dir, verbose=0, save_best_only=True)

        fit_verbosity = 1
        history = model.fit(x_train, y_train,
                            batch_size      = batch_size,
                            epochs          = epochs,
                            verbose         = fit_verbosity,
                            validation_data = (x_test, y_test),
                            callbacks = [savemodel_callback])

        pwk.plot_history(history, figsize=(form_number, 4), save_as='03-history')

    else:
        model = keras.models.load_model(save_dir)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Evaluate model")
    score = model.evaluate(x_test, y_test, verbose=0)

    if version == "v1":
        print(f'Test loss     : {score[0]:4.4f}')
        print(f'Test accuracy : {score[1]:4.4f}')
    else:
        print("x_test / loss  : {:5.4f}".format(score[0]))
        print("x_test / mae   : {:5.4f}".format(score[1]))
        print("x_test / mse   : {:5.4f}".format(score[2]))

        print("min(val_mae)   : {:.4f}".format(min(history.history["val_mae"])))

    return model, x_train, y_train, x_test, y_test


def handle_arguments():
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-formes", type=int, default=8, help="number of formes for training")
    argParser.add_argument("-cell", "--cell_size", type=int, default=40, help="cell size for figures")
    argParser.add_argument("-data", "--data_size", type=int, default=100, help="data size for traning")

    group1 = argParser.add_mutually_exclusive_group()
    group1.add_argument("-figures", action="store_true", help="build figures")
    group1.add_argument("-run", action="store_true", help="run")

    argParser.add_argument("-f", "--figure", type=int, default=None, help="figure to build")


    # par défaut: c'est l'action "load" qui est vraie
    argParser.add_argument("-build_data", action="store_true", help="build data")
    argParser.add_argument("-build_model", action="store_true", help="build model")

    args = argParser.parse_args()

    return args.cell_size, args.data_size, args.formes, args.figures, args.figure, \
           args.build_data, args.build_model, \
           args.run


# ===================================================================================================================


def main():
    zoom, data_size, form_number, build_figures, figure, build_data, build_model, run = handle_arguments()

    figures = Figures()

    # ============ generlal parameters=================
    os.makedirs("./dataset", mode=0o750, exist_ok=True)
    os.makedirs("./data", mode=0o750, exist_ok=True)

    if build_figures:
        if figure is None:
            print("Formes> Rebuild all figures")
            figures.prepare_source_images(zoom=zoom, form_number=form_number)
        else:
            print("Formes> Rebuild figure # {}".format(figure))
            figures.prepare_source_images(zoom=zoom, form_number=form_number, rebuild_forme=figure)
        return
    elif run:
        print("Formes> Run with build_data={} build_model={}".format(build_data, build_model))
    else:
        print("Formes> no action")
        return

    version = "v1"
    # version = "v2"

    model, x_train, y_train, x_test, y_test = do_run(figures,
                                                     form_number=form_number,
                                                     zoom=zoom,
                                                     data_size=data_size,
                                                     version=version,
                                                     rebuild_forms=False,
                                                     rebuild_data=build_data,
                                                     rebuild_model=build_model)

    # pwk.plot_images(x_train, y_train, [27], x_size=5, y_size=5, colorbar=True, save_as='01-one-digit')
    # pwk.plot_images(x_train, y_train, range(0,64), columns=8, save_as='02-many-digits')

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Predictions")

    """
    test_number = 1
    for i in range(10):
        n = randint(0, 6000)
        A = x_test[n:n+1, :, :, :]
        B = y_test[n:n+1]

        now = datetime.datetime.now()
        # y_sigmoid = model.predict(A, verbose=0)
        result = model(A)
        y_pred = np.argmax(result, axis=-1)
        t = datetime.datetime.now() - now
        errors = [i for i in range(len(A)) if y_pred[i] != B[i]]
        # print("errors", errors)
        print("n=", n, "A", A.shape, B, "pred=", y_pred[0], "durée=", t)

    return
    """

    print("len(x_test)=", len(x_test))
    result = model(x_test)
    y_pred = np.argmax(result, axis=-1)
    pwk.plot_images(x_test, y_test, range(0, 200), columns=8, x_size=1, y_size=1, y_pred=y_pred,
                    save_as='04-predictions')
    errors = [i for i in range(len(x_test)) if y_pred[i] != y_test[i]]
    # print("errors", errors)
    # errors=errors[:min(24,len(errors))]
    # pwk.plot_images(x_test, y_test, errors[:15], columns=8, x_size=2, y_size=2, y_pred=y_pred, save_as='05-some-errors')

    pwk.plot_confusion_matrix(y_test, y_pred, range(8), normalize=True, save_as='06-confusion-matrix')
    # pwk.end()

    return

if __name__ == "__main__":
    main()
