import argparse

import os
import numpy as np
import cv2 as cv
from random import *
import random as rng
import math
import re

import tensorflow as tf
from tensorflow import keras

HERE = os.path.normpath(os.path.dirname(__file__)).replace("\\", "/")
TOP = os.path.dirname(HERE)
DATA = TOP + "/AnumbyFormes"
FOND = TOP + "/AnumbyVehicule"

print(HERE, DATA)

# les couleurs de base
R = (0, 0, 255)
G = (0, 255, 0)
B = (255, 0, 0)
M = (255, 0, 255)
C = (255, 255, 0)
Y = (0, 255, 255)
COLOR_X = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))


# conversion Radians -> degrés
def rad2deg(angle):
    return 180 * angle / np.pi


# conversion degrés -> Radians
def deg2rad(angle):
    return np.pi * angle / 180


# Somme des éléments d'un tuple
def tuple_sum(x):
    return np.sum(x)


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


# Montrer les commandes actionnées par le clavier numérique
class Help(object):
    def __init__(self):
        # cet affichage se présente sous forme d'une grille 3 x 3 représentant
        # le clavier numérique
        self.cell = 60
        self.margin = 5
        self.width = self.margin*2 + self.cell*3
        self.height = self.margin*2 + self.cell*3

        # La grille
        self.image = np.zeros((self.height, self.width, 3), np.float32)
        for row in range(4):
            y = row*self.cell + self.margin
            cv.line(self.image, (self.margin, y), (self.margin + self.cell*3, y), color=G)
            for col in range(4):
                x = col * self.cell + self.margin
                cv.line(self.image, (x, self.margin), (x, self.margin + self.cell*3), color=G)

        # afficher un texte dans une cellule de la grille
        def draw_text(text, row, col):
            text_width, text_height = cv.getTextSize(text=text,
                                                     fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                                     fontScale=0.4,
                                                     thickness=1)[0]
            cv.putText(img=self.image, text=text, org=(self.xy(row, col, text_height, text_width)),
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=Y)

        draw_text("Gauche", 1, 0)
        draw_text("Droite", 1, 2)
        draw_text("Stop", 1, 1)

        draw_text("V+", 0, 1)
        draw_text("V-", 2, 1)

        draw_text("A-", 0, 0)
        draw_text("A+", 0, 2)

        draw_text("Avance", 2, 2)
        draw_text("Recule", 2, 0)

    # coordonnées d'une cellule
    def xy(self, row, col, dy=None, dx=None):
        y = row * self.cell + self.margin + int(self.cell/2)
        x = col * self.cell + self.margin + int((self.cell - dx) / 2)
        return x, y

    def draw(self):
        cv.imshow("Help", self.image)


# Représentation de la table de jeu
# on va y installer les figures tournées alatoirement
class Table(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.image = None
        self.hidden = None
        # Lors des essais on doit réinitialiser l'image et la liste des zones où on installe les figures
        self.reset_image()
        self.zones = []

    def reset_image(self):
        self.zones = []

        # on installe le fond clippé à la taille de la table
        with_fond = True

        self.image = np.ones((self.height, self.width, 3), np.uint8) * 130

        if with_fond:
            fond_origin = cv.imread(FOND + '/fond.jpg')
            shape = fond_origin.shape
            w = shape[1]
            if w > self.width: w = self.width
            h = shape[0]
            if h > self.height: h = self.height
            self.image[:h, :w, :] = fond_origin[:h, :w, :]

        self.hidden = np.ones((self.height, self.width, 3), np.uint8)
        self.hidden[:,:,:] = self.image[:,:,:]
        return self.image

    def update_hidden(self):
        self.hidden[:,:,:] = self.image[:,:,:]

    def draw(self):
        cv.imshow("Table", self.image)

    # au fur et à mesure que l'on installe les figures, on vérifie si la position proposée
    # n'est pas trop proche d'aucune figure déjà installée
    def test_occupe(self, n, y, x, margin):
        # print("test_occupe> 1 ================== est-ce que ", n, "(y, x)", y, x, "peut s'installer ?")
        if len(self.zones) == 0:
            # print("test_occupe> 2 crée la première zone (n, y, x)", n, y, x)
            self.zones.append((n, y, x))
            return True

        ok = True
        for i, zone in enumerate(self.zones):
            nz, yz, xz = zone

            # print("test_occupe> 3 teste la zone (n, y, x)", nz, yz, xz, yz - margin, yz + margin, xz - margin, xz + margin)

            if ((y + margin) > (yz - margin)) and \
                    ((y - margin) <= (yz + margin)) and \
                    ((x + margin) > (xz - margin)) and \
                    ((x - margin) <= (xz + margin)):
                # print("test_occupe> 4-1", i, "test=(", n, y, x, ") conflit avec la zone=(", nz, yz, xz, ")")
                ok = False
                break
            else:
                # print("test_occupe> 4-2", i, "test=(", n, y, x, ") compatible avec la zone=(", nz, yz, xz, ")")
                pass

        if ok:
            # print("test_occupe> 3 crée une nouvelle zone (n, y, x)", n, y, x)
            self.zones.append((n, y, x))
            return True

        return False

    # installation d'une figure sur la table
    #   on part de l'image brute d'une figure (figure tracée en noir sur un carré blanc)
    #   on retrace la figure en rouge
    #   on fait tourner cette image : le carré (tourné) se retrouve au centre d'une image élargie à fond noir
    #   on recopie cette image sur la table en considéront que le fond noir est transparent
    def install_form(self, table, n, image_origin):

        def rotation_positionnement(n, image, center, height, width):
            # essai de rotation/positionnement de la figure
            essai = 1
            while True:
                # on fait tourner la figure
                angle = randrange(360)
                rotate_matrix = cv.getRotationMatrix2D(center=center, angle=angle, scale=1.)
                rad = math.radians(angle)
                sin = math.sin(rad)
                cos = math.cos(rad)
                b_w = int((height * abs(sin)) + (width * abs(cos))) * 2
                b_h = int((height * abs(cos)) + (width * abs(sin))) * 2

                # print("b_w, b_h=", b_w, b_h, "center=", center)

                rotate_matrix[0, 2] += ((b_w / 2) - center[0])
                rotate_matrix[1, 2] += ((b_h / 2) - center[1])

                rotated = cv.warpAffine(src=image, M=rotate_matrix, dsize=(b_w, b_h))

                # on positionne la figure sur la table
                margin = 80
                y = randrange(margin, table.height - margin)
                x = randrange(margin, table.width - margin)
                # print("fond size=", table.height, table.width, "y, x=", y, x, "b_w, b_h=", b_w, b_h, "center=", center)

                # on teste si cette nouvelle position est compatible avec les figures déjà installées
                test = table.test_occupe(n, y, x, margin)
                if test:
                    # print(essai, "rotation_positionnement> libre n=", n, "y, x=", y, x)
                    return rotated, y, x
                else:
                    # print(essai, "rotation_positionnement> occupé n=", n, "y, x=", y, x)
                    essai += 1

        height, width = image_origin.shape[:2]

        # image = image_origin

        # colorie la figure en rouge (une figure brute ne contient que des pixels noirs)
        mask = (image_origin < 5) * 255
        image = mask.astype(np.uint8)
        image[:, :, 0:2] = 0
        image = image | image_origin

        center = (int(height / 2), int(width / 2))

        rotated, y, x = rotation_positionnement(n, image, center, height, width)
        # copie de la figure transformée sur la table
        crop(table.image, n, pos=(y, x), img=rotated)
        # draw_text(table.image, "{}".format(n), x, y, R)
        # print("rotation_positionnement> n=", n, "y, x=", y, x)



# installation d'un figure "img" sur la table "to_image" à une position "pos"
# chaque figure est tournée aléatoirement et donc elle est dessinée sur un fond noir
# une figure est un carré blanc et la figure elle-même est tracée en rouge
#
def crop(to_img, n, pos, img):
    to_height, to_width = to_img.shape[:2]
    height, width = img.shape[:2]

    def contraste():
        # augmentation du contraste de l'image de la figure pour obtenir 3 couleurs
        # - noir (le fond de l'image)
        # - blanc (la base carrée tournée de la figure)
        # - rouge (le dessin de la figure)
        for y in range(height):
            for x in range(width):
                c0 = img[y, x, 0]
                c1 = img[y, x, 1]
                c2 = img[y, x, 2]
                if c0 == 0 and c1 == 0 and c2 == 0:
                    # noir
                    pass
                elif c0 == 255 and c1 == 255 and c2 == 255:
                    # blanc
                    pass
                elif c0 == 0 and c1 == 0 and c2 == 255:
                    # rouge
                    img[y, x, 2] = 255
                elif c2 == 255:
                    # presque rouge
                    img[y, x, :] = 0
                    img[y, x, 2] = 255
                elif c0 == 0 and c1 == 0:
                    # presque rouge
                    img[y, x, :] = 0
                    img[y, x, 2] = 255
                elif c0 == c1 and c1 != c2:
                    # presque rouge
                    img[y, x, :] = 0
                    img[y, x, 2] = 255
                elif c0 == c1 and c1 == c2 and c0 < 128:
                    # gris foncé
                    img[y, x, :] = 0
                elif c0 == c1 and c1 == c2 and c0 >= 128:
                    # gris clair
                    img[y, x, :] = 255
                else:
                    # d'autres cas ??? mais ceci ne devrait jamais arriver
                    # print("crop> ", x, y, "color=", c0, c1, c2)
                    img[y, x, :] = 255

    def transfert():
        # print("crop> ========================================================================")
        # Transfert de la figure vers la table
        for y in range(y0, y1):
            for x in range(x0, x1):
                # print("crop> ", x, y, "seuil=", seuil, "s=", s)
                c0 = img[y - y0, x - x0, 0]
                c1 = img[y - y0, x - x0, 1]
                c2 = img[y - y0, x - x0, 2]

                # tout ce qui était noir deviendra transparent
                # tout ce qui était blanc reste blanc
                # tout ce qui était rouge devient noir
                if c0 < 5 and c1 < 5 and c2 > 5:
                    # rouge -> noir
                    to_img[y, x, :] = 0
                elif c0 < 5 and c1 < 5 and c2 < 5:
                    # noir -> transparent
                    pass
                else:
                    # recopie de l'image
                    to_img[y, x, :] = img[y - y0, x - x0, :]

    # augmentation du constraste
    contraste()

    # cv.imshow("img", img)
    # cv.waitKey()

    # print("crop> ", to_height, to_width, height, width)

    # encadrement sur la table où la figure va être installée
    y0, x0 = pos
    y0 -= int(height/2)
    x0 -= int(width/2)
    if y0 < 0:
        y0 = 0
    if x0 < 0:
        x0 = 0

    y1 = y0 + int(height)
    if y1 >= to_height:
        y1 = to_height - 1
    x1 = x0 + int(width)
    if x1 >= to_width:
        x1 = to_width - 1

    # transfert de l'image de la figure vaers la table
    transfert()

    # exit()


# Simulation de la Camera: on recopie une partie de l'image de la table située à la position courante du robot
class Camera(object):
    def __init__(self, size, model, data_min, data_max, forms, pattern):
        self.width = size
        self.height = size
        self.margin = 60
        self.w2 = int(self.width/2)
        self.h2 = int(self.height/2)
        self.camera = None
        self.hidden = None
        self.model = model
        self.data_min = data_min
        self.data_max = data_max
        self.forms = forms
        self.pattern = pattern

    def draw(self, table, model, x, y):
        # le robot ne peut jamais s'approcher trop près des bord de la table

        extra = 10
        seuil = self.width/2 + extra

        if x < seuil: x = seuil
        if x > (table.width - seuil): x = table.width - seuil

        if y < seuil: y = seuil
        if y > (table.height - seuil): y = table.height - seuil

        x_left = x - self.width/2
        x_right = x + self.width/2
        y_top = y - self.width/2
        y_bottom = y + self.width/2

        # cv.circle(img=table.image, center=(int(x), int(y)), radius=3, color=R, lineType=cv.FILLED)

        # on crée l'image courante correspondant à la position du robot agrandie de
        img_height = self.height + 2*extra + 1
        img_width = self.width + 2*extra + 1
        self.camera = np.zeros((img_height, img_width, 3), np.uint8)
        self.hidden = np.zeros((img_height, img_width, 3), np.uint8)

        y1 = int(y_top - extra)
        y2 = int(y1 + img_height)
        x1 = int(x_left - extra)
        x2 = int(x1 + img_width)
        self.camera[:, :, :] = table.image[y1:y2, x1:x2, :]
        self.hidden[:, :, :] = table.hidden[y1:y2, x1:x2, :]

        # print("extract from table>  (x_left, y_top)=", x_left, y_top, "(x_right, y_bottom)=", x_right, y_bottom,
        # "w=", x_right - x_left, "h=", y_bottom - y_top, "self.camera.shape=", self.camera.shape)

        # on dessine un rectangle vert
        cv.rectangle(self.camera, (extra, extra), (extra + self.width, extra + self.height), G, 1)

        form_find_figures(self.camera, self.model, self.forms, self.data_min, self.data_max, self.pattern)

        cv.imshow("camera", self.camera)
        # cv.waitKey()
        return


# cv.circle(img=src, center=(int(xc), int(yc)), radius=2, color=color, lineType=cv.FILLED)

def extract_contour(src, cnt, data_min, data_max):
    def demi_plan(src, x1, x2, y1, y2):
        x1 = xcorners[0]
        y1 = ycorners[0]
        x2 = xcorners[1]
        y2 = ycorners[1]
        m = (y2 - y1) / (x2 - x1)
        p = y1 - m * x1

        for x in range(x1, x2):
            for y in range(y1, y2):
                yy = m * x + p
                if y < yy:
                    src[y, x, 0] = 0
                    src[y, x, 1] = 0
                    src[y, x, 2] = 255
                if y == yy:
                    src[y, x, 0] = 0
                    src[y, x, 1] = 255
                    src[y, x, 2] = 255
                else:
                    src[y, x, 0] = 255
                    src[y, x, 1] = 0
                    src[y, x, 2] = 0

    x, y, w, h = cv.boundingRect(cnt)
    area = Area(cnt)
    xcorners, ycorners, center, alpha, radius = BoundingRectangle(cnt)
    xc = int(center[0])
    yc = int(center[1])
    side = int(np.sqrt(area))

    print("rectangle", x, y, w, h, xcorners, ycorners, "center", xc, yc)

    for s in range(4):
        demi_plan(src, xcorners[s], xcorners[s + 1], ycorners[s], ycorners[s + 1])

    cv.imshow("half", src)
    cv.waitKey(0)

    x1 = xc - side
    x2 = xc + side
    y1 = yc - side
    y2 = yc + side

    if (x2 - x1) != (y2 - y1): return

    cv.rectangle(src, (x1, y1), (x2, y2), G, 1)

    extract = np.zeros((y2 - y1, x2 - x1, 3), np.uint8)

    extract[:, :, :] = src[y1:y2, x1:x2, :]

    mask = (extract < 1) * 255
    red = mask.astype(np.uint8)
    red[:, :, 0:2] = 0
    extract = red | extract

    crop(extract, 0, (0, 0), extract)
    resized = cv.resize(extract, (80, 80), interpolation=cv.INTER_LINEAR)
    a = np.zeros((1, 80, 80, 1), np.float64)
    for i in range(3):
        a[0, :, :, 0] += resized[:, :, i]
    a = a / 3.

    a = a / data_max

    return a


def form_find_figures(src, model, forms, data_min, data_max, pattern):
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    shape = pattern[0].shape
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
        # if area <= 10: continue
        # La surface est évidemment dépendante du facteur de grandissement
        # sans grandissement on obtient une surface de l'ordre de 2500
        # il faudra construire un étalonnage en fonction du grandissement image réelle

        if area >= max_area:
            new_figure = i
            continue

        if i == new_figure + 1:
            ratio = (np.sqrt(area) - 1) / shape[0]
            print(i, "Area=", area, ratio, max_area)

            cv.drawContours(src, contours, i, R, 2)

            A = None

            extract = extract_contour(src, cnt, data_min, data_max)

            # print("data.shape", pattern[i:i + 1, :, :, :].shape)

            a = np.zeros_like(pattern)
            for r in range(a.shape[1]):
                for c in range(a.shape[2]):
                    a[0, r, c, 0] = extract[0, r, c, 0]

            print("shapes=", extract.shape, a.shape)

            result = model(a)
            r = np.zeros(8)
            for k in range(8):
                r[k] = result[0, k]
            a_test = np.argmax(r)

            print("prédiction=", forms[a_test])

            cv.imshow("extract", a[0])
            cv.waitKey()



            # return ratio

        # break
    return 0




# draw some texte upon an OpenCV image
def draw_text(img, text, x, y, color):
    coordinates = (x, y)
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5

    thickness = 1
    cv.putText(img, text, coordinates, font, fontScale, color, thickness, cv.LINE_AA)



"""
===================================================================================================================
"""

def handle_arguments():
    argParser = argparse.ArgumentParser()

    argParser.add_argument("-width", type=int, default=800, help="Largeur de la table")
    argParser.add_argument("-height", type=int, default=600, help="Hauteur de la table")
    argParser.add_argument("-camera_size", type=int, default=100, help="Taille de la caméra (carrée)")

    argParser.add_argument("-scale", type=int, default=None, help="Facteur d'échelle pour toutes les dimensions")


    args = argParser.parse_args()

    return args.width, args.height, args.camera_size, args.scale




def main():
    print("Vehicule")

    width, height, camera_size, scale = handle_arguments()

    if scale is not None:
        width = 800 * scale
        height = 600 * scale
        camera_size = 100 * scale

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

    # Initialisations
    forms = ["Rond", "Square", "Triangle", "Star5", "Star4", "Eclair", "Coeur", "Lune"]
    save_dir = DATA + "/run/models/best_model.h5"
    model = keras.models.load_model(save_dir)
    pattern = np.load(DATA + "/dataset/pattern.npy", allow_pickle=True)

    table = Table(width, height)
    camera = Camera(camera_size, model, data_min, data_max, forms, pattern)

    help = Help()

    # initialisation des formes brutes
    images = []
    for form in forms:
        image = cv.imread(DATA + '/dataset/{}/RawImages{}.jpg'.format(form, form))
        images.append(image)

    # préparation de la table
    table.reset_image()

    for f, form in enumerate(forms):
        table.install_form(table, f, images[f])

    # il existe deux versions de l'image de la table:
    #  - une qui est affichée (avec les traces de travail)
    #  - une qui ne contient que le fond et les images tournées
    table.update_hidden()

    # dessine une barrière pour les limites de déplacement du véhicule sur la table
    m = 20
    cv.rectangle(table.image, (m, m), (table.width - m - 1, table.height - m - 1), (0, 255, 255), 1)

    # pour tester on cherche toutes les fiures installées sur la table
    form_find_figures(table.image, model, forms, data_min, data_max, pattern)

    # première visualisation de la table
    cv.imshow("table", table.image)
    cv.waitKey()

    return

    # gestion des déplacement du véhicule.
    # le véhicule est positionné au départ au milieu de la table
    x = table.width/2.
    y = table.height/2.

    # variables de contrôle du véhicule
    #  alpha: orientation du déplacement
    #  v:     vitesse
    #  a:     accélération
    #  t:     temps
    alpha = 0
    v = 0
    a = 1
    t = 0
    dt = 1
    d = 1

    raw_w, raw_h = images[0].shape[0:2]
    raw_w2 = int(raw_w/2)
    raw_h2 = int(raw_h/2)

    # pilotage avec le clavier numérique du robot
    while True:
        k = cv.waitKey(0)
        # print("k=", k)

        zero = 48
        if k == zero + 4:
            # tourne à droite
            alpha -= 10
        elif k == zero + 6:
            # tourne à gauche
            alpha += 10

        if k == zero + 7:
            # freine
            a -= 1
        elif k == zero + 9:
            # accélére
            a += 1
        if a < 0:
            a = 0

        if k == zero + 8:
            v += a
        elif k == zero + 2:
            v -= a

        if k == zero + 1:
            # recule
            d = -1
        elif k == zero + 3:
            # avance
            d = 1

        if k == zero + 5:
            # stoppe
            a = 1
            v = 0
            alpha = 0

        if v > 0:
            x += d * v * dt * np.cos(deg2rad(alpha))
            if x < (camera.w2 + raw_w2):
                x = camera.w2 + raw_w2
            if x >= table.width - camera.w2 - raw_w2:
                x = table.width - camera.w2 - raw_w2 - 1

            y += d * v * dt * np.sin(deg2rad(alpha))
            if y < (camera.h2 + raw_h2):
                y = camera.h2 + raw_h2
            if y >= table.height - camera.h2 - raw_h2:
                y = table.height - camera.h2 - raw_h2 - 1

        # print("t=", t, "(x, y)=", x, y, "v=", v, "alpha=", alpha, "a=", a)
        table.draw()
        camera.draw(table, model, x, y)

        # camera.find_figures(model, image)
        # cv.waitKey(0)

        help.draw()

        t += dt

        # sortie de l'application
        if k == 27:
            break
        if k == 113:
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
