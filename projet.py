import numpy as np
import time
from math import sqrt
from matplotlib import pyplot as plt
from copy import deepcopy

Mx,My = np.zeros((3,3)),np.zeros((3,3))

Mx[0][0] = -1
Mx[0][1] = -2
Mx[0][2] = -1
Mx[2][0] = 1
Mx[2][1] = 2
Mx[2][2] = 1

My = Mx.transpose()

def image_to_array(filename):
    temp = plt.imread(filename)
    Im = np.zeros(np.shape(temp)[0:2])
    Im = temp[:,:,0]
    return Im

def superposition(i,j,tableau,masque):
    n,m = tableau.shape
    p,_ = masque.shape
    p2 = p//2

    gauche = not ((i-p//2 < 0) or (j-p//2 < 0)) #True si c'est ok, False sinon, le côté gauche dépasse du tableau
    droite = not ((i+p//2 >= n) or (j+p//2 >= m)) #True si c'est ok, False sinon, le côté droite dépasse du tableau

    return gauche and droite

def calcul(i,j,tableau,masque):
    #c i,j = combinaison linéaire du masque et de tableau
    n,m = tableau.shape
    p,_ = masque.shape
    p2 = p//2
    #masque recurrence (0,0)->(p-1,p-1) ==> (i',j') dans tableau
    #liaison une case de masque et une case de tableau
    #(p//2,p//2) masque <--> (i,j) tableau (1)
    #(0,0) masque <--> (i-p//2,j-p//2) tableau (2)
    #(p-1,p-1) masque <--> (i-p//2+p-1, j-p//2+p-1) = (i+p//2,j+p//2)
    #(i',j') masque <--> (i+i'-p//2,j+j'-p//2) tableau
    somme = 0
    for ip in range(p):
        for jp in range(p):
            somme += masque[ip][jp] * tableau[i+ip-p2][j+jp-p2]
    
    return somme

def masque(image, mask, multiplier=1):
    C = np.zeros(image.shape)
    #traitement sur chaque (i,j) de image
    n,m = image.shape
    for i in range(n):
        if i%100==0: print("Calcul de la ligne", i)
        for j in range(m):
            if(superposition(i,j,image,mask)):
                C[i][j] = calcul(i,j,image,mask) #on calcule
            else:
                C[i][j] = image[i][j]*multiplier #c'est en dehors
            pass

    return C

def gausse(x,y,sigma):
    a = np.exp(-1 * ((x**2 + y**2)/(2 * sigma**2)))
    return (1/(2*np.pi*sigma**2))*a

def bruit(taille, sigma, image):
    mask = np.zeros((taille,taille))
    t2 = taille//2
   
    for i in range(taille):                                         
        for j in range(taille):
            mask[i][j] = gausse(i-t2,j-t2,sigma)
   
    image = masque(image, mask)
    return image


def f(x,y):
    return sqrt(x**2 + y**2)

def gradient(image):
    Gx = masque(image, Mx, 0)
    Gy = masque(image, My, 0) 
    
    G = np.zeros(Gx.shape)

    n,m = Gx.shape

    for i in range(n):
        for j in range(m):
            G[i][j] = f(Gx[i][j], Gy[i][j])

    return (Gx,Gy,G)

def affinage(Gx,Gy,image):
    sortie = deepcopy(image)
    n,m = Gx.shape

    for i in range(1,n-1):
        for j in range(1,m-1):
            norme = image[i][j] #norme
            if norme == 0:
                continue
            vecteur = np.array([Gx[i][j], Gy[i][j]]) #vecteur gradient
            vecteur_normalise = (1/norme) * vecteur #normalisation du vecteur
            alpha = np.arctan2(vecteur[1],vecteur[0]) #angle € [-pi,pi]
            #carré 1 => -pi/8 à pi/8
            #carré 2 => pi/8 à 3pi/8
            #carré 3 => 3pi/8 à 5pi/8
            #carré 4 => 5pi/8 à 7pi/8
            #carré 5 => 7pi/8 à -7pi/8
            #carré 6 => -7pi/8 à -5pi/8
            #carré 7 => -5pi/8 à -3pi/8
            #carré 8 => -3pi/8 à -pi/8
            x,y = 1,0
            appart = lambda min, max: alpha >= min and alpha < max
            appart2 = lambda max, min: alpha >= min and alpha < max
            pi8 = np.pi/8
            if appart(0, pi8):
                x,y=1,0 #carré à droite
            elif appart(pi8,3*pi8):
                x,y=1,1 #carré en haut à droite
            elif appart(3*pi8, 5*pi8):
                x,y=0,1 #carré en haut
            elif appart(5*pi8, 7*pi8):
                x,y=-1,1 #carré en haut à gauche
            elif appart(7*pi8,np.pi):
                x,y=-1,0 #carré à gauche
            elif appart2(-7*pi8, -np.pi):
                x,y=-1,0 
            elif appart2(-5*pi8, -7*pi8):
                x,y=-1,-1 #en bas à gauche
            elif appart2(-3*pi8, -5*pi8):
                x,y=0,-1 #en bas
            elif appart2(-pi8, -3*pi8):
                x,y=1,-1 #en bas à droite
            else:
                x,y=1,0 #à droite

            if(image[i+x][j+y] > image[i][j]):
                sortie[i][j] = 0
    return sortie


def voisin(i,j,image): #renvoi le nombre de voisin blanc du pixel (i,j)
    for x in range(3): #1
        for y in range(3): #1
            if x==y==1: #pour éviter le cas où image[i][j] == 1 si seuil_haut = 1 
                continue
            if image[i+(x-1)][j+(y-1)] == 1:
                return True
    return False

def seuillage(image, seuil_bas, seuil_haut):
    temp = 0
    s21 = []
    n,m = image.shape
    sortie = deepcopy(image)
    for i in range(n):
        for j in range(m):
            if image[i][j] >= seuil_haut: 
                sortie[i][j] = 1
            elif image[i][j] <= seuil_bas:
                sortie[i][j] = 0
            else:
                s21.append((i,j))
                temp+=int(voisin(i,j,image))
    while temp != 0:
        for i,j in s21:
            blanc = voisin(i,j,image)
            if blanc == True:
                sortie[i][j] = 1
                s21.remove((i,j))
            temp-=1
        
    for i,j in s21:
        sortie[i][j] = 0

    return sortie

fig,(a1,a2) = plt.subplots(nrows=1, ncols=2)
a1.set(xlabel="image précédente")
a2.set(xlabel="image courrante")


def get(string, f, except_string, default=0):
    temp = default
    while(temp == default):
        try:
            temp = input(string)
            temp = f(temp)
        except:
            print(except_string)
            temp = default
    return temp

#image1 = []
#while(image1 == []):
#    try:
#        image = input("Image à charger (.png) : ")
#        image1 = image_to_array(image)
#    except:
#        print("Ceci n'est pas une image")
#        image1 = []

image1 = get("Image à charger (.png) : ", image_to_array, "Ceci n'est pas une image", default=[])

taille = 0
while(taille == 0):
    try:
        taille = int(input("Taille du masque : "))
        if(taille%2 == 0):
            print("Veuillez donner un nombre impair")
            taille = 0
    except:
        print("Veuillez indiquer un nombre")
        taille = 0

sigma = get("Veuillez entrer l'écart-type : ", float, "Veuillez indiquer un nombre", default=0)
