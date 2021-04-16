import numpy as np
from matplotlib import pyplot as plt

def image_to_array(filename):
    temp = plt.imread(filename + '.png')
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

def masque(image, masque):
    C = np.zeros(image.shape)
    #traitement sur chaque (i,j) de image
    n,m = image.shape
    for i in range(n):
        for j in range(m):
            if(superposition(i,j,image,masque)):
                C[i][j] = calcul(i,j,image,masque) #on calcule
            else:
                C[i][j] = image[i][j] #c'est en dehors
            pass

    return C

def gausse(x,y,sigma):
    a = np.exp(-1 * ((x**2 + y**2)/(2 * sigma**2)))
    return (1/(2*np.pi*sigma**2))*a

def bruit(taille, sigma, image):
    pass