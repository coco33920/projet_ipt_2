import numpy as np
import time
import os
from math import sqrt
from matplotlib import pyplot as plt
from copy import deepcopy


class Progressbar: 
    """ 
        Simple class to emulate progress bar in standard out
        @params: 
            title: name of the progress bar (displayed first)
            total: number of jobs to be executed
            length: default 60, length of the progress bar
            char: default '#' character used to display the progress
    """
    def __init__(self,title,total,length=60,char="#"):
        self.title = title
        self.total = total
        self.started = time.time()
        self.length = self.compute_ideal_length()
        self.char = char
    
    @staticmethod
    def time_display(second):
        """Format time in second with the hours-minute-second format"""
        return time.strftime("%H:%M:%S", time.gmtime(second))

    def pretty_time_display(self):
        """Format time in second from the starting point"""
        return self.time_display(time.time() - self.started)
    

    def compute_eta(self,current):
        """Compute the ETA to finish the jobs based on the current time and progress"""
        elapsed = (time.time() - self.started)
        total = int((float(self.total)*float(elapsed))/float(current)) #produit en croix
        return total-elapsed #temps restant

    def compute_ideal_length(self):
        """Compute the ideal length of the bar based on the width of the terminal"""
        width = os.get_terminal_size()[0] #total width
        number = len(str(self.total))*2 + 6 #width used by the progress display (XXX/XXX) 
        title_width = 15 #width of the title
        end_width = 36 #width used by the XXX% and time elapsed/ETA display and the blank after the |
        remaining_size = width - (number+title_width+end_width)
        return remaining_size

    def update(self,current):
        """Updates the progress of the bar"""
        current = current + 1 #on va de 0 à total-1 sinon...
        self.length = self.compute_ideal_length() #au cas où l'envie de réduire le terminal vienne à l'idée...
        progress = (float(current)/float(self.total)) #proportion
        inner = self.char*int(((progress)*(self.length))) #nombre de caractère à afficher
        space = ' '*int((self.length-len(inner))) #reste rempli avec des espaces
       
        title_padding =' '*(15-len(self.title)) #padding après le titre pour rester aligner

        progress_padding = ' '*(len(str(self.total))-len(str(current))) #padding avant le nombre pour rester aligner quand on passe de 0 à 10 à 100 etc.
        percent_padding = 1 #padding avant le pourcentage

        if progress < 0.1:
            percent_padding = 2

        if current == self.total:
            percent_padding = 0

        percent_padding = ' '*percent_padding
        eta = self.time_display(self.compute_eta(current))

        print("%s%s %s(%d / %d) |%s%s| %s%d%% Durée %s ETA %s" % (
            self.title,
            title_padding,
            progress_padding,
            current,
            self.total,
            inner,
            space,
            percent_padding,
            progress*100,
            self.pretty_time_display(),
            eta), end='\r')
        

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
    #masque recurrence (0,0)->(p-1,p-1) ==> (i',j') dans tableau
    #liaison une case de masque et une case de tableau
    #(p//2,p//2) masque <--> (i,j) tableau (1)
    #(0,0) masque <--> (i-p//2,j-p//2) tableau (2)
    #(p-1,p-1) masque <--> (i-p//2+p-1, j-p//2+p-1) = (i+p//2,j+p//2)
    #(i',j') masque <--> (i+i'-p//2,j+j'-p//2) tableau
    
    n,m = tableau.shape
    p,_ = masque.shape
    p2 = p//2
    somme = 0
    for ip in range(p):
        for jp in range(p):
            somme += masque[ip][jp] * tableau[i+ip-p2][j+jp-p2]
    
    return somme

def masque(image, mask, multiplier=1, name="Masque",char="#"):
    #traitement sur chaque (i,j) de image
    C = np.zeros(image.shape)
    n,m = image.shape
    bar = Progressbar(name, n,char=char)
    for i in range(n):
        for j in range(m):
            if(superposition(i,j,image,mask)):
                C[i][j] = calcul(i,j,image,mask) #on calcule
            else:
                C[i][j] = image[i][j]*multiplier #c'est en dehors
        bar.update(i)
    print("")
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
   
    image = masque(image, mask, name="Bruit")
    return image


def f(x,y):
    return sqrt(x**2 + y**2)

def gradient(image):
    Gx = masque(image, Mx, 0, name="Gradient X")
    Gy = masque(image, My, 0, name="Gradient Y") 
    
    G = np.zeros(Gx.shape)

    n,m = Gx.shape

    bar = Progressbar("Gradient", n)
    for i in range(n):
        for j in range(m):
            G[i][j] = f(Gx[i][j], Gy[i][j])
        bar.update(i)
    print("")
    return (Gx,Gy,G)

def affinage(Gx,Gy,image):
    sortie = deepcopy(image)
    n,m = Gx.shape
    bar = Progressbar("Affinage", n-1)
    for i in range(1,n-1):
        for j in range(1,m-1):
            norme = image[i][j] #norme
            if norme == 0:
                continue
            vecteur = np.array([Gx[i][j], Gy[i][j]]) #vecteur gradient
            vecteur_normalise = (1/norme) * vecteur #normalisation du vecteur
            alpha = np.arctan2(vecteur[1],vecteur[0]) #angle € [-pi,pi]

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

            if image[i+x][j+y] > image[i][j]:
                sortie[i][j] = 0
        bar.update(i)
    print("")
    return sortie


def voisin(i,j,image): #renvoi le nombre de voisin blanc du pixel (i,j)
    for x in range(3): #1
        for y in range(3): #1
            if x==y==1: #pour éviter le cas où image[i][j] == 1 si seuil_haut = 1 
                continue
            if image[i+(x-1)][j+(y-1)] == 1:
                return True
    return False

def seuillage(seuil_haut, seuil_bas,image): #implémente l'algorithme de seuillage
    temp = 0 #nombre de pixel avec un voisin blanc
    s21 = [] #nombre de pixel entre s2 et s1
    n,m = image.shape
    sortie = deepcopy(image) 
    bar1 = Progressbar("Seuillage", n) #déclaration de la progressbar
    for i in range(n):
        for j in range(m):
            if image[i][j] >= seuil_haut: 
                sortie[i][j] = 1 #on met à blanc si on dépasse le seuil haut
            elif image[i][j] <= seuil_bas: 
                sortie[i][j] = 0 #on met à noir si on est en dessous
            else:
                s21.append((i,j))
                temp+=int(voisin(i,j,image)) #on update temp si le pixel est entre s2 et s1
        bar1.update(i)
    
    while temp != 0:
        for i,j in s21:
            blanc = voisin(i,j,image) #on vérifie si le pixel a un voisin blanc
            if blanc == True:
                sortie[i][j] = 1
                s21.remove((i,j))
                temp-=1
        
    for i,j in s21:
        sortie[i][j] = 0 #on met le reste à 0
    print("")
    return sortie


#Fonction générale pour récupérer une entrée clavier et vérifier qu'elle est valide
#fonction predicate => renvoi TRUE si le résultat peut être gardé, False sinon
def get(string, f, except_string, default=0, predicate=lambda x: True, additionnal_string=""):
    temp = default
    while temp == default:
        try:
            temp = input(string) #on prend l'entrée
            temp = f(temp) #on applique la fonction de traitement dessus
            result = predicate(temp) #on vérifie qu'elle est valide avec le prédicat donné
            if result == False: #si elle est invalide on reboucle
                print(additionnal_string) 
                temp = default
        except KeyboardInterrupt: #pour pouvoir sortir du programme avec CTRL + C
            exit("Interruption du programme")
        except: #si on a un soucis on reboucle
            print(except_string)
            temp = default
    return temp

def get_phrase(count):
    if count < 5:
        return "Êtes-vous satisfait.e ? (O/N) : "
    elif count < 7:
        return "T'as pas encore fini???? (O/N) : "
    elif count < 10:
        return "Sérieusement..... : "
    else:
        return "Je vais envoyer Ô Grand Lapinou vous chercher.... "

#Fonction générale pour éviter de répéter la même chose dans les étapes => continue tant que "O" n'est pas donné
#Pour les fonction demandant des arguments supplémentaire la liste des fonction à appliquer est en paramètre
def etape(traitement,out, arguments, additional_args_functions = []):
    count = 0
    while True:
        args = []        
        for f in additional_args_functions:
            try:
                args.append(f()) #pour la majorité des fonction elles n'ont pas d'argument
            except TypeError: 
                args.append(f(args[-1])) #si elles en ont un => seuil_bas => on met le dernier argument en date (seuil_haut)
            
        argument = args + arguments #on ajoute les arguments dans l'ordre (image à la fin)
        sortie = traitement(*argument)
        
        image = sortie
        if isinstance(sortie, tuple): #si la sortie est un tuple => gradient => l'image est le 3eme element
            image = sortie[2]


        plt.imsave("result/"+out+".png",image,cmap='gray') #on sauvegarde l'image
        a = input(get_phrase(count)) #on demande si l'utilisateur.ice est satisfait.e
        if a == "O":
            print(" ") 
            return sortie #si oui on renvoi la sortie du traitement
        count += 1

image1 = get("Image à charger (.png) : ", image_to_array, "Ceci n'est pas une image", default=[])
print(" ")

taille = lambda: get("Taille du masque : ", int, "Veuillez indiquer un nombre", predicate=lambda x: x%2==1, additionnal_string="Veuillez donner un nombre impair")
ecart_type = lambda: get("Ecart Type : ", float, "Veuillez indiquer un nombre")
bruit_functions = [taille,ecart_type]
image2 = etape(bruit, "bruit", [image1], bruit_functions)

Gx,Gy,image3 = etape(gradient, "gradient", [image2])
image4 = etape(affinage, "affinage", [Gx,Gy,image3])

seuil_haut = lambda: get("Seuil haut : ", float, "Veuillez indiquer un nombre", predicate=lambda x: x >= 0 and x <= 1, additionnal_string="Veuille entrer un nombre entre 0 et 1")
seuil_bas = lambda y: get("Seuil bas : ", float, "Veuillez indiquer un nombre", predicate=lambda x: x >= 0 and x < y, additionnal_string="Veuille entrer un nombre entre 0 et "+str(y))
seuillage_function = [seuil_haut,seuil_bas]

image5 = etape(seuillage, "seuillage", [image4], seuillage_function)
