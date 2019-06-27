#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 13 14:15:50 2019
@author: nicolas
"""

#Importation de bibliothèque
import cv2
import couleur
import os.path
from skimage.feature import greycomatrix
from skimage.feature.texture import greycoprops
import numpy as np

pathDataset = "./dataset"
pathFichierTrain = "./train"

"""  DEUXIEME PARTIE MATRICE DE COOCCURENCE  """

#Transformation d'une image en niveau de gris
def Gris(chemin):
    image = cv2.imread(chemin)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = normalisationImage(gray)
    return gray

#Normalisation de l'image
def normalisationImage(image):
    normImage = image//16
    normImage = normImage.astype('uint32')
    return normImage

    
#Histogramme d'une image à niveau de gris
def Histogramme(mat):
    histogramme = cv2.calcHist([mat], [0], None, [24,24, 24],[0, 256])
    return histogramme
    
#Calcul de la distance
def CalculDistance(image1,image2):
    d = (np.linalg.norm((image1-image2))/5)
    return d

#Matrice de co-occurence
def MatCooccurence(image_gris):
    matCo = greycomatrix(image_gris, [5], [0,np.pi/2,np.pi/4,(np.pi*3)/4], 256,
                         symmetric=True, normed=True)
    return matCo

#Calcul des parametre de la co occurence
def ParamCooccurence(HistomatCoo):
    #Calcul de l'energie
    energie = greycoprops(HistomatCoo,'energy')
    contraste = greycoprops(HistomatCoo,'contrast')
    dissimilarite = greycoprops(HistomatCoo,'dissimilarity')
    homogeneite = greycoprops(HistomatCoo,'homogeneity')
    correlation = greycoprops(HistomatCoo,'correlation')
    return energie, contraste, dissimilarite,homogeneite,correlation

#Apprentissage
def Apprentissage():
    #Création du fichier de stockage des histogrammes
    f = open((pathFichierTrain+"/cooccurence"+".txt"),'wb')
    listeImage = os.listdir(pathDataset)
    paramCooccure = {}
    for image in listeImage:
        param = np.zeros(5)
        greyImage = Gris(pathDataset+"/"+image)
        MatCoo = MatCooccurence(greyImage)
        energie,contraste,dissimilarite,homogeneite,correlation = ParamCooccurence(MatCoo)
        energie = energie[0][0]
        param[0] = energie
        contraste = contraste[0][0]
        param[1] = contraste
        dissimilarite = dissimilarite[0][0]
        param[2] = dissimilarite
        homogeneite = homogeneite[0][0]
        param[3] = homogeneite
        correlation = correlation[0][0]
        param[4] = correlation
        paramCooccure[image] = param
        
    couleur.pickle_hist(f,paramCooccure)
    f.close
    
#Recherche des ressemblance   
def Ressemblance(chemin,k):
    listeDistance ={}
    param = np.zeros(5)
    image_gris = Gris(chemin)
    MatCoo = MatCooccurence(image_gris)
    energie,contraste,dissimilarite,homogeneite,correlation = ParamCooccurence(MatCoo)
    energie = energie[0][0]
    param[0] = energie
    contraste = contraste[0][0]
    param[1] = contraste
    dissimilarite = dissimilarite[0][0]
    param[2] = dissimilarite
    homogeneite = homogeneite[0][0]
    param[3] = homogeneite
    correlation = correlation[0][0]
    param[4] = correlation
    f = open((pathFichierTrain+"/cooccurence"+".txt"),"rb")
    list_hist = couleur.unpickle_hist(f)
    for key, valeur in list_hist.items():
        valeur = np.asanyarray(valeur)
        listeDistance[CalculDistance(valeur,param)] = key
    listeDistances = sorted(listeDistance.items(), key=lambda t: t[0])
    f.close
    return(listeDistances[:k])

def main():
    #Apprentissage()
    print("********************************************************************")
    print("                                                                     ")
    print("              DISTANCE EN UTILLISANT LA MATRICE DE COOCCURRENCE                  ")
    print(" Auteur: Nicolas & Olivier                                                                                        ")
    print("                                                                                         ")
    chemin = input("Entrer le chemin de l'image: ")
    k = int(input("Entrer le nombre de ressemblances à trouver: "))
    listeImage = Ressemblance(chemin,k)
    chemin = chemin.split("/")
    classe = (chemin[len(chemin)-1]).split("_")
    classe = classe[0]
    print("----------------------------------------------------------")
    print("                                                        ")
    print("                                                        ")
    print("------------LISTE des resssemblances en utilisant la cooccurence---------------")
    print("                                                        ")  
    nbTrouve = 0
    for i in range(len(listeImage)):
        print(listeImage[i][1],"                         ",listeImage[i][0])
        chaine = listeImage[i][1] 
        chaine = chaine.split("_")
        if(classe == chaine[0]):
            nbTrouve=1+nbTrouve
    #Evatualtion
    print("                                                        ")

    print("-------         EVALUATION     ---------")       
    precision = nbTrouve/k  
    rappel = nbTrouve/72
    fmesure = 2/((1/precision)+(1/rappel))
    print("La précision est de = ", precision)
    print("Le rappel est de = ", rappel)
    print("La F-mesure est de = ", fmesure)

    print("********     FIN      ***********")


if __name__ == '__main__':
    main()     