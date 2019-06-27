#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 23:37:12 2019

@author: nicolas
"""
import time
import cv2
import couleur
from scipy.spatial import distance
import os.path
start_time = time.time()
import numpy as np

pathDataset = "./dataset"
pathFichierTrain = "./train"

"""  DEUXIEME PARTIE MOMENTS DE HU  """
#Transformation d'une image en niveau de gris
def Gris(chemin):
    gray = cv2.imread(chemin,cv2.IMREAD_GRAYSCALE)
    _,im = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return im

#Normalisation de l'image
def normalisationImage(image):
    normImage = image//8
    normImage = normImage.astype('uint32')
    return normImage

#Calcul des moment de HU
def momentHu(image):
    moments = cv2.moments(image)
    huMoments = cv2.HuMoments(moments)
    return huMoments

#Fonction de calcul de distance euclidienne entre deux moment de Hu   
def CalculDistance(moment1,moment2):
    distances = distance.euclidean(moment1,moment2)
    return distances

#Fonction de stockage des moments de hu 
def Apprentissage():
    #Création du fichier de stockage des histogrammes
    f = open((pathFichierTrain+"/moment"+".txt"),'wb')
    listeImage = os.listdir(pathDataset)
    moment_obj = {}
    for image in listeImage:
        chemin = (pathDataset+"/"+image)
        gris = Gris(chemin)
        moment = momentHu(gris)
        moment_obj[image] = moment
    couleur.pickle_hist(f,moment_obj)
    f.close

#Chercher les plus proches voisins
def RessemblaceImage(cheminImageTest,k):
    listeDistance ={}
    #Calcul des moments de hu de l'image de test
    gris = Gris(cheminImageTest)
    moment_test = momentHu(gris)
    f = open((pathFichierTrain+"/moment"+".txt"),"rb")
    list_hist = couleur.unpickle_hist(f)
    for key, valeur in list_hist.items():
        d = CalculDistance(valeur,moment_test)
        listeDistance[d] = key
    listeDistances = sorted(listeDistance.items(), key=lambda t:t[0])
    f.close
    
    return(listeDistances[:k])   
    
def main():
    print("********************************************************************")
    #Apprentissage()
    print("                                                                     ")
    print("              DISTANCE EN UTILLISANT LA FORME (MOMENT DE HU)                  ")
    print(" Auteur: Nicolas & Olivier                                                                                        ")
    print("                                                                                         ")
    chemin = input("Entrer le chemin de l'image: ")
    k = int(input("Nombre d'image a cherché: "))
    """k = (input("Nombre d'image a cherché: "))
    gris = Gris(chemin)
    moment_test1 = momentHu(gris)
    
    gris2 = Gris(k)
    moment_test2 = momentHu(gris2)
    print(CalculDistance(moment_test1,moment_test2))"""
    listeImage = RessemblaceImage(chemin,k)
    chemin = chemin.split("/")
    classe = (chemin[len(chemin)-1]).split("_")
    classe = classe[0]
    print("                                                        ")
    print("-------------LISTE des resssemblances---------------")
    print("                                                        ")
    print("Images                                 Distance")
    nbTrouve = 0
    for i in range(len(listeImage)):
        print(listeImage[i][1],"                         ",listeImage[i][0])
        chaine = listeImage[i][1] 
        chaine = chaine.split("_")
        if(classe == chaine[0]):
            nbTrouve=1+nbTrouve
    #Evatualtion 
    print("                                                        ")
    print("-------      EVALUATION      ---------")       
    precision = nbTrouve/k  
    rappel = nbTrouve/72
    fmesure = 2/((1/precision)+(1/rappel))
    print("La précision est de = ", precision)
    print("Le rappel est de = ", rappel)
    print("La F-mesure est de = ", fmesure)
        
if __name__ == '__main__':
    main()

