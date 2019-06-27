#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 08:47:53 2019

@author: nicolas
"""

#Importation des bibliothèques
import cv2
import numpy as np
import os.path
import pickle
from scipy.spatial import distance
import time

start_time = time.time()
pathDataset = "./dataset"
pathFichierTrain = "./train"

#Calcul de l'histogramme et normalisation connaissant le chemin de l'image
def Histogramme(chemin):
    image = cv2.imread(chemin)
    histogramme = cv2.calcHist([image], [0, 1, 2], None, [8,8,8],[0, 256, 0, 256, 0, 256])
    histogramme = cv2.normalize(histogramme, histogramme).flatten()
    return histogramme
    
#Creation du fichier pour stockage des descripteurs pickle
def pickle_hist(fichier,histogramme):   
    pkl=pickle.Pickler(fichier)
    pkl.dump(histogramme)

#Récupération du descripteur Unpickle
def unpickle_hist(fichier):   
    Unpkl=pickle.Unpickler(fichier)
    fic=(Unpkl.load())
    return fic

#Fonction de stockage des histogrammes normalisées
def Apprentissage():
    #Création du fichier de stockage des histogrammes
    f = open((pathFichierTrain+"/histogramme"+".txt"),'wb')
    listeImage = os.listdir(pathDataset)
    hist_obj = {}
    for image in listeImage:
        histogramme = Histogramme(pathDataset+"/"+image)
        hist_obj[image] = histogramme
    pickle_hist(f,hist_obj)
    f.close

#Fonction de calcul de distance entre deux histogrammes   
def CalculDistance(hist1,hist2):
    distances = distance.euclidean(hist1,hist2)
    return distances

#Chercher les plus proches voisins
def RessemblaceImage(cheminImageTest,k):
    listeDistance ={}
    #Calcul de l'histogramme de l'image de test
    hist1 = Histogramme(cheminImageTest)
    hist1 = np.asanyarray(hist1)
    f = open((pathFichierTrain+"/histogramme"+".txt"),"rb")
    list_hist = unpickle_hist(f)
    for key, valeur in list_hist.items():
        valeur = np.asanyarray(valeur)
        CalculDistance(valeur,hist1)
        listeDistance[CalculDistance(valeur,hist1)] = key
    listeDistances = sorted(listeDistance.items(), key=lambda t: t[0])
    f.close
    return(listeDistances[:k])
   

    
def main():
    #Apprentissage()
    #print("Durée d'apprentissage = ", start_time)
    print("********************************************************************")
    print("*                                                                  *")
    print("*         RECHERCHE D'IMAGE EN UTILISANT L'HISTOGRAMME             *")
    print(" Auteur: Nicolas & Olivier                                                                                        ")
    print("                                                                                         ")
    chemin = input("=> Entrer le chemin de l'image: ")
    k = int(input("=> Nombre d'image a cherché: "))
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

    print("-------       EVALUATION     ---------")       
    precision = nbTrouve/k  
    rappel = nbTrouve/72
    fmesure = 2/((1/precision)+(1/rappel))
    print("La précision est de = ", precision)
    print("Le rappel est de = ", rappel)
    print("La F-mesure est de = ", fmesure)

if __name__ == '__main__':
    main()


        
        
    
    
    