#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 23:22:13 2019

@author: nicolas
"""

import couleur
import moment
import cooccurence
import numpy as np

pathFichierTrain = "./train"

def RessemblaceImage(chemin,w1,w2,w3,k):
    listeDistance = {}
    #Calcul de l'histogramme de l'image de test
    hist1 = couleur.Histogramme(chemin)
    hist1 = np.asanyarray(hist1)
    
    #Calcul des moements
    gris = moment.Gris(chemin)
    moment_test = moment.momentHu(gris)
    
    #coocurence
    param = np.zeros(5)
    image_gris = cooccurence.Gris(chemin)
    MatCoo = cooccurence.MatCooccurence(image_gris)
    energie,contraste,dissimilarite,homogeneite,correlation = cooccurence.ParamCooccurence(MatCoo)
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
    
    f = open((pathFichierTrain+"/histogramme"+".txt"),"rb")
    list_hist = couleur.unpickle_hist(f)
    
    f1 = open((pathFichierTrain+"/moment"+".txt"),"rb")
    list_moment = couleur.unpickle_hist(f1)
    
    f2 = open((pathFichierTrain+"/cooccurence"+".txt"),"rb")
    list_co = couleur.unpickle_hist(f2)
    
    for key, valeur in list_hist.items():
        d = d_moment = d_hist = d_co = 0
        if key in list_moment:
            d_moment =  moment.CalculDistance(moment_test,list_moment[key])
        if key in  list_co:
            d_co = cooccurence.CalculDistance(param,list_co[key])
        valeur = np.asanyarray(valeur)
        d_hist = couleur.CalculDistance(valeur,hist1)
        
        d = w1*d_hist +w2*d_co+ w3*d_moment
        listeDistance[d] = key
    listeDistances = sorted(listeDistance.items(), key=lambda t: t[0])
    f.close
    f1.close
    f2.close
    return(listeDistances[:k])
def menu():
    print("********************************************************************")
    print("* ===================> Auteur: Nicolas & Olivier  <==========================*")
    print("*    1-Similarité en utilisant la couleur.(Histogramme)            *")
    print("*    2-Similarité en utilisant la texture(matrice de cooccurrence) *")
    print("*    3-Similarité en utilisant la forme(Moment de Hu)              *")
    print("*    4-Fonction gloabale de similarité entre 2 images              *")
    print("*                                                                  *")
    print("********************************************************************")
    
def main():
    menu()
    choix = int(input("    Votre choix: "))
    k = int(input("    Nombre d'images a trouvé: " ))
    chemin = input("    Entrer le lien de l'image:"  )
    
    if choix == 1:
        listeImage = couleur.RessemblaceImage(chemin,k)
    
    if choix == 2:
        listeImage = cooccurence.Ressemblance(chemin,k)
    
    if choix ==3:
        listeImage = moment.RessemblaceImage(chemin,k)
        
    if choix == 4:
        w1 = float(input("Entrer w1: "))
        w2 = float(input("Entrer w2: "))
        w3 = float(input("Entrer w3: "))
        listeImage = RessemblaceImage(chemin,w1,w2,w3,k)

    chemin = chemin.split("/")
    classe = (chemin[len(chemin)-1]).split("_")
    classe = classe[0]
    print("                                                        ")
    
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