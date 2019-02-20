# -*- coding: utf-8 -*-

#####
# Duran Audrey (19019081)
###

import numpy as np
import random
from sklearn import linear_model
import operator


class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """
        # AJOUTER CODE ICI

        phi_x = np.array([x**m for m in range(self.M)]).T
        return phi_x

    def recherche_hyperparametre(self, X, t):
        """
        Validation croisee de type "k-fold" pour k=10 utilisee pour trouver la meilleure valeur pour
        l'hyper-parametre self.M.

        Le resultat est mis dans la variable self.M

        X: vecteur de donnees
        t: vecteur de cibles
        """
        # AJOUTER CODE ICI

        k = 10
        X_folds = np.array_split(X, k)
        t_folds = np.array_split(t, k)

        k_to_accuracies = {}

        m_liste = [1, 2, 3, 4, 5, 6, 7, 8]

        for m in m_liste:

            self.M = m
            err = 0.0

            for ki in range(k):
                Xval = np.hstack(np.delete(X_folds, ki, axis=0))
                tval = np.hstack(np.delete(t_folds, ki, axis=0))
                self.entrainement(Xval, tval) # define a W

                tpred = np.array([self.prediction(X) for X in X_folds[ki]])
                err += sum(np.array([self.erreur(tp, ty) for tp, ty in zip(tpred, t_folds[ki])]))

            k_to_accuracies[m] = err/k

        best_M = min(k_to_accuracies.items(), key=operator.itemgetter(1))[0]
        print(best_M)
        self.M = best_M


    def entrainement(self, X, t, using_sklearn=True):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à l'entree
        x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille D+1, tel que specifie à la section 3.1.4
        du livre de Bishop.
        
        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de 
        la librairie sklearn (voir http://scikit-learn.org/stable/modules/linear_model.html)
        
        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M

        """
        #AJOUTER CODE ICI

        if self.M <= 0:
            self.recherche_hyperparametre(X, t)

        phi_x = self.fonction_base_polynomiale(X)

        if using_sklearn == True:
            reg = linear_model.Ridge(self.lamb)
            reg.fit(phi_x, t)
            self.w = reg.coef_

        if using_sklearn == False:
            self.w = np.linalg.solve(np.dot(self.lamb, np.eye(self.M, self.M)) + \
                                     np.dot(phi_x.T,phi_x), np.dot(phi_x.T, t))


    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """
        # AJOUTER CODE ICI

        phi_x = self.fonction_base_polynomiale(x)
        y_x = np.dot(phi_x.T, self.w)
        return y_x

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """
        # AJOUTER CODE ICI

        err = (t-prediction)**2
        return err #0.0
