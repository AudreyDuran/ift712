# -*- coding: utf-8 -*-

#####
#  DURAN Audrey
####

import numpy as np
from sklearn.linear_model import Perceptron, SGDClassifier
import matplotlib.pyplot as plt
import math


class ClassifieurLineaire:
    def __init__(self, lamb, methode):
        """
        Algorithmes de classification lineaire

        L'argument ``lamb`` est une constante pour régulariser la magnitude
        des poids w et w_0

        ``methode`` :   1 pour classification generative
                        2 pour Perceptron
                        3 pour Perceptron sklearn
        """
        self.w = np.array([1., 2.]) # paramètre aléatoire
        self.w_0 = -5.              # paramètre aléatoire
        self.lamb = lamb
        self.methode = methode

    def entrainement(self, x_train, t_train):
        """
        Entraîne deux classifieurs sur l'ensemble d'entraînement formé des
        entrées ``x_train`` (un tableau 2D Numpy) et des étiquettes de classe cibles
        ``t_train`` (un tableau 1D Numpy).

        Lorsque self.method = 1 : implémenter la classification générative de
        la section 4.2.2 du libre de Bishop. Cette méthode doit calculer les
        variables suivantes:

        - ``p`` scalaire spécifié à l'équation 4.73 du livre de Bishop.

        - ``mu_1`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.75 du livre de Bishop.

        - ``mu_2`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.76 du livre de Bishop.

        - ``sigma`` matrice de covariance (tableau Numpy 2D) de taille DxD,
                    telle que spécifiée à l'équation 4.78 du livre de Bishop,
                    mais à laquelle ``self.lamb`` doit être ADDITIONNÉ À LA
                    DIAGONALE (comme à l'équation 3.28).

        - ``self.w`` un vecteur (tableau Numpy 1D) de taille D tel que
                    spécifié à l'équation 4.66 du livre de Bishop.

        - ``self.w_0`` un scalaire, tel que spécifié à l'équation 4.67
                    du livre de Bishop.

        lorsque method = 2 : Implementer l'algorithme de descente de gradient
                        stochastique du perceptron avec 1000 iterations

        lorsque method = 3 : utiliser la librairie sklearn pour effectuer une
                        classification binaire à l'aide du perceptron

        """
        if self.methode == 1:  # Classification generative
            print('Classification generative')
            # AJOUTER CODE ICI

            nb_train = len(t_train)
            n1 = sum(t_train) #classe 1 : tn = 1
            n2 = nb_train - n1 #classe 2 : tn = 0
            p1 = n1/nb_train
            p2 = n2/nb_train

            # Calcul of mu_1 and mu_2
            mu_1 = t_train.dot(x_train) / n1
            mu_2 = (1-t_train).dot(x_train)/n2

            diff1 = x_train[t_train == 1] - mu_1
            diff2 = x_train[t_train == 0] - mu_2

            s1 = (diff1.T.dot(diff1)) / n1
            s2 = (diff2.T.dot(diff2)) / n2

            # Calcul  sigma inv matrix
            sigma = (n1/nb_train) * s1 + (n2/nb_train) * s2
            np.fill_diagonal(sigma, sigma.diagonal() + self.lamb)
            sigma_inv = np.linalg.inv(sigma)


            self.w = np.dot(sigma_inv, mu_1 - mu_2)
            self.w_0 = (- 1 / 2 * (mu_1.T).dot(sigma_inv).dot(mu_1) + 1 / 2 * (mu_2.T).dot(sigma_inv).dot(
                mu_2) + math.log(p1/p2)).squeeze()


        elif self.methode == 2:  # Perceptron + SGD, learning rate = 0.001, nb_iterations_max = 1000
            print('Perceptron')
            # AJOUTER CODE ICI
            nb_train = len(t_train)
            lr = 0.001
            i=0
            while i < 1000:
                predictions = np.array([self.prediction(x) for x in x_train])
                erreurs_pred = np.array([self.erreur(t, pred)
                 for t, pred in zip(t_train, predictions)])
                if np.sum(erreurs_pred) == 0: #si pas erreurs de pred
                    break
                else:#SGD
                    erreurs_mat = np.repeat(erreurs_pred, 2).reshape(nb_train, 2)
                    grad = np.sum(- erreurs_mat * x_train)
                    grad_w0 = np.sum(- erreurs_pred)
                    self.w = self.w - lr * grad
                    self.w_0 = self.w_0 - lr * grad_w0
                    i = i + 1

        else:  # Perceptron + SGD [sklearn] + learning rate = 0.001 (lambda?)+ penalty 'l2' voir http://scikit-learn.org/
            print('Perceptron [sklearn]')
            cl = Perceptron(penalty='l2',
                            alpha=self.lamb,
                            max_iter=1000,
                            fit_intercept=True,
                            eta0=0.001)

            #fit linear model with Stochastic Gradient Descent
            cl = cl.fit(X=x_train, y=t_train, coef_init = self.w, intercept_init = self.w_0)
            self.w = cl.coef_[0]
            self.w_0 = cl.intercept_[0]

        print('w = ', self.w, 'w_0 = ', self.w_0, '\n')

    def prediction(self, x):
        """
        Retourne la prédiction du classifieur lineaire.  Retourne 1 si x est
        devant la frontière de décision et 0 sinon.

        ``x`` est un tableau 1D Numpy

        Cette méthode suppose que la méthode ``entrainement()``
        a préalablement été appelée. Elle doit utiliser les champs ``self.w``
        et ``self.w_0`` afin de faire cette classification.
        """
        # AJOUTER CODE ICI
        pred = np.sign(self.w_0 + self.w.dot(x))
        pred = max(0, pred) #pour renvoyer 1 ou 0 et pas -1 ou 1

        return pred

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de classification, i.e.
        1. si la cible ``t`` et la prédiction ``prediction``
        sont différentes, 0. sinon.
        """
        # AJOUTER CODE ICI
        if t == prediction:
            err = 0.
        else:
            err = 1.

        return err

    def afficher_donnees_et_modele(self, x_train, t_train, x_test, t_test):
        """
        afficher les donnees et le modele

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=t_train * 100 + 20, c=t_train)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=t_test * 100 + 20, c=t_test)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')

        plt.show()

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.w_0, self.w
