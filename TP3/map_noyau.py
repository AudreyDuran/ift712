# -*- coding: utf-8 -*-

#####
# Audrey Duran
###

import numpy as np
import matplotlib.pyplot as plt


class MAPnoyau:
    def __init__(self, lamb=0.2, sigma_square=1.06, b=1.0, c=0.1, d=1.0, M=2, noyau='rbf'):
        """
        Classe effectuant de la segmentation de données 2D 2 classes à l'aide de la méthode à noyau.

        lamb: coefficiant de régularisation L2
        sigma_square: paramètre du noyau rbf
        b, d: paramètres du noyau sigmoidal
        M,c: paramètres du noyau polynomial
        noyau: rbf, lineaire, polynomial ou sigmoidal
        """
        self.lamb = lamb
        self.a = None
        self.sigma_square = sigma_square
        self.M = M
        self.c = c
        self.b = b
        self.d = d
        self.noyau = noyau
        self.x_train = None



    def entrainement(self, x_train, t_train):
        """
        Entraîne une méthode d'apprentissage à noyau de type Maximum a
        posteriori (MAP) avec un terme d'attache aux données de type
        "moindre carrés" et un terme de lissage quadratique (voir
        Eq.(1.67) et Eq.(6.2) du livre de Bishop).  La variable x_train
        contient les entrées (un tableau 2D Numpy, où la n-ième rangée
        correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
        où le n-ième élément correspond à la cible t_n).

        L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
        ou polynomial (spécifié par ''self.noyau'') et dont les parametres
        sont contenus dans les variables self.sigma_square, self.c, self.b, self.d
        et self.M et un poids de régularisation spécifié par ``self.lamb``.

        Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
        l'equation 6.8 du livre de Bishop et garder en mémoire les données
        d'apprentissage dans ``self.x_train``
        """
        #AJOUTER CODE ICI
        N = len(t_train)
        # x = x_train[:,0]
        # y = x_train[:,1]

        if self.noyau == 'rbf':
            # We use ||a-b|| = a**2 + 2ab - b**2
            x_norm = np.sum(x_train ** 2, axis = 1)
            K = np.exp(- ((x_norm[:, None] + x_norm[None, :] - 2 * \
             x_train.dot(x_train.T)) / 2 * self.sigma_square))

        elif self.noyau == 'lineaire':
            K = np.zeros((N, N))
            for n in range(N):
                for m in range(N):
                    K[n, m] = x_train[n].T.dot(x_train[m])

        elif self.noyau == 'polynomial':
            K = np.zeros((N, N))
            for n in range(N):
                for m in range(N):
                    K[n,m] = (x_train[n].T.dot(x_train[m]) + self.c)**self.M

        elif self.noyau == 'sigmoidal':
            K = np.zeros((N, N))
            for n in range(N):
                for m in range(N):
                    K = np.tanh(self.b * x_train[n].T.dot(x_train[m]) + self.d)

        #mettre else erreur

        self.a = (K + self.lamb * np.eye(N, N))**-1 * t_train
        self.x_train = x_train

    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.

        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée. Elle doit utiliser le champs ``self.a`` afin de calculer
        la prédiction y(x) (équation 6.9).

        NOTE : Puisque nous utilisons cette classe pour faire de la
        classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
        sinon
        """
        #AJOUTER CODE ICI
        if self.noyau == 'rbf':
            K = np.exp(-(self.x_train - x)**2 / 2 * self.sigma_square)

        elif self.noyau == 'lineaire':
            # K = x_train.T.dot(x_train)
            K = self.x_train.dot(x)

        elif self.noyau == 'polynomial':
            K = (self.x_train.dot(x) + self.c)**self.M

        elif self.noyau == 'sigmoidal':
            K = np.tanh(self.b * self.x_train.dot(x) + self.d)

        y = np.sum(K.T.dot(self.a))
        return int(y > 0.5)

    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
        # AJOUTER CODE ICI
        return (t-prediction)**2

    def validation_croisee(self, x_tab, t_tab):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
        ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
        ``self.lamb`` avec une validation croisée de type "k-fold" où k=1 avec les
        données contenues dans x_tab et t_tab.  Une fois les meilleurs hyperparamètres
        trouvés, le modèle est entraîné une dernière fois.

        SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
        """
        # AJOUTER CODE ICI
        lamb_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2]
        best_lamb = None
        best_err = 100 #initialisation a 100% erreur

        N = len(t_tab)
        indexes = np.random.choice(range(N), size = round(N/3), replace = False)
        x_fold = x_tab[indexes] #test prediction sur 1/3 des donnees
        t_fold = t_tab[indexes]

        x_train = x_tab[-indexes]
        t_train = t_tab[-indexes]

        if self.noyau == 'rbf':
            # self.sigma_square
            best_sigma_square = None

            sigma_square_values = \
             [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2]

            for sigma_square in sigma_square_values :
                self.sigma_square = sigma_square
                for lamb in lamb_values:
                    self.lamb = lamb
                    self.entrainement(x_train, t_train)

                    t_pred = np.array([self.prediction(x) for x in x_fold])
                    err =100*np.sum(self.erreur(t_fold, t_pred))/len(t_fold)
                    print(sigma_square, lamb, err)
                    if err < best_err :
                         best_err = err
                         best_lamb = lamb
                         best_sigma_square = sigma_square


            self.lamb = best_lamb
            self.sigma_square = best_sigma_square

            self.entrainement(x_tab, t_tab)


        elif self.noyau == 'lineaire':
            pass


        elif self.noyau == 'polynomial':
            # self.c et self.M
            best_c = None
            best_M = None
            c_values = [0, 1, 2, 3, 4, 5]
            M_values = [2, 3, 4, 5, 6]

        elif self.noyau == 'sigmoidal':
            #self.b et self.d
            best_b = None
            best_d = None

            b_values = [1e-5, 1e-4, 1e-3, 1e-2]
            d_values = [1e-5, 1e-4, 1e-3, 1e-2]



    def affichage(self, x_tab, t_tab):

        # Affichage
        ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
        iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
        iX, iY = np.meshgrid(ix, iy)
        x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
        contour_out = np.array([self.prediction(x) for x in x_vis])
        contour_out = contour_out.reshape(iX.shape)

        plt.contourf(iX, iY, contour_out > 0.5)
        plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')
        plt.show()
