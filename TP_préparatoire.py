##Annexe :TP préparatoire de Physique Quantique
from math import *
from scipy.integrate import *
import matplotlib.pyplot as plt
import numpy as np

##Partie A

def voh(x,R):
    "calcul le potentiel harmonique normalisé et adimensioné"
    return np.pi**2/4*R**2*(x-1/2)**2

#graphe des potentiels voh
X1 = np.linspace(-0.5,1.5,400)
X2=np.linspace(0,1,400)
V = np.zeros(X1.size)
V[100:300] = 160.0
Y1 = [voh(x,5) for x in X2]
Y2 = [voh(x,10) for x in X2]
Y3 = [voh(x,24) for x in X2]
plt.figure(1)
plt.plot(X1,V,label='Vinf')
plt.plot(X2,Y1,label='R = 5')
plt.plot(X2,Y2,label='R = 10')
plt.plot(X2,Y3,label='R = 24')
plt.xlabel("x")
plt.ylabel("V")
plt.xlim(-0.5, 1.5)
plt.ylim(0, 150)
plt.legend()
plt.grid()


##Partie B
##Question 1

R=24

def h(n,m) :
    "fonction qui calcul les coefficients de l'hamiltonien normalisé et adimensionés, prend en argument la ligne n et la colonne m"
    def Integrande(x):
        return np.sin(np.pi*(n)*x)*voh(x,R)*np.sin(np.pi*(m)*x)
    coef, eps= quad(Integrande,0,1)
    if n==m:
        coef+= n**2/2
    return 2*coef

def calc_H(N):
    H=np.zeros((N,N),dtype=float)
    for m in range(1,N+1):
        for n in range(1,N+1):
            H[n-1,m-1]=h(n,m)
    return H


##question 2a
N=50

H=calc_H(N)
n=[] #liste des abscices
EHO=[] # liste des energie de l'oscillateur
for i in range(N):
    n.append(i+1)
    EHO.append(R*(i-1/2))

vp,vect=np.linalg.eig(H) #on récupère les valeurs propres et vecteurs propres de H, ces valeurs propres sont toutes les valeurs possibles pour l'énergi!e


epsilon=vp.tolist() #on trie les valeurs propres, pour avoir une liste représentant epsilon(n)
epsilon.sort()

#graphe des énergies
plt.figure(2)
X=np.arange(50)
plt.plot(X,epsilon,label ='epsilon(n)',marker='o',linestyle='')
plt.plot(X,EHO, label ='Eho',marker='*',linestyle='')
plt.xlabel("n")
plt.ylabel("E")
plt.legend()
plt.grid()


##Question 2c
E_Q=[]# liste des énergies quadratiques
C=0
for k in range(N):
     E_Q.append((k+1)**2+C)
for k in range(N):
    C+=abs((k+1)**2-epsilon[k])/N
print(C)
for k in range(N):
     E_Q[k]+=C

plt.figure(0)
X=np.arange(50)
plt.plot(X,epsilon,label ='epsilon(n)',marker='o',linestyle='')
plt.plot(X,E_Q, label ='E_Quadratique',marker='*',linestyle='')
plt.xlabel("n")
plt.ylabel("E")
plt.legend()
plt.grid()


##Question 3
a=1

#calculons le produit scalaire <x|psi>
def phiHarm(x,n,a):
    "retourne la fonction phi_harmonique en x"
    return np.square(2/a)*np.sin(np.pi*(n+1)*x/a)

def prodScal(x,N,a,vectPropre,s):
    "calcul de <x|psi>"
    ps=0
    for n in range(N):
        ps+=phiHarm(x,n,a)*vectPropre[n]*s
    return ps

def vecteurpropre(vp,vect):
    "fontion qui renvoie les vecteurs propres correspondants aux energies 0,1 et 2"
    vect = np.transpose(vect)
    index = vp.argsort()
    return vp[index], vect[index]


X=np.linspace(0,a,500)

#Calculons les fonctions Psi0,Psi1etPsi2 pour differents N
N=3
H=calc_H(N)
vp,vect=np.linalg.eig(H)
vp,vect =vecteurpropre(vp,vect)

Psi0_3=[prodScal(x,N,a,vect[0],-1) for x in X]
Psi1_3=[prodScal(x,N,a,vect[1],-1) for x in X]
Psi2_3=[prodScal(x,N,a,vect[2],-1) for x in X]

N=5
H=calc_H(N)
vp,vect=np.linalg.eig(H)
vp,vect =vecteurpropre(vp,vect)
Psi0_5=[prodScal(x,N,a,vect[0],1) for x in X]
Psi1_5=[prodScal(x,N,a,vect[1],-1) for x in X]
Psi2_5=[prodScal(x,N,a,vect[2],-1) for x in X]

N=8
H=calc_H(N)
vp,vect=np.linalg.eig(H)
vp,vect =vecteurpropre(vp,vect)
Psi0_8=[prodScal(x,N,a,vect[0],-1) for x in X]
Psi1_8=[prodScal(x,N,a,vect[1],1) for x in X]
Psi2_8=[prodScal(x,N,a,vect[2],1) for x in X]

N=15
H=calc_H(N)
vp,vect=np.linalg.eig(H)
vp,vect =vecteurpropre(vp,vect)
Psi0_15=[prodScal(x,N,a,vect[0],-1) for x in X]
Psi1_15=[prodScal(x,N,a,vect[1],1) for x in X]
Psi2_15=[prodScal(x,N,a,vect[2],-1) for x in X]

#calculons les psi_theoriques:
PsiT0=[(np.pi/(2*a**2)*R)**0.25*np.exp(-np.pi**2/4*R*(x/a-0.5)**2) for x in X]
PsiT1=[(np.pi**5/(2*a**2)*R**3)**0.25*(x/a-0.5)*np.exp(-np.pi**2/4*R*(x/a-0.5)**2) for x in X]
PsiT2=[(np.pi/(8*a**2)*R)**0.25*(np.pi**2*R*(x/a-0.5)**2-1)*np.exp(-np.pi**2/4*R*(x/a-0.5)**2) for x in X]

#traçage des graphes
#psi0
plt.figure(3)
plt.plot(X,PsiT0,label ='Psi0_OH')
plt.plot(X,Psi0_3, label ='N=3')
plt.plot(X,Psi0_5, label ='N=5')
plt.plot(X,Psi0_8, label ='N=8')
plt.plot(X,Psi0_15, label ='N=15')
plt.xlabel("x")
plt.ylabel("Psi0(x)")
plt.legend()
plt.grid()

#psi1
plt.figure(4)
plt.plot(X,PsiT1,label ='Psi1_OH')
plt.plot(X,Psi1_3, label ='N=3')
plt.plot(X,Psi1_5, label ='N=5')
plt.plot(X,Psi1_8, label ='N=8')
plt.plot(X,Psi1_15, label ='N=15')
plt.xlabel("x")
plt.ylabel("Psi1(x)")
plt.legend()
plt.grid()
#psi2
plt.figure(5)
plt.plot(X,PsiT2,label ='Psi2_OH')
plt.plot(X,Psi2_3, label ='N=3')
plt.plot(X,Psi2_5, label ='N=5')
plt.plot(X,Psi2_8, label ='N=8')
plt.plot(X,Psi2_15, label ='N=15')
plt.xlabel("x")
plt.ylabel("Psi2(x)")
plt.legend()
plt.grid()

plt.show()