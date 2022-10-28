#Developed by Javier Bernabé García
#9-5-22
import warnings
from KNN import *
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.exceptions import *

conjuntoDeEntrenamiento, conjuntoDePrueba = [], []
atributosEntrenamiento, atributosPrueba = [], []
clases = 0
numeroDeAtributos=0

def lecturaDeConjuntoDeEntrenamiento(archivo = "sb1-T.txt"):
    with open(archivo) as f:#Leemos el conjunto de entranamiento
        global atributosEntrenamiento, numeroDeAtributos, atributosEntrenamiento
        numeroDeElementosEntrenanmiento = int(f.readline())
        numeroDeAtributosEntrenamiento = int(f.readline())
        numeroDeAtributos = numeroDeAtributosEntrenamiento
        atributosEntrenamiento = f.readline()[:-1]#Borramos los saltos de linea
        atributosEntrenamiento = list(map(lambda i: int(i), atributosEntrenamiento.split(",")))
        global clases
        clases = atributosEntrenamiento[-1]
        for line in f:
            if line.split():#En caso de encontrar espacios en blanco
                instancia = line[:-1]
                conjuntoDeEntrenamiento.append(list(map(lambda i: float(i), instancia.split(","))))
        f.close()

def lecturaDeConjuntoDePrueba(archivo = "sb1-P.txt"):
    with open(archivo) as f:#Leemos el conjunto de prueba
        global atributosPrueba
        numeroDeElementosPrueba = int(f.readline())
        numeroDeAtributosPrueba = int(f.readline())
        atributosPrueba = f.readline()[:-1]#Borramos los saltos de linea
        atributosPrueba = list(map(lambda i: int(i), atributosPrueba.split(",")))

        for line in f:
            if line.split():#En caso de encontrar espacios en blanco
                instancia = line[:-1]
                conjuntoDePrueba.append(list(map(lambda i: float(i), instancia.split(","))))
        f.close()

def imprimirMatrizConfusion(y_train_pred,y_train):
    classes = "auto"
    cf = confusion_matrix(y_train,y_train_pred)
    sns.heatmap(cf,annot=True,yticklabels=classes,xticklabels=classes,cmap='Blues', fmt='g')
    plt.tight_layout()
    plt.show()

def ArbolClasificacion_KNN():
    global conjuntoDeEntrenamiento, conjuntoDePrueba, atributosEntrenamiento, knn, k

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning,module="sklearn")

        #Se extraen los datos en pandas para que puedan ser usados por sklearn
        dataTrain = pd.DataFrame(conjuntoDeEntrenamiento)
        dataTest = pd.DataFrame(conjuntoDePrueba)
        x_train = dataTrain.iloc[:,:-1].values
        y_train = dataTrain.iloc[:,-1].values
        x_test = dataTest.iloc[:,:-1].values
        y_test = dataTest.iloc[:,-1].values

        #Se imprime la matriz de confusion del knn con el fit anterior el cual usa todos los atributos
        imprimirMatrizConfusion(knn.pred,y_test)
        
        #Arbol podado
        print("\nArbol de Clasificacion podado\n")
        tree = DecisionTreeClassifier()
        params = {'max_depth': [4,6,8,10],
                        'min_samples_split': [2,3,4,5],
                        'min_samples_leaf': [1,2,3,4]}
        gscv = GridSearchCV(estimator=tree,param_grid=params)
        gscv.fit(x_train,y_train)
        model = gscv.best_estimator_
        model.fit(x_train,y_train)
        y_test_pred2 = model.predict(x_test)
        print(f'Score con el conjunto de prueba: {accuracy_score(y_test_pred2,y_test)}')
        #Se imprime el arbol
        atributosDelArbol = model.tree_.feature
        atributosDelArbol = set(list(filter(lambda x: x>=0,atributosDelArbol)))#Se seleccionan los atributos usados en el arbol
        print("Atributos usados ",atributosDelArbol)
        atributosDelArbol = list(atributosDelArbol)
        atributosDelArbol.append(len(knn.atributos)-1)
        plt.figure(figsize=(19,12))
        plot_tree(decision_tree=model, filled = True)
        plt.show()
        
        #Se crea un nuevo subconjunto
        nuevosAtributos = []
        for idx in atributosDelArbol:#Se actualizan los atributos a usar
            nuevosAtributos.append(knn.atributos[idx])
        #Se crea un nuevo objeto knn
        knn2 = KNN(k,nuevosAtributos)
        #Se obtienen los subconjuntos de entrenamiento y de prueba
        train = (dataTrain.iloc[:,list(atributosDelArbol)]).values.tolist()
        test = (dataTest.iloc[:,list(atributosDelArbol)]).values.tolist()
        #Se entrena el clasificador
        knn2.fit(train, test)
        #Se imprime la matriz de confusion
        imprimirMatrizConfusion(knn2.pred,y_test)

lecturaDeConjuntoDeEntrenamiento()
lecturaDeConjuntoDePrueba()
print("Conjunto de Prueba y Entrenamiento leidos\n")
k=int(input("Introduce el numero de vecinos k: "))
knn = KNN(k, atributosEntrenamiento)#Se crea el objeto que realiza el knn
if(len(atributosEntrenamiento)==len(atributosPrueba)):#Que coincidan el conjunto de Training y Test
        knn.fit(conjuntoDeEntrenamiento,conjuntoDePrueba)#Se realiza el knn
        ArbolClasificacion_KNN()


