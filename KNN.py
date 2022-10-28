#Developed by Javier Bernabé García
#9-5-22
from math import sqrt
import numpy as np
import operator
class KNN():
    def __init__(self, kVecinos, atributosEntrenamiento):
        self.kVecinos = kVecinos
        self.atributos = atributosEntrenamiento.copy()
        self.nominal = dict()
        self.classNominal = dict()
        self.pred = []

    def nominalCalculations(self, trainingSet):
        for i in range(self.atributos[-1]):#Diccionario con los numeros de atributos de acuerdo a la clase
            self.classNominal[i] = dict()
        for i in range(len(trainingSet[0])):
            if self.atributos[i]>0:#Si se trata de un atributo nominal
                self.nominal[i] = dict.fromkeys(range(self.atributos[i]),0)#Se crea un diccionario para cada valor de atributo
                for x in self.classNominal.keys():#Cada atributo se agrega al diccionario de cada clase y sus posibles valores
                    self.classNominal[x][i]=dict.fromkeys(range(self.atributos[i]),0)#Se crea un diccionario cada posible valor
                for j in range(len(trainingSet)):
                    self.nominal[i][trainingSet[j][i]]+=1#Se cuentan los valores encontrados de cada atributo
                    self.classNominal[trainingSet[j][-1]][i][trainingSet[j][i]]+=1

    def HVDM(self, instancia, instanciaEntrenamiento, desviacionEstandar):
        sumatoria = 0
        for i in range(len(instancia)-1):#Se itera sobre los atributos de la instancia de prueba
            if(self.atributos[i]>0):#Si es atributo nominal
                sumatoriaInterna = 0
                for clase in self.classNominal:
                    sumatoriaInterna += ((self.classNominal[clase][i][instanciaEntrenamiento[i]]/self.nominal[i][instanciaEntrenamiento[i]])-(self.classNominal[clase][i][instancia[i]]/self.nominal[i][instancia[i]]))
                sumatoria+= pow(abs(sumatoriaInterna),2)#Se obtiene el valor absoluto y se eleva al cuadrado
            else:
                sumatoria+=pow((instancia[i] - instanciaEntrenamiento[i])/(4*desviacionEstandar[i]),2)#HVDM NUMERICO
        return sqrt(sumatoria)

    def clasificar(self,instancia, trainingSet, desviacionEstandar):
        distancias = []
        for instanciaEntrenamiento in trainingSet:
            distancias.append((instanciaEntrenamiento[-1], self.HVDM(instancia, instanciaEntrenamiento, desviacionEstandar)))
        clasesOrdenadas = sorted(distancias, key=operator.itemgetter(1))#Se ordenan las distancias
        vecinosCercanos = clasesOrdenadas[:self.kVecinos]#Se obtienen los kvecinos mas cercanos
        vecinosCercanosDiccionario = {}
        for clase, dist in vecinosCercanos:
            if clase in vecinosCercanosDiccionario:
                vecinosCercanosDiccionario[clase][0]+=1
                vecinosCercanosDiccionario[clase][1]-=dist
            else:
                vecinosCercanosDiccionario[clase]=[1,-dist]
        vecinosCercanosDiccionario = sorted(vecinosCercanosDiccionario.items(), key=operator.itemgetter(1))#Se ordenan los vecinos de acuerdo a su numero de apariciones y distancia de menor a mayor
        return vecinosCercanosDiccionario[-1][0]#Devuelvo el ultimo elemento, el vecino con mas repeticiones y menor distancia

    def fit(self, trainingSet, testingSet):
        accuracy=[]
        self.nominalCalculations(trainingSet)
        desviacionEstandar = np.std(trainingSet,axis=0)
        for instancia in testingSet:#Se recorren los elementos del conjunto de prueba y se clasifican
            clase = self.clasificar(instancia, trainingSet, desviacionEstandar)
            self.pred.append(clase)
            if(clase == instancia[-1]):
                accuracy.append(1)
            else:
                accuracy.append(0)
        return print("\n######KNN######\nPorcentaje de Clasificacion de KNN: ", (accuracy.count(1)/len(accuracy))*100, "%")