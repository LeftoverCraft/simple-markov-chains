import numpy as np
from matplotlib import pyplot
import random


class markovChain():
    # TransitionsMatrix : matrice carrée de taille n*n contenant les probabilités de transitions entre les n états
    # EmissionsMatrix : matrice de taille n*2 contenant les moyennes et écart-types d'émissions de chaque état
    # CurrentState : index de l'état en cours
    def __init__(self, transitionsMatrix, emissionsMatrix, currentState):
        if np.shape(transitionsMatrix)[0] != np.shape(transitionsMatrix)[1]:
            raise ValueError('La matrice de transition doit être carrée.')
        if emissionsMatrix.shape[0] != transitionsMatrix.shape[0]:
            raise ValueError("Les tailles des matrices de transition ({}) et d'émission ({}) ne correspondent pas.".format(transitionsMatrix.shape[0], emissionsMatrix.shape[0]))

        self.transitionMat = np.matrix(transitionsMatrix)
        self.emissionMat = np.matrix(emissionsMatrix)
        self.currentState = currentState
        self.nbStates = np.shape(self.transitionMat)[0]
        self.step = 1

        # On construit une matrice de transition par intervalles (pour tirer les probabilités à l'aide d'un random entre 0 et 1)
        self.transitionMatAsIntervals = []
        for i in range(self.nbStates):
            self.transitionMatAsIntervals.append([])
            currentStart = 0
            for j in range(self.nbStates):
                self.transitionMatAsIntervals[i].append([currentStart, currentStart + self.transitionMat[i,j]])
                currentStart += self.transitionMat[i,j]

        self.currentEmission = random.gauss(self.emissionMat[currentState, 0], self.emissionMat[currentState, 1])


    def computeOneStep(self):
        x = random.random()
        oldState = self.currentState
        # print("Currently in state " + str(self.currentState) + " ; tirage = " + str(x))
        for i in range(self.nbStates):
            if x >= self.transitionMatAsIntervals[oldState][i][0] and x < self.transitionMatAsIntervals[oldState][i][1]:
                self.currentState = i
                if oldState != self.currentState:
                    # print("Changement d'état !!! nouvel état : " + str(self.currentState))
                    self.currentEmission = random.gauss(self.emissionMat[self.currentState,0], self.emissionMat[self.currentState,1])

        return self.currentEmission

    def plotStatesProbabilities(self, horizon):
        plot_data = []
        self.v = np.zeros([self.nbStates])
        self.v[self.currentState] = 1
        for step in range(horizon):
            result = self.v * self.transitionMat ** step
            plot_data.append(np.array(result).flatten())

        # Convert the data format
        plot_data = np.array(plot_data)
        pyplot.figure(1)
        pyplot.xlabel("Steps")
        pyplot.ylabel("Probabilities")
        lines = []
        for i, shape in zip(range(3), ['x', 'h', 'H']):
            line, = pyplot.plot(plot_data[:, i], shape, label="S%i" % (i + 1))
            lines.append(line)
        pyplot.legend(handles=lines, loc=1)
        pyplot.show()

def testMarkovChains(ntimes):
    test = markovChain(np.matrix([[1439 / 1440, 1 / 1440, 0],
                                  [2 / 1440, 1437 / 1440, 1 / 1440],
                                  [2 / 1440, 0, 1438 / 1440]]), np.matrix([[100, 0], [100, 5], [100, 10]]), 0)

    test.plotStatesProbabilities(ntimes)