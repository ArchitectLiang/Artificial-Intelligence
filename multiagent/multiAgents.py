# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        if successorGameState.isWin():
            return 5000
        if successorGameState.isLose():
            return -5000
        score = currentGameState.getScore()
        if successorGameState.getNumFood() < currentGameState.getNumFood():
            score += 200
        newGhostPos = [i.getPosition() for i in newGhostStates]
        if len(newGhostPos) != 0:
            minGhostDis = min([manhattanDistance(newPos, i) for i in newGhostPos])
            if minGhostDis < 3:
                score -= 300/minGhostDis
            else:
                score -= 1/minGhostDis
        newFoodList = newFood.asList()
        minFoodDis = min([manhattanDistance(newPos, i) for i in newFoodList])
        score -= 10 * minFoodDis
        newCapsulePos = successorGameState.getCapsules()
        if len(newCapsulePos) != 0:
            minCapsuleDis = min([manhattanDistance(newPos, i) for i in newCapsulePos])
            if minCapsuleDis < 2:
                score -= 100 * minCapsuleDis
            else:
                score -= minCapsuleDis
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        maxscore = -math.inf
        for i in gameState.getLegalActions():
            succ = gameState.generateSuccessor(0, i)
            if succ.isWin() or succ.isLose():
                succ_score = self.evaluationFunction(succ)
            else:
                succ_score = self.minValue(succ, 1, self.depth)
            if succ_score > maxscore:
                maxscore = succ_score
                action = i
        return action
        util.raiseNotDefined()

    def maxValue(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = -math.inf
        actions = gameState.getLegalActions(0)
        for i in actions:
            successor = gameState.generateSuccessor(0, i)
            v = max(v, self.minValue(successor, 1, depth))
        return v

    def minValue(self, gameState, index, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = math.inf
        actions = gameState.getLegalActions(index)
        for i in actions:
            successor = gameState.generateSuccessor(index, i)
            if index == gameState.getNumAgents() - 1:
                v = min(v, self.maxValue(successor, depth - 1))
            else:
                v = min(v, self.minValue(successor, index + 1, depth))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        maxscore = -math.inf
        alpha = -math.inf
        beta = math.inf
        for i in gameState.getLegalActions():
            succ = gameState.generateSuccessor(0, i)
            if succ.isWin() or succ.isLose():
                succ_score = self.evaluationFunction(succ)
            else:
                succ_score = self.minValue(succ, 1, self.depth, alpha, beta)
            if succ_score > maxscore:
                maxscore = succ_score
                action = i
            alpha = max(maxscore, alpha)
        return action
        util.raiseNotDefined()

    def maxValue(self, gameState, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = -math.inf
        actions = gameState.getLegalActions(0)
        for i in actions:
            successor = gameState.generateSuccessor(0, i)
            v = max(v, self.minValue(successor, 1, depth, alpha, beta))
            if v > beta:
                return v
            alpha = max(v, alpha)
        return v

    def minValue(self, gameState, index, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = math.inf
        actions = gameState.getLegalActions(index)
        for i in actions:
            successor = gameState.generateSuccessor(index, i)
            if index == gameState.getNumAgents() - 1:
                v = min(v, self.maxValue(successor, depth - 1, alpha, beta))
            else:
                v = min(v, self.minValue(successor, index + 1, depth, alpha, beta))
            if v < alpha:
                return v
            beta = min(v, beta)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        maxscore = -math.inf
        for i in gameState.getLegalActions():
            succ = gameState.generateSuccessor(0, i)
            if succ.isWin() or succ.isLose():
                succ_score = self.evaluationFunction(succ)
            else:
                succ_score = self.expValue(succ, 1, self.depth)
            if succ_score > maxscore:
                maxscore = succ_score
                action = i
        return action
        util.raiseNotDefined()

    def maxValue(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = -math.inf
        actions = gameState.getLegalActions(0)
        for i in actions:
            successor = gameState.generateSuccessor(0, i)
            v = max(v, self.expValue(successor, 1, depth))
        return v

    def expValue(self, gameState, index, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)
        v = 0
        actions = gameState.getLegalActions(index)
        p = 1/len(actions)
        for i in actions:
            successor = gameState.generateSuccessor(index, i)
            if index == gameState.getNumAgents() - 1:
                v += p * self.maxValue(successor, depth - 1)
            else:
                v += p * self.expValue(successor, index + 1, depth)
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    if currentGameState.isWin():
        return 5000
    if currentGameState.isLose():
        return -5000
    score = currentGameState.getScore()
    score += 300/currentGameState.getNumFood()
    newGhostPos = [i.getPosition() for i in newGhostStates]
    if len(newGhostPos) != 0:
        minGhostDis = min([manhattanDistance(newPos, i) for i in newGhostPos])
        if minGhostDis < 3:
            score -= 300/minGhostDis
        else:
            score -= 1/minGhostDis
    newFoodList = newFood.asList()
    minFoodDis = min([manhattanDistance(newPos, i) for i in newFoodList])
    score += 10/minFoodDis
    newCapsulePos = currentGameState.getCapsules()
    score += 200/(len(newCapsulePos) + 1)
    if len(newCapsulePos) != 0:
        minCapsuleDis = min([manhattanDistance(newPos, i) for i in newCapsulePos])
        if minCapsuleDis < 1:
            score += 100/minCapsuleDis
        else:
            score += 1/minCapsuleDis
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
