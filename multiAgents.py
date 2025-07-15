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
from json.encoder import INFINITY

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
        from util import manhattanDistance
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
        newFood = successorGameState.getFood().asList()
        currentFood = currentGameState.getFood().asList()
        score = 0
        closest = 0
        distances = []

        if len(newFood) < len(currentFood):
            score += 50

        elif newFood:
            distances = [manhattanDistance(newPos, food) for food in newFood]
            closest = min(distances)
            score += 10 / (closest + 1)
        else:
            score += 100

        for ghost in newGhostStates:
            if manhattanDistance(newPos, ghost.getPosition()) <= 1:
                score -= 1000
            elif manhattanDistance(newPos, ghost.getPosition()) <= 3:
                score -= 500
            #return successorGameState.getScore()
        #currentDirection = currentGameState.getPacmanState().configuration.direction
        #reverseDirection = Directions.REVERSE.get(currentDirection, None)

        #if action == currentDirection:
         #   score += 1
        #elif action == reverseDirection:
           # score -= .5

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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
    def minimax(self, gameState : GameState, agentIndex , depth):
        #if base case, agentindex == 0:
            #return max of generate successors values
        #else
            #
    #           for each successors
    #          return min (minimax(successor game state, agent index - 1, depth - 1))
        max_depth = self.depth
        if gameState.isWin() or gameState.isLose() or depth == max_depth:
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            best_val = -10000000000
            for action in gameState.getLegalActions(0):
                successor = gameState.generateSuccessor(0, action)
                val = self.minimax(successor, agentIndex + 1, depth)
                best_val = max(best_val, val)
            return best_val
        else:
            best_valz = 10000000000
            for action in gameState.getLegalActions(agentIndex):
                success = gameState.generateSuccessor(agentIndex, action)
                if agentIndex == gameState.getNumAgents() - 1:
                    Nextdepth = depth + 1
                    NextagentIndex = 0
                else:
                    NextagentIndex = agentIndex + 1
                    Nextdepth = depth
                valz = self.minimax(success, NextagentIndex, Nextdepth)
                best_valz = min(best_valz, valz)
            return best_valz

    def getAction(self, gameState: GameState):
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
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.minimax(successor, 1, 0)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the best action for Pacman using alpha-beta pruning.
        """
        alpha = float('-inf')
        beta = float('inf')
        bestScore = float('-inf')
        bestAction = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.alphabeta(successor, 1, 0, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction

    def alphabeta(self, gameState: GameState, agentIndex, currDepth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState)

        numAgents = gameState.getNumAgents()

        if agentIndex == 0:  # Pacman (maximizer)
            return self.max_value(gameState, agentIndex, currDepth, alpha, beta)
        else:  # Ghosts (minimizers)
            return self.min_value(gameState, agentIndex, currDepth, alpha, beta)

    def max_value(self, gameState: GameState, agentIndex, currDepth, alpha, beta):
        v = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            score = self.alphabeta(successor, agentIndex + 1, currDepth, alpha, beta)
            v = max(v, score)
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, gameState: GameState, agentIndex, currDepth, alpha, beta):
        v = float('inf')
        numAgents = gameState.getNumAgents()
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = currDepth + 1 if nextAgent == 0 else currDepth
            score = self.alphabeta(successor, nextAgent, nextDepth, alpha, beta)
            v = min(v, score)
            if v < alpha:
                return v
            beta = min(beta, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.value(gameState, self.index, 0)[1]

    def value(self, gameState: GameState, agent, currDepth):
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return (self.evaluationFunction(gameState), None)
        if agent == 0:
            return self.max_value(gameState, agent, currDepth)
        else:
            return self.expectimax_value(gameState, agent, currDepth)

    def max_value(self, gameState: GameState, agent, currDepth):
        v = -float('inf')
        actions = gameState.getLegalActions(agent)
        max_action = None
        for action in actions:
            successor = gameState.generateSuccessor(agent, action)
            successor_value = self.value(successor, agent + 1, currDepth)[0]
            v = max(v, successor_value)
            if v == successor_value:
                max_action = action
        return (v, max_action)

    def expectimax_value(self, gameState: GameState, agent, currDepth):
        actions = gameState.getLegalActions(agent)
        numAgents = gameState.getNumAgents()
        value = 0
        num_actions = 0
        for action in actions:
            num_actions += 1
            nextAgent = (agent + 1) % numAgents
            nextDepth = currDepth + 1 if nextAgent == 0 else currDepth
            successor = gameState.generateSuccessor(agent, action)
            successor_value = self.value(successor, nextAgent, nextDepth)[0]
            value += successor_value
        return (value/num_actions, None)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
