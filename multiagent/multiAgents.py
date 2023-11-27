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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        # print(legalMoves, scores)
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

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        nearest_scared_ghost_dis = 1e9
        nearset_normal_ghost_dis = 1e9

        for ghostState in newGhostStates:
            mdis = manhattanDistance(newPos, ghostState.getPosition())
            scared_time = ghostState.scaredTimer
            if scared_time > 0:
                nearest_scared_ghost_dis = min(nearest_scared_ghost_dis, mdis)
                # + 1 / (nearest_scared_ghost_dis + 1)
            else:
                nearset_normal_ghost_dis = min(nearset_normal_ghost_dis, mdis)

        if nearset_normal_ghost_dis <= 1:  # 紧急避险
            return -100

        nearest_food_dis = 1e9
        foods = newFood.asList()
        for food in foods:
            mdis = manhattanDistance(newPos, food)
            nearest_food_dis = min(nearest_food_dis, mdis)

        stateScore = successorGameState.getScore()

        return (
            stateScore + 1 / (nearest_food_dis + 1) + 1 / (nearest_scared_ghost_dis + 1)
        )


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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


from pacman import GameState
from game import AgentState
from game import Grid


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

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

        pacman_state: AgentState = gameState.getPacmanState()
        # pacman_state.getPosition()
        # pacman_state.getDirection()

        pacman_legal_actions: list[str] = gameState.getLegalPacmanActions()
        pacman_successor: GameState = gameState.generatePacmanSuccessor(
            pacman_legal_actions[0]
        )

        # successor = gameState.generateSuccessor(0, legal_actions[0])
        num_agents: int = gameState.getNumAgents()
        ghost_states: list[AgentState] = gameState.getGhostStates()
        # .getGhostState(agentIndex)
        # ghost_states[0].getPosition()
        # ghost_states[0].getDirection()

        capsules: list[tuple] = gameState.getCapsules()
        food_grid: Grid = gameState.getFood()  # if food_grid[x][y] == True: ...
        walls_grid: Grid = gameState.getWalls()  # if walls_grid[x][y] == True: ...
        # .hasFood(x, y) .hasWall(x, y)
        # .isWin(), .isLose()
        """
        "*** YOUR CODE HERE ***"

        value, action = self.minimax_search(gameState, 0, self.depth)
        return action

    def minimax_search(self, gameState: GameState, agentIndex: int, depth: int):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP
        elif agentIndex == 0:  # pacman
            return self.max_value(gameState, agentIndex, depth)
        else:  # ghost
            return self.min_value(gameState, agentIndex, depth)

    def max_value(self, gameState: GameState, agentIndex: int, depth: int):
        max = -1e9
        action = Directions.STOP

        next_agent_index = agentIndex + 1
        next_depth = depth
        for legal_action in gameState.getLegalActions(agentIndex):
            curGameState = gameState.generateSuccessor(agentIndex, legal_action)
            v, a = self.minimax_search(curGameState, next_agent_index, next_depth)
            if v > max:
                max = v
                action = legal_action
        return max, action

    def min_value(self, gameState: GameState, agentIndex: int, depth: int):
        min = 1e9
        action = Directions.STOP

        if agentIndex == gameState.getNumAgents() - 1:
            # last ghost
            next_agent_index = 0
            next_depth = depth - 1
        else:
            next_agent_index = agentIndex + 1
            next_depth = depth
            
        for legal_action in gameState.getLegalActions(agentIndex):
            curGameState = gameState.generateSuccessor(agentIndex, legal_action)
            v, a = self.minimax_search(curGameState, next_agent_index, next_depth)
            if v < min:
                min = v
                action = legal_action
        return min, action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
