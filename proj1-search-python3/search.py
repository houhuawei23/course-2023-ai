# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


class Node:
    """
    node struct in the search tree
    """

    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent: Node = parent
        self.action = action
        self.path_cost = path_cost

    def __repr__(self):
        if self.parent:
            return f"Node({self.state}, {self.parent.state}, {self.action}, {self.path_cost})"
        else:
            return f"Node({self.state}, {None}, {None}, {self.path_cost})"


def traceBackNode(node: Node):
    actions = []
    while node.parent is not None:
        actions.append(node.action)
        node = node.parent
    actions.reverse()
    # print(f"actions: {actions}")
    return actions


def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    reached_states = set()
    frontier = util.Stack()
    frontier.push(Node(problem.getStartState(), None, None, 0))

    while not frontier.isEmpty():
        node: Node = frontier.pop()
        # print(node)
        if problem.isGoalState(node.state):
            return traceBackNode(node)
        if node.state not in reached_states:
            reached_states.add(node.state)
            for s, a, c in problem.getSuccessors(node.state):
                frontier.push(Node(s, node, a, node.path_cost + c))
    return []


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    reached_states = []
    frontier = util.Queue()
    frontier.push(Node(problem.getStartState(), None, None, 0))


    while not frontier.isEmpty():
        node: Node = frontier.pop()
        if problem.isGoalState(node.state):
            return traceBackNode(node)
        if node.state not in reached_states:
            reached_states.append(node.state)
            for s, a, c in problem.getSuccessors(node.state):
                new_node = Node(s, node, a, node.path_cost + c)
                frontier.push(new_node)
    return []


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    reached_states = set()
    frontier = util.PriorityQueue()
    frontier.push(Node(problem.getStartState(), None, None, 0), 0)

    while not frontier.isEmpty():
        node: Node = frontier.pop()
        # print(node)
        if problem.isGoalState(node.state):
            return traceBackNode(node)
        if node.state not in reached_states:
            reached_states.add(node.state)
            for s, a, c in problem.getSuccessors(node.state):
                frontier.push(Node(s, node, a, node.path_cost + c), node.path_cost + c)
    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """

    return 0


# from searchAgents import PositionSearchProblem


# def heuristic_1(state, problem: PositionSearchProblem = None):
#     return util.manhattanDistance(state, problem.goal)


def heuristic_1(state, problem=None):
    return util.manhattanDistance(state, problem.goal)


# def f(problem:SearchProblem, node: Node, h=nullHeuristic):
#     return node.path_cost + h(node.state,)


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    reached_states = set()
    frontier = util.PriorityQueue()
    node: Node = Node(problem.getStartState(), None, None, 0)
    frontier.push(node, node.path_cost + heuristic(node.state, problem))

    while not frontier.isEmpty():
        node: Node = frontier.pop()
        # print(node)
        if problem.isGoalState(node.state):
            return traceBackNode(node)
        if node.state not in reached_states:
            reached_states.add(node.state)
            for s, a, c in problem.getSuccessors(node.state):
                newnode = Node(s, node, a, node.path_cost + c)
                frontier.push(
                    newnode, newnode.path_cost + heuristic(newnode.state, problem)
                )
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
