import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    #raise NotImplementedError("you need to write this!")
    # Min --> P1, Max --> P2
    value = 0.0
    moveList = []
    moveTree = dict()

    if depth == 0:
        value = chess.lib.heuristics.evaluate(board)
        return value, moveList, moveTree
    
    min_value, max_value = float("inf"), -float("inf")
    min_move, max_move = None, None
    min_moveList, max_moveList = [], []
    
    for move in generateMoves(side, board, flags):
        newside, newboard, newflags = chess.lib.makeMove(side, board, move[0], move[1], flags, move[2])
        next_val, next_moveList, next_moveTree = minimax(newside, newboard, newflags, depth - 1)
        
        if next_val < min_value:
            min_value = next_val
            min_move = move
            min_moveList = next_moveList

        if next_val > max_value:
            max_value = next_val
            max_move = move
            max_moveList = next_moveList
        
        moveTree[encode(*move)] = next_moveTree
            
    if side:
        min_moveList = [min_move] + min_moveList
        return min_value, min_moveList, moveTree
    else:
        max_moveList = [max_move] + max_moveList
        return max_value, max_moveList, moveTree


def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    #raise NotImplementedError("you need to write this!")
    value = 0.0
    moveList = []
    moveTree = dict()

    if depth == 0:
        value = chess.lib.heuristics.evaluate(board)
        return value, moveList, moveTree
    
    min_value, max_value = float("inf"), -float("inf")
    min_move, max_move = None, None
    min_moveList, max_moveList = [], []

    for move in generateMoves(side, board, flags):
        newside, newboard, newflags = chess.lib.makeMove(side, board, move[0], move[1], flags, move[2])
        next_val, next_moveList, next_moveTree = alphabeta(newside, newboard, newflags, depth - 1, alpha, beta)

        if next_val < min_value:
            min_value = next_val
            min_move = move
            min_moveList = next_moveList

        if next_val > max_value:
            max_value = next_val
            max_move = move
            max_moveList = next_moveList
            
        if side: # Min update beta, max update alpha
            beta = min(beta, min_value)
        else:
            alpha = max(alpha, max_value)
        
        moveTree[encode(*move)] = next_moveTree

        if beta <= alpha: # Prune children if beta falls below alpha
            break
        
    if side:
        min_moveList = [min_move] + min_moveList
        return min_value, min_moveList, moveTree
    else:
        max_moveList = [max_move] + max_moveList
        return max_value, max_moveList, moveTree
    

def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    #raise NotImplementedError("you need to write this!")
