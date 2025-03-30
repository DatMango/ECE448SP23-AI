# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

import queue

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    path_taken = []

    start_point = maze.start

    if maze.waypoints[0] == maze.start:
        path_taken.append(start_point)
        return path_taken
    
    queue = []
    explored_set = set()
    prev_pts = {}
    queue.append(start_point)
    explored_set.add(start_point)

    while queue:
        curr_pt = queue.pop(0)
        if maze.waypoints[0] == curr_pt:
            path_taken.append(curr_pt)
            while path_taken[-1] != start_point:
                path_taken.append(prev_pts[path_taken[-1]])
            path_taken.reverse()
            return path_taken
        
        curr_neigh = maze.neighbors(curr_pt[0], curr_pt[1])
        for pts in curr_neigh:
            if pts not in explored_set and pts not in queue:
                prev_pts[pts] = curr_pt
                queue.append(pts)
                explored_set.add(pts)

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    path_taken = []
    start_point = maze.start
    waypt = maze.waypoints[0]
    priority_queue = queue.PriorityQueue()
    explored_set = set()
    prev_pts = {}
    start_node = (manhattan_distance(start_point, waypt), maze.start, 0)
    explored_set.add(start_point)
    priority_queue.put(start_node)

    while priority_queue:
        curr_pt = priority_queue.get()
        if curr_pt[1] == waypt:
            path_taken.append(curr_pt)
            break
        
        curr_neighbors = maze.neighbors(curr_pt[1][0], curr_pt[1][1])
        for pts in curr_neighbors:
            if pts not in explored_set:
                prev_pts[pts] = curr_pt[1]
                priority_queue.put((manhattan_distance(pts, waypt) + curr_pt[2] + 1, pts, curr_pt[2] + 1))
                explored_set.add(pts)

    path_taken = [waypt]
    while path_taken[-1] != start_point:
        path_taken.append(prev_pts[path_taken[-1]])
    path_taken.reverse()
    
    return path_taken

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    path_taken = []
    start_point = maze.start
    waypt = maze.waypoints[0]
    priority_queue = queue.PriorityQueue()
    explored_set = set()
    prev_pts = {}
    prev_pts[(start_point, waypt, 0)] = None

    
    return []
