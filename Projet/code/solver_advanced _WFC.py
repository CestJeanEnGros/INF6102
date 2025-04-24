import random
import sys
import copy
from collections import defaultdict
from itertools import product
from eternity_puzzle import EternityPuzzle
import heapq

GRAY = 0

NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3

BEST = 0

def solve_advanced(puzzle : EternityPuzzle):
    solver = WFCEternitySolver(puzzle)
    result = solver.solve()

    if result:
        # Ordonner les résultats comme attendu par display_solution
        ordered = [result[(x, y)] for y in range(puzzle.board_size) for x in range(puzzle.board_size)]
        puzzle.display_solution(ordered, "solution.png")
        puzzle.print_solution(ordered, "solution.txt")
        return ordered, puzzle.get_total_n_conflict(ordered)
    else:
        print("Pas de solution trouvée.")





class WFCEternitySolver:
    def __init__(self, puzzle: EternityPuzzle):
        self.puzzle = puzzle
        self.size = puzzle.board_size
        self.n = self.size ** 2
        self.pieces = puzzle.piece_list
        self.all_rotations = {p: puzzle.generate_rotation(p) for p in self.pieces}
        self.cells = [(x, y) for y in range(self.size) for x in range(self.size)]
        self.board = [[None for _ in range(self.size)] for _ in range(self.size)]

        # Initial domain: all piece/rotation combos
        self.domains = {
            (x, y): [
                (p, r)
                for p in self.pieces
                for r in puzzle.generate_rotation(p)
            ]
            for x, y in self.cells
        }

        # Min-heap based on entropy of the domains
        self.priority_queue = []
        self._update_priority_queue()

    def _update_priority_queue(self):
        """Met à jour la file de priorité (min-heap) en fonction des entropies."""
        heapq.heapify(self.priority_queue)
        for (x, y), domain in self.domains.items():
            heapq.heappush(self.priority_queue, (len(domain), (x, y)))

    def solve(self):
        return self._backtrack({}, set(), 0)

    def _backtrack(self, assignment, used_pieces, best):
        if len(assignment) == self.n:
            return assignment

        # Case avec plus faible entropie
        unassigned = [c for c in self.cells if c not in assignment]
        next_cell = min(unassigned, key=lambda c: len(self.domains[c]))

        domain_copy = list(self.domains[next_cell])
        random.shuffle(domain_copy)

        for piece, rotation in domain_copy:
            if piece in used_pieces:
                continue

            if self._is_valid(next_cell, rotation, assignment):
                new_assignment = assignment.copy()
                new_assignment[next_cell] = rotation
                new_used = used_pieces.copy()
                new_used.add(piece)

                saved_domains = copy.deepcopy(self.domains)
    
                self._propagate_constraints(next_cell, rotation)

                global BEST
                if len(assignment) > BEST:
                    BEST = len(assignment)
                    print(BEST)

                result = self._backtrack(new_assignment, new_used, best)
                if result:
                    return result

                # Backtrack
                # Backtrack
                self.domains = saved_domains
                self._update_priority_queue()


        return None

    def _is_valid(self, cell, tile, assignment):
        x, y = cell
        north, south, west, east = tile

        def get_neighbor(pos):
            return assignment.get(pos, None)

        if y == self.size - 1:
            if north != GRAY:
                return False
        elif (x, y + 1) in assignment:
            if assignment[(x, y + 1)][1] != north:
                return False

        if y == 0:
            if south != GRAY:
                return False
        elif (x, y - 1) in assignment:
            if assignment[(x, y - 1)][0] != south:
                return False

        if x == 0:
            if west != GRAY:
                return False
        elif (x - 1, y) in assignment:
            if assignment[(x - 1, y)][3] != west:
                return False

        if x == self.size - 1:
            if east != GRAY:
                return False
        elif (x + 1, y) in assignment:
            if assignment[(x + 1, y)][2] != east:
                return False

        return True

    def _propagate_constraints(self, cell, tile):
        # Contraintes locales simples : on réduit le domaine des voisins immédiats
        x, y = cell
        directions = {
            (0, 1): (NORTH, SOUTH),
            (0, -1): (SOUTH, NORTH),
            (-1, 0): (WEST, EAST),
            (1, 0): (EAST, WEST),
        }

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                continue
            neighbor = (nx, ny)
            if neighbor not in self.domains:
                continue

            edge_out = tile[directions[(dx, dy)][0]]
            edge_in_pos = directions[(dx, dy)][1]

            new_domain = []
            for piece, rot in self.domains[neighbor]:
                if rot[edge_in_pos] == edge_out:
                    new_domain.append((piece, rot))
            self.domains[neighbor] = new_domain
