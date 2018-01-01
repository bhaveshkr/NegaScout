"""NegaScout/Principal Variation Search
"""
__author__ = 'Bhavesh Kumar'


import random
import sys
from copy import deepcopy
import os
import pandas as pd


class Node:
    def __init__(self, evalue, daughter=None):
        self.evalue = evalue
        if daughter is None:
            daughter = []
        self.daughter = daughter
        self.daughter_1 = []


class GameTree:
    def __init__(self, bfactor, height, approx):
        self.root_value = random.randrange(-2500, 2500)
        self.negative_root_value = -self.root_value
        self.bfactor = bfactor
        self.height = height
        self.approx = approx
        self.max_value = 10001
        self.root = Node(None)
        self.total_nodes = []
        self.static_evaluation_count = 0
        self.principle_variation = []

    def traverse_tree(self, node, height):
        if height is 0:
            return
        if isinstance(node, Node):
            if node is self.root:
                self.total_nodes.append(node)
            if len(node.daughter) > 0:
                for daughter in node.daughter:
                    self.total_nodes.append(daughter)
                    self.traverse_tree(daughter, height - 1)

    def print_by_level(self, node, height):
        for level in range(height + 1):
            self.print_level(node, level)
            print()

    def print_level(self, node, level):
        if node is None:
            return
        if level is 0:
            print(node.evalue, end='\t')
        elif level > 0:
            for daughter in node.daughter:
                self.print_level(daughter, level - 1)

    # print object description
    def print_tree_objects(self):
        print('-------------Printing tree with daughter and daughter_1(with move reordered)-------------')
        self.traverse_tree(self.root, 10)
        for node in self.total_nodes:
            if len(node.daughter) > 0:
                print('Node E value :', node.evalue)
                print('Daughter:')
            else:
                print('Leaf node value :', node.evalue)
            for i in range(len(node.daughter)):
                print(node.daughter[i].__dict__)
            if len(node.daughter) > 0:
                print('Copy of original daughter')
            for i in range(len(node.daughter_1)):
                print(node.daughter_1[i].__dict__)
            print('---------------------------')

    def construct_tree(self, node, height, t_value):
        if height is 0:
            return
        is_root = False
        if isinstance(node, Node):
            if node is self.root:
                is_root = True
                node.evalue = t_value + self.get_delta()
            if len(node.daughter) > 0:
                for daughter in node.daughter:
                    self.construct_tree(daughter, height, t_value)
            else:
                if node.evalue is self.max_value - 1:  # node with E value 10,000 should not have child
                    return

                # change of branching factor
                if is_root is True:
                    predicted_branching_factor = self.bfactor
                else:
                    random_no = random.randrange(100)
                    if 90 < random_no <= 95:
                        predicted_branching_factor = self.bfactor + 1
                    elif 95 < random_no <= 100:
                        predicted_branching_factor = self.bfactor - 1
                    else:
                        predicted_branching_factor = self.bfactor

                random_daughter_index = random.randrange(predicted_branching_factor)
                for bf_index in range(predicted_branching_factor):
                    if random_daughter_index is bf_index:
                        if height is 1:  # leaf node case
                            node.daughter.append(Node(t_value))
                        else:
                            tmp_node = Node(-t_value + self.get_delta())
                            node.daughter.append(tmp_node)
                            node.daughter_1.append(tmp_node)
                        self.construct_tree(node.daughter[bf_index], height - 1, -t_value)
                    else:
                        random_t_value = random.randrange(t_value, self.max_value)  # generate random T value
                        if height is 1:  # leaf node case
                            tmp_node = Node(random_t_value)
                            node.daughter.append(tmp_node)
                            node.daughter_1.append(tmp_node)
                        else:
                            tmp_node = Node(random_t_value + self.get_delta())
                            node.daughter.append(tmp_node)
                            node.daughter_1.append(tmp_node)
                        self.construct_tree(node.daughter[bf_index], height - 1, -random_t_value)

    def get_delta(self):
        if self.approx is 0:
            return self.approx
        return random.randrange(-self.approx, self.approx)

    # Negamax style alpha-beta with iterative deepening implementation
    def negamax_alpha_beta_itr_deepening(self, node, achievable, hope, height, reordering, depth=0):
        if len(node.daughter) is 0 or depth is height:
            return node.evalue

        # Move reordering
        if reordering is True:
            best_daughter = self.get_best_daughter(node.daughter)
            if node is self.root:
                self.principle_variation.append(deepcopy(node))
            if len(self.principle_variation) <= height:
                self.principle_variation.append(deepcopy(best_daughter))
            if node.daughter.index(best_daughter) is not 0:
                node.daughter.remove(best_daughter)
                node.daughter.insert(0, deepcopy(best_daughter))

        for daughter_node in node.daughter:
            temp = -self.negamax_alpha_beta_itr_deepening(daughter_node, -hope, -achievable,
                                                                                  height, reordering, depth + 1)
            if temp >= hope:
                return temp
            if len(daughter_node.daughter) is 0:
                self.static_evaluation_count += 1
            achievable = max(temp, achievable)
        return achievable

    def get_best_daughter(self, daughters):
        return min(daughters, key=lambda node: node.evalue)

    # Negascout / PVS algorithm with iterative deepening
    def pvs_iterative_deepening(self, node, alpha, beta, depth, reordering):
        if len(node.daughter) is 0 or depth is 0:
            return node.evalue

        # Move reordering
        if reordering is True:
            best_daughter = self.get_best_daughter(node.daughter)
            if node.daughter.index(best_daughter) is not 0:
                node.daughter.remove(best_daughter)
                node.daughter.insert(0, best_daughter)

        for index, daughter in enumerate(node.daughter):
            if index is 0:
                score = -self.pvs_iterative_deepening(daughter, -beta, -alpha, depth-1, reordering)
            else:
                score = -self.pvs_iterative_deepening(daughter, -alpha-1, -alpha, depth-1, reordering)
                if alpha < score < beta:
                    score = -self.pvs_iterative_deepening(daughter, -beta, -score, depth-1, reordering)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
            if len(daughter.daughter) is 0:
                self.static_evaluation_count += 1
        return alpha


def main():
    random.seed(1000)
    bfactor, height, approx = 2, 3, 3
    t_value = random.randrange(-2500, 2500)
    tree = GameTree(bfactor, height, approx)
    root = tree.root
    tree.construct_tree(root, tree.height, t_value)
    tree.print_by_level(root, height)
    reordering = True

    print('-------------Negamax style - alpha-beta -------------------')
    print('Game Result : ', tree.negamax_alpha_beta_itr_deepening(root, -sys.maxsize, sys.maxsize, height, reordering))
    print('Static evaluation count : ', tree.static_evaluation_count)
    print('Principle Variation : ')
    for item in tree.principle_variation:
        print(item.evalue, end=' => ')
    print('[]')

    # reinitialize tree
    tree.root.daughter = deepcopy(tree.root.daughter_1)

    print('-------------Negascout/PVS -------------------')
    tree.static_evaluation_count = 0 # reset evaluation counter
    print('Game Result : ', tree.pvs_iterative_deepening(root, -sys.maxsize, sys.maxsize, height, reordering))
    print('Static evaluation count : ', tree.static_evaluation_count)


def run_experiment(pv_reordering=False):
    random.seed(1000)
    for height in range(4, 7, 1):
        for branching_factor in range(3, 22, 3):
            for approx in range(0, 301, 50):
                for no_of_tree in range(1, 26, 1):
                    t_value = random.randrange(-2500, 2500)
                    tree = GameTree(branching_factor, height, approx)
                    root = tree.root
                    tree.construct_tree(root, tree.height, t_value)

                    # Run experiment for Negamax
                    negamax_score = tree.negamax_alpha_beta_itr_deepening(root, -sys.maxsize, sys.maxsize, height, pv_reordering)
                    negamax_static_evaluation = tree.static_evaluation_count

                    principle_variation = []
                    if pv_reordering is True:
                        for item in tree.principle_variation:
                            principle_variation.append(item.evalue)

                    df = pd.DataFrame({'Height': [height], 'BF': [branching_factor], 'Approx': [approx], 'Algorithm_Name': ['Negamax'],
                                      'Result':[negamax_score], 'Static Evaluation': [negamax_static_evaluation], 'PV': [principle_variation], 'Reordering':[pv_reordering]})


                    # Reinitialize tree for Principle variation search algorithm
                    tree.root.daughter = deepcopy(tree.root.daughter_1)

                    # Run experiment for Negascout/PVS
                    tree.static_evaluation_count = 0  # reset evaluation counter
                    negascout_score = tree.pvs_iterative_deepening(root, -sys.maxsize, sys.maxsize, height, pv_reordering)
                    negascout_static_evaluation = tree.static_evaluation_count

                    df1 = pd.DataFrame({'Height': [height], 'BF': [branching_factor], 'Approx': [approx], 'Algorithm_Name': ['Negascout'],
                                        'Result': [negascout_score], 'Static Evaluation': [negascout_static_evaluation], 'PV': [principle_variation], 'Reordering': [pv_reordering]})

                    df.to_csv(excel_path, index=False, header=False, mode='a')
                    df1.to_csv(excel_path, index=False, header=False, mode='a')

                    global experiment_counter
                    experiment_counter += 1
                    print(experiment_counter)
    run_experiment(True)  # Now run this experiment with PV reordering


if __name__ == '__main__':

    '''
    Use main() to run general configuration
    '''
    main()

    '''
    Uncomment code below to run experiment arrangement
    '''
    # Windows specific
    excel_path = None
    #path = os.path.join(os.environ["HOMEPATH"], "Desktop\\experiment.log")
    #excel_path = os.path.join(os.environ["HOMEPATH"], "Desktop\\experiment.csv")

    # Linux/Mac environment
    # excel_path = '/home/Desktop/experiment.csv'
    # experiment_counter = 0
    # run_experiment()
    # print('completed')

