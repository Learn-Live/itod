"""Data-driven structure:
    1. Data module: focus on data, such as, feature extraction, and standardization
    2. Algorithms: focus on algorithm, such as, parameter tuning
    3. Plot: focus on data visualization. keep in mind that "data might need to visualize at anytime"

    Should always consider how to decouple each two modules
    "Simple is better than complexity"
"""
from examples._config import *
from examples.clean import remove_file

from treelib import Tree

from examples.base_tree import save_data, save2xlsx
from examples.data_tree import DATA_TREE
from examples.alg_tree import call_alg
from examples.result_tree import RESULT_TREE


def main():
    """
        1. data tree
        2. alg tree
        3. plot tree
    """

    # 1. get data
    dir_in = 'output_data_PCA'
    dir_out = 'output_data_PCA'
    file_data_tree = f'{dir_out}/data_tree.dat'
    lg(file_data_tree)
    overwrite = True
    if pth.exists(file_data_tree) and overwrite: os.remove(file_data_tree)
    if pth.exists(file_data_tree):
        with open(file_data_tree, 'rb') as f:
            dt = pickle.load(f)
    else:
        dt = DATA_TREE()
        dt.create(dir_in=dir_in, dir_out=dir_out)
        dt.save_tree(dt.data_tree, output_file=file_data_tree + '.txt')  # only save tree
        dt.dump_data(dt, output_file=file_data_tree)  # save dt object
    dt.show_tree(dt.data_tree, show_type='TAG')

    rt = RESULT_TREE(dt)
    # Use depth-first traversal (DFT) to obtain datasets (located on leaves of the current data_tree)
    for i, (nid_data) in enumerate(dt.data_tree.expand_tree(mode=Tree.DEPTH, sorting=False)):
        node = dt.data_tree[nid_data]
        if node.is_leaf():
            lg(f"data_node.identifier: {node.identifier}")
            try:
                # call alg and get results on the given data
                alg = call_alg(node, model=['GMM', ])
                rt.merge(nid_data, alg.data_tree, deep=True)  # new tree's root is not be pasted
                rt.show_tree(rt.data_tree)
            except Exception as e:
                le(f'{i}, {e}')
                continue
    rt.show_tree(rt.data_tree)
    result_file = f'{dir_out}/results_tree'
    rt.save_tree(rt.data_tree, output_file=result_file + '.txt')
    rt.dump_data(rt, output_file=result_file + '.dat')

    save_data(rt.data_tree, output_file=result_file + '.csv', type='auc')
    # format results: (path, x_train, x_test, gs=False, itod_kjl=False, iat, size, iat+size, ...)
    save2xlsx(rt.data_tree, output_file=result_file + '-all.csv', type='auc')


if __name__ == '__main__':
    # dir_out ='output_data_PCA'
    # with open(f'{dir_out}/results_tree.dat', 'rb') as f:
    #     rt = pickle.load(f)
    # print()
    # save2xlsx(rt, output_file=f'{dir_out}/results-all.csv', type='auc')
    remove_file(dir_in='.log', dura=3 * (24 * 60 * 60))
    main()
