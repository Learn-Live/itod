# """Build data tree
#
# """
# from examples._config import *
# import os.path as pth
# from treelib import Tree, Node
# import textwrap
# import numpy as np
# from numpy import genfromtxt
#
#
# def extract_data(normal_pth, abnormal_pth, meta_data={}):
#     """Get normal and abnormal data from txt, and store them into a dict
#     # NORMAL(inliers): 0, ABNORMAL(outliers): 1
#     Returns
#     -------
#         data: dict
#
#     """
#     NORMAL = 0  # Use 0 to label normal data
#     ABNORMAL = 1  # Use 1 to label abnormal data
#
#     # Normal and abnormal are the same size in the test set
#     if meta_data['train_size'] <= 0:
#         n_normal = -1
#         n_abnormal = -1
#     else:
#         n_abnormal = int(meta_data['test_size'] // 2)
#         n_normal = meta_data['train_size'] + n_abnormal
#     start = meta_data['idxs_feat'][0]
#     end = meta_data['idxs_feat'][1]
#
#     def _label_and_combine_data(X, size=-1, data_type='normal'):
#         if size == -1:
#             size = X.shape[0]
#         idx = np.random.randint(0, high=X.shape[0], size=size)
#         X = X[idx, :]
#         if data_type.upper() == 'normal'.upper():
#             y = np.ones((X.shape[0], 1)) * NORMAL
#         elif data_type.upper() == 'abnormal'.upper():
#             y = np.ones((X.shape[0], 1)) * ABNORMAL
#         else:
#             # todo
#             le(f"KeyError: {data_type}")
#             raise KeyError(f'{data_type}')
#         _data = np.hstack((X, y))
#         nans = np.isnan(_data).any(axis=1)  # remove NaNs
#         _data = _data[~nans]
#         return _data
#
#     # Get normal data
#     try:
#         if end == -1:
#             X = genfromtxt(normal_pth, delimiter=',', skip_header=1)[:, start:]  # skip_header=1
#         else:
#             X = genfromtxt(normal_pth, delimiter=',', skip_header=1)[:, start:end]  # skip_header=1
#     except FileNotFoundError as e:
#         le(f'FileNotFoundError: {e}')
#         raise FileNotFoundError(e)
#
#     normal_data = _label_and_combine_data(X, size=n_normal, data_type='normal')
#
#     # Get abnormal data
#     try:
#         if end == -1:
#             X = genfromtxt(abnormal_pth, delimiter=',', skip_header=1)[:, start:]
#         else:
#             X = genfromtxt(abnormal_pth, delimiter=',', skip_header=1)[:, start:end]
#     except FileNotFoundError as e:
#         le(f'FileNotFoundError: {e}')
#         raise FileNotFoundError(e)
#     abnormal_data = _label_and_combine_data(X, size=n_abnormal, data_type='abnormal')
#
#     # data={'X_train':'', 'y_train':'', 'X_test':'', 'y_test':''}
#     data = {'normal_data': normal_data, 'abnormal_data': abnormal_data,
#             'label': {'NORMAL': NORMAL, 'ABNORMAL': ABNORMAL}}
#
#     return data
#
#
#
#
# def show_tree(data_tree, show_type='tag'):
#     def get_label(node):
#         if show_type.upper() == 'NID':
#             v_str = node.identifier
#         else:  # default: tag
#             v_str = node.tag
#
#         _data = node.data
#         len_v = 40
#         if type(node.data) == dict:
#             if 'aucs' in _data.keys():
#                 v_str += ':' + ','.join([str(v) for v in _data['aucs']]) + \
#                          ', best_params: ' + ', '.join(
#                     [f"{k}:{textwrap.shorten(str(v), width=len_v, placeholder='...')}"
#                      for k, v in _data['models'][-1]['best_params'].items()])
#             else:
#                 v_str += ':' + ','.join([f"{k}:{textwrap.shorten(str(v), width=len_v, placeholder='...')}"
#                                          for k, v in _data.items()])
#         else:
#             v_str += ':' + str(node.data)[:2 * len_v]
#         return v_str
#
#     # Adapted from  __print_backend() in tree.py
#     #: ROOT, DEPTH, WIDTH, ZIGZAG constants :
#     (ROOT, DEPTH, WIDTH, ZIGZAG) = list(range(4))
#     nid = None
#     level = ROOT
#     filter = None
#     key = None
#     reverse = False
#     line_type = 'ascii-ex'
#     data_tree._reader = "\n"
#
#     def write(line):
#         data_tree._reader += line.decode('utf-8') + "\n"
#
#     func = write
#
#     for pre, node in data_tree._Tree__get(nid, level, filter, key, reverse, line_type):
#         label = get_label(node)
#         func('{0}{1}'.format(pre, label).encode('utf-8'))
#     lg('\n')
#     lg(data_tree._reader)
#
#     return 0
#
#
# def _build_data_subtree(node):
#     tree = Tree()
#     tree.add_node(node, parent=None)
#     nid = node.identifier
#     # The node (i.e., data source) might generate multiple children nodes (e.g., with and without header)
#     for i, _ in enumerate([node.tag]):  # Might be needed to be expanded in the future
#         tag = f"data_{i}"
#         child_nid = f"{nid}->{tag}"
#         try:
#             data = extract_data(normal_pth=node.data['normal_pth'], abnormal_pth=node.data['abnormal_pth'],
#                                 meta_data=node.data)
#         except IOError as e:
#             le(f'IOError:{e}')
#             data = {}
#         tree.create_node(tag=tag, identifier=child_nid, parent=nid, data=data)
#
#     # # breadth-first traversal of the tree (BFT)
#     # for i, tag in enumerate(data_tree.expand_tree(mode=Tree.WIDTH)):
#     #     lg(f"{i}, {tag}, {data_tree[tag]}")
#     #     node = data_tree[tag]
#     #     # get the depth of the current node. The depth of the root node is 0: data_tree.depth(data_tree['root'])=>0
#     #     if data_tree.depth(node) > 1:
#     #         break
#     #     else:
#     #         data = extract_data(normal_pth=node.data['normla_pth'], abnormal_pth=node.data['abnormal_pth'])
#     #         data_tree.create_node(tag=f'{tag}_data', parent=tag, data=data)
#     #         pass
#
#     return tree
#
#
# def _grow_tree(tree, new_nodes_meta, current_nid='', result=''):
#     """DFT: recursive build subtree
#
#     Parameters
#     ----------
#     new_node_tags
#     nid
#     results
#     data_tree
#
#     Returns
#     -------
#
#     """
#
#     if len(new_nodes_meta) > 0:
#         new_tag, new_data = new_nodes_meta.pop(0)
#         new_nid = f"{current_nid}->{new_tag}"
#         if not tree.contains(nid=new_nid):
#             tree.create_node(tag=new_tag, identifier=new_nid, parent=current_nid, data=new_data)
#             _grow_tree(tree, new_nodes_meta, current_nid=new_nid, result=result)
#         else:
#             # new_tag, new_data = new_nodes_meta[0]
#             current_nid += f"->{new_tag}"
#             _grow_tree(tree, new_nodes_meta, current_nid, result=result)
#     else:  # Add result into the leaves
#         new_tag = 'result'
#         new_nid = f"{current_nid}->{new_tag}"
#         tree.create_node(tag=new_tag, identifier=new_nid, parent=current_nid, data=result)
#
#     return ''
#
# def dump_tree(data_tree, output_file='data_tree.dat'):
#     with open(output_file, 'wb') as f:
#         pickle.dump(data_tree, f)
#
# def get_node_data(data_tree, nid):
#     """get combined data from root to the given node
#
#     Parameters
#     ----------
#     nodes
#
#     Returns
#     -------
#
#     """
#     node = data_tree.get_node(nid)
#     data = node.data
#     node = data_tree.parent(nid)
#     nid = node.identifier
#     while node:
#         data= dict(**data, **node.data)
#         if node.identifier==data_tree.root: # data_tree.root is the root_ID
#             break
#         node = data_tree.parent(nid)
#         nid = node.identifier
#     return data
#
# class DATA_TREE():
#
#     def __init__(self):
#         self.data_tree = ''
#
#     def create(self):
#         # Build tree
#         # Store data and the corresponding results generated by different algorithms into a tree
#         self.data_tree = Tree()
#         # Tag can be the same with others (tree.show() prints tag); however, identifier cannot be
#         root_ID = 'data_root'
#         self.data_tree.create_node(tag=root_ID, identifier=root_ID, parent=None, data={})  #
#
#         # 1. Add data nodes into the tree
#         datasets = {}
#         # Data source: examples/output_data_PCA/DS20_PU_SMTV/DS21-srcIP_10.42.0.1-all_features-header_False-gs_False
#         input_dir = './output_data_PCA/DS20_PU_SMTV/'
#         for i, data_source in enumerate(['/DS20_PU_SMTV',]):
#             tag = data_source
#             parent_id = root_ID
#             nid = f"{root_ID}->{tag}"
#             data = {'input_dir': input_dir, 'data_source':data_source}
#             node_data_source = Node(tag=tag, identifier=nid, data=data)
#             self.data_tree.add_node(node_data_source, parent=parent_id)
#
#             for j, sub_data in enumerate(['PC1', 'PC2']):
#                 tag = sub_data
#                 parent_id = node_data_source.identifier
#                 nid += f'->{tag}'
#                 data = {'sub_data': sub_data}
#                 node_sub_data = Node(tag=tag, identifier=nid, data=data)
#                 self.data_tree.add_node(node_sub_data, parent=parent_id)
#
#                 for s, feat_set in enumerate(['iat', 'size', 'iat_size', 'fft_iat', 'fft_size', 'fft_iat_size', 'stat']):
#                     tag = feat_set
#                     parent_id = node_sub_data.identifier
#                     nid += f'->{tag}'
#                     data = {'feat_set':feat_set}
#                     node_feat_set = Node(tag=tag, identifier=nid, data=data)
#                     self.data_tree.add_node(node_feat_set, parent=parent_id)
#
#                     for t, header in enumerate(['is_header:False', 'is_header:True']):
#                         try:
#                             normal_pth = pth.join(input_dir,data_source, sub_data, feat_set, header,
#                                                   f'normal.csv')
#                             abnormal_pth = pth.join(input_dir,data_source, sub_data, feat_set, header,
#                                                     f'abnormal.csv')
#                             tag = header
#                             parent_id = node_feat_set.identifier
#                             nid +=f'->{tag}'
#
#                             data =  {'header':header, 'normal_pth':normal_pth, 'abnormal_pth':abnormal_pth,
#                                      'idxs_feat': [0, -1], 'train_size': -1, 'test_size': -1}
#                             data1= extract_data(normal_pth, abnormal_pth, meta_data=data)
#                             data = dict(**data, **data1)
#                             node_header = Node(tag=tag, identifier=nid,  data=data)
#                             self.data_tree.add_node(node_header, parent=parent_id)
#
#                         except (ValueError, FileNotFoundError, FileExistsError, KeyError) as e:
#                             le(f"Error: {e}")
#                             continue
#
#
#         # # Data source: IoT_1
#         # data_name = 'IoT_1'
#         # input_dir = './input_data/IoT_data/'
#         # normal_pth = pth.join(input_dir, 'traffic-securitycam.csv')
#         # abnormal_pth = pth.join(input_dir, 'traffic-thermostat.csv')
#         # meta_data = {'input_dir': input_dir, 'normal_pth': normal_pth, 'abnormal_pth': abnormal_pth,
#         #              'idxs_feat': [0, 115], 'train_size': 8000, 'test_size': 2000}
#         # datasets[data_name] = meta_data
#         #
#         # # Data source: IoT_2
#         # data_name = 'IoT_2'
#         # input_dir = './input_data/IoT_data/'
#         # normal_pth = pth.join(input_dir, 'traffic-securitycam.csv')
#         # abnormal_pth = pth.join(input_dir, 'traffic-doorbell1.csv')
#         # meta_data = {'input_dir': input_dir, 'normal_pth': normal_pth, 'abnormal_pth': abnormal_pth,
#         #              'idxs_feat': [0, 115], 'train_size': 8000, 'test_size': 2000}
#         # datasets[data_name] = meta_data
#         #
#         # # Data source: ...
#         # data_name = 'xxx'
#         # input_dir = './input_data/IoT_data/'
#         # normal_pth = pth.join(input_dir, '')
#         # abnormal_pth = pth.join(input_dir, '')
#         # meta_data = {'input_dir': input_dir, 'normal_pth': normal_pth, 'abnormal_pth': abnormal_pth,
#         #              'idxs_feat': [0, 115], 'train_size': 8000, 'test_size': 2000}
#         # datasets[data_name] = meta_data
#
#
# if __name__ == '__main__':
#     dt = DATA_TREE()
#     dt.create()
#
#     dump_tree(dt.data_tree, output_file='data_tree.dat')
#     show_tree(dt.data_tree)
#
#     data = get_node_data(dt.data_tree, nid='data_root->/DS20_PU_SMTV->PC1->iat')
#     print(data)
