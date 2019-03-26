import os
import os.path as osp
from keras import backend as K
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
from keras.models import Sequential,load_model
from tensorflow.python.framework import graph_io

# def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
#     from tensorflow.python.framework.graph_util import convert_variables_to_constants
#     graph = session.graph
#     with graph.as_default():
#         freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
#         output_names = output_names or []
#         output_names += [v.op.name for v in tf.global_variables()]
#         input_graph_def = graph.as_graph_def()
#         if clear_devices:
#             for node in input_graph_def.node:
#                 node.device = ""
#         frozen_graph = convert_variables_to_constants(session, input_graph_def,
#                                                       output_names, freeze_var_names)
#         return frozen_graph
#
# input_fld = '/Users/suzhenyu/Downloads/'
# weight_file = 'machine-intelligence190306102726ResNet_18_model.h5'
# output_graph_name = 'machine-intelligence190306102726ResNet_18_model.pb'
# output_fld = '/Users/suzhenyu/Downloads/'
# if not os.path.isdir(output_fld):
#     os.mkdir(output_fld)
# weight_file_path = osp.join(input_fld , weight_file)
# print weight_file_path
# K.set_learning_phase(0)
# net_model = load_model(weight_file_path)
# print('input is :', net_model.input.name)
# print ('output is:', net_model.output.name)
# sess = K.get_session()
# frozen_graph = freeze_session(K.get_session(), output_names=[net_model.output.op.name])
# graph_io.write_graph(frozen_graph, output_fld, output_graph_name, as_text=True)
# print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))
# export_dir = input_fld + '/tensorflow_model/'
# graph_pb = osp.join(output_fld, output_graph_name)
# builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
# with tf.gfile.GFile(graph_pb, "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
# sigs = {}
# with tf.Session(graph=tf.Graph()) as sess:
#     tf.import_graph_def(graph_def, name="")
#     g = tf.get_default_graph()
#     inp = g.get_tensor_by_name(net_model.input.name)
#     out = g.get_tensor_by_name(net_model.output.name)
#     sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
#         tf.saved_model.signature_def_utils.predict_signature_def(
#             {"in": inp}, {"out": out})
#     builder.add_meta_graph_and_variables(sess,
#                                          [tag_constants.SERVING],
#                                          signature_def_map=sigs)
# builder.save()
# print("save normal graph sucess")

input_path = '/Users/suzhenyu/Downloads/'
weight_file = 'ResNet_18_model.h5'
weight_file_path = osp.join(input_path,weight_file)
output_graph_name = weight_file[:-3] + '.pb'

def h5_to_pb(h5_model,output_dir,model_name,out_prefix = "output_",log_tensorboard = True):
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i],out_prefix + str(i + 1))
    sess = K.get_session()
    from tensorflow.python.framework import graph_util,graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
    graph_io.write_graph(main_graph,output_dir,name = model_name,as_text = False)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir,model_name),output_dir)

output_dir = osp.join(os.getcwd(),"trans_model")
if not os.path.isdir(output_dir):
     os.mkdir(output_dir)
h5_model = load_model(weight_file_path)
h5_to_pb(h5_model,output_dir = output_dir,model_name = output_graph_name)
print('model saved')