"""
Get the total number of FLOPs used in a single forward pass for a saved model.
This code is also a quick way to load the neural net of a specific agent.


FLOPs should be read from the root:

Profile:
node name | # float_ops
_TFProfRoot (--/3.54k flops)
"""
import AZ_helper_lib as AZh
import tensorflow as tf


# Name of the directory where the model is saved.
dir_name = 'pentago'
model_name = 'q_0_0'
path = './models/' + dir_name + '/' + model_name + '/'

print(dir_name)
print(model_name)

config = AZh.load_config(path)
model = AZh.load_model_from_checkpoint(config)
# print model profile:
tf.profiler.profile(model._session.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
