import tensorflow as tf

######################################################
# Creating a simple graph and putting a value into it
######################################################

'''
At this point TensorFlow has already started managing a lot of state for us. 
There's already an implicit default graph, for example. Internally, the default graph lives 
in the _default_graph_stack, but we don't have access to that directly. We use tf.get_default_graph().
'''
graph = tf.get_default_graph()

'''
Currently, there isn't anything in the graph. We’ll need to put everything we want TensorFlow to compute 
into that graph. Let's start with a simple constant input value of one.
'''
input_value = tf.constant(1.0)


'''
The nodes of the TensorFlow graph are called “operations” or “ops”. We can see what 
operations are in the graph with graph.get_operations().
'''
operations = graph.get_operations()
print('`operations` is now: {}'.format(operations))
# print(operations[0].node_def)


'''
Note that this doesn't tell us what that number is. To evaluate input_value and get a numerical value out,
 we need to create a “session” where graph operations can be evaluated and then explicitly ask to evaluate 
 or “run” input_value. (The session picks up the default graph by default.)
 
It may feel a little strange to “run” a constant. But it isn't so different from evaluating 
an expression as usual in Python; it's just that TensorFlow is managing its own space of things—the 
computational graph—and it has its own method of evaluation.
'''

sess = tf.Session()
print('`input_value` evaluates to: {}'.format(sess.run(input_value)))


######################################################
# Creating a neuron
######################################################

weight = tf.Variable(0.8)
for op in graph.get_operations(): print(op.name)
output_value = weight * input_value
op = graph.get_operations()[-1]
print('The `op.name` is: {}'.format(op.name))
for op_input in op.inputs: print(op_input)

init = tf.global_variables_initializer()
sess.run(init)

print('`output_value` evaluates to: {}'.format(sess.run(output_value)))

