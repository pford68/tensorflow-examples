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


# =============================== Tensorboard
x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.multiply(w, x, name='output')     # tf.mul was replaced ny tf.multiply

'''
Creating the TensorBoard visual:
The first argument is aa output directory name.  That directory will be created if needed.

tf.train.SummaryWriter is deprecated, instead use tf.summary.FileWriter.
It will be removed after 2016-11-30. Instructions for updating: 
Please switch to tf.summary.FileWriter. The interface and behavior is the same; this is just a rename.
'''
summary_writer = tf.summary.FileWriter('log_simple_graph', sess.graph)


######################################################
# Making the neuron learn
######################################################
'''
A big part of what makes TensorFlow useful is that it handles gradient descent nicely.

Now that we’ve built our neuron, how does it learn? We set up an input value of 1.0. 
Let's say the correct output value is zero. That is, we have a very simple “training set” 
of just one example with one feature, which has the value one, and one label, which is zero. 
We want the neuron to learn the function taking one to zero.

Currently, the system takes the input one and returns 0.8, which is not correct. 
We need a way to measure how wrong the system is. We'll call that measure of wrongness 
the “loss” and give our system the goal of minimizing the loss. If the loss can be negative, 
then minimizing it could be silly, so let's make the loss the square of the difference 
between the current output and the desired output.
'''
y_ = tf.constant(0.0)
loss = (y - y_)**2


'''
So far, nothing in the graph does any learning. For that, we need an optimizer. 
We'll use a gradient descent optimizer so that we can update the weight based on the derivative of the loss. 
The optimizer takes a learning rate to moderate the size of the updates, which we'll set at 0.025.

We can make one operation that calculates and applies the gradients: the train_step.
Running the training step many times, the weight and the output value are now very close to zero. 
The neuron has learned!
'''
sess.run(tf.global_variables_initializer())
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

for i in range(100):
    sess.run(train_step)

print('The neuron now outputs: {}'.format(sess.run(y)))

######################################################
# Training diagnostics in TensorBoard
######################################################
summary_y = tf.summary.scalar('output', y)
summary_writer = tf.summary.FileWriter('log_simple_stat')

sess.run(tf.global_variables_initializer())

for i in range(100):
    summary_str = sess.run(summary_y)
    summary_writer.add_summary(summary_str, i)
    sess.run(train_step)