import tensorflow as tf

##### TensorFlow Constant #####

const1 = tf.constant ( 5 )
const2 = tf.constant ( 10 )

print ( const1 ) ## these values are not printed.
print ( const2 ) ##

with tf.Session () as sess : ## after session start, the results are printed.
    const1_result, const2_result = sess.run ( [ const1, const2 ] )
    print ( const1_result )
    print ( const2_result )

##### TensorFlow Calculation #####
add_op = tf.add ( const1, const2 )
mul_op = tf.multiply ( add_op, const2 )

print ( add_op ) ## these are not printed, either.
print ( mul_op ) ##

with tf.Session () as sess :
    add_result, mul_result = sess.run ( [ add_op, mul_op ] )
    print ( add_result )
    print ( mul_result )

add_op_2 = const1 + const2
mul_op_2 = add_op_2 * const2

with tf.Session () as sess :
    add_op_2_result, mul_op_2_result = sess.run ( [ add_op_2, mul_op_2 ] )
    print ( add_op_2_result )
    print ( mul_op_2_result )

##### TensorFlow Variable #####
var_v = tf.Variable ( 10 )
const_v = tf.constant ( 5 )
calc_op = var_v * const_v
assign_op = tf.assign ( var_v, calc_op ) ## var_v is assgined calc_op result.

## variables have to be initialized when the session start
with tf.Session () as sess :
    sess.run ( tf.global_variables_initializer () )
    print ( sess.run ( var_v ) ) ## 10

    sess.run ( assign_op )
    print ( sess.run ( var_v ) ) ## 50 ( = 10 * 5 )

    sess.run ( assign_op )
    print ( sess.run ( var_v ) ) ## 250 ( = 50 * 5 )

##### Tensorflow placeholder #####
## placeholder is the storage for data.
## the data is not defined when the graph is constracted.
## the value for the data is given when execution.
const_h = tf.constant ( 1 )
holder_h = tf.placeholder ( tf.int32 )
add_op_h = const_h + holder_h

with tf.Session () as sess :
    result = sess.run ( add_op_h, feed_dict = { holder_h: 5 } )
    print ( result ) ## 6 ( = 1 + 5 )

    result = sess.run ( add_op_h, feed_dict = { holder_h: 10 } )
    print ( result ) ## 11 ( = 1 + 10 )


holder1_s = tf.placeholder ( tf.int32 )
holder2_s = tf.placeholder ( tf.int32, [ 3 ] ) ## array can be set by shape
mul_op_s = holder1_s * holder2_s

with tf.Session () as sess :
    result = sess.run ( mul_op_s, feed_dict = { holder1_s: 2, holder2_s: [ 0, 1, 2 ] } )
    print ( result ) ## [ 0 2 4] ( = [ 2 * 0, 2 * 1, 2 * 2 ] )

    result = sess.run ( mul_op_s, feed_dict = { holder1_s: 5, holder2_s: [ 0, 10, 20 ] } )
    print ( result ) ## [ 0 50 100 ] ( = [ 5 * 0, 5 * 10, 5 * 20 ] )
    

holder1_n = tf.placeholder ( tf.int32 )
holder2_n = tf.placeholder ( tf.int32, [ None ] )
mul_op_n = holder1_n * holder2_n

with tf.Session () as sess :
    result = sess.run ( mul_op_n, feed_dict = { holder1_n: 2, holder2_n: [ 0, 1 ] } )
    print ( result ) ## [ 0 2 ] ( = [ 2 * 0, 2 * 1 ] )

    result = sess.run ( mul_op_n, feed_dict = { holder1_n: 5, holder2_n: [ 0, 1, 2, 3, 4 ] } )
    print ( result ) ## [ 0 5 10 15 20 ] ( = [ 5 * 0, 5 * 1, 5 * 2, 5 * 3, 5 * 4 ] )
