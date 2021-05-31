import tensorflow as tf

def cross_entropy_loss(logits, annotation):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                         labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                         name="entropy"))
    return loss

