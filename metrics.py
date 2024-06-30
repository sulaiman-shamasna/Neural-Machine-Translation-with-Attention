import tensorflow as tf

def masked_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
 
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_acc(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:

    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    
    return tf.reduce_sum(match) / tf.reduce_sum(mask)