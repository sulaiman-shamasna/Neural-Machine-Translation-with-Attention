import tensorflow as tf

from preprocessing import *
from models import *
# from train import masked_loss, masked_acc

def tf_lower_and_split_punct(text: tf.Tensor) -> tf.Tensor:
    """
    Text standardization function to lower case and split punctuation.
    
    Args:
        text (tf.Tensor): Input text tensor.
    
    Returns:
        tf.Tensor: Processed text tensor.
    """
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

def masked_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the masked loss.
    
    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.
    
    Returns:
        tf.Tensor: Computed masked loss.
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)

def masked_acc(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the masked accuracy.
    
    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.
    
    Returns:
        tf.Tensor: Computed masked accuracy.
    """
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    
    return tf.reduce_sum(match) / tf.reduce_sum(mask)


custom_objects = {
    'tf_lower_and_split_punct': tf_lower_and_split_punct,
    'masked_loss': masked_loss,
    'masked_acc': masked_acc
}

new_model = tf.keras.models.load_model('big_model6', custom_objects=custom_objects)

# Check its architecture
new_model.summary()


# @Translator.add_method
# def translate(self,
#               texts, *,
#               max_length=50,
#               temperature=0.0):
#   # Process the input texts
#   context = self.encoder.convert_input(texts)
#   batch_size = tf.shape(texts)[0]

#   # Setup the loop inputs
#   tokens = []
#   attention_weights = []
#   next_token, done, state = self.decoder.get_initial_state(context)

#   for _ in range(max_length):
#     # Generate the next token
#     next_token, done, state = self.decoder.get_next_token(
#         context, next_token, done,  state, temperature)
        
#     # Collect the generated tokens
#     tokens.append(next_token)
#     attention_weights.append(self.decoder.last_attention_weights)
    
#     if tf.executing_eagerly() and tf.reduce_all(done):
#       break

#   # Stack the lists of tokens and attention weights.
#   tokens = tf.concat(tokens, axis=-1)   # t*[(batch 1)] -> (batch, t)
#   self.last_attention_weights = tf.concat(attention_weights, axis=1)  # t*[(batch 1 s)] -> (batch, t s)

#   result = self.decoder.tokens_to_text(tokens)
#   return result

result = new_model.translate(['¿Todavía está en casa?']) # Are you still home
result[0].numpy().decode()