import tensorflow as tf
from models import model

from metrics import masked_loss, masked_acc
from preprocessing import target_text_processor, train_ds, val_ds


model.compile(optimizer='adam',
              loss=masked_loss, 
              metrics=[masked_acc, masked_loss])

vocab_size = target_text_processor.vocabulary_size()

print({
    "expected_loss": tf.math.log(tf.cast(vocab_size, tf.float32)).numpy(),
    "expected_acc": 1 / vocab_size
})

model.evaluate(val_ds, steps=20, return_dict=True)

history = model.fit(
    train_ds.repeat(), 
    epochs=5,
    steps_per_epoch=50,
    validation_data=val_ds,
    validation_steps=20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3),
    ]
)

