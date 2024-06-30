import tensorflow as tf

inputs = [
    'Hace mucho frio aqui.', # "It's really cold here."
    'Esta es mi vida.', # "This is my life."
    'Su cuarto es un desastre.' # "His room is a mess"
]

# reloaded = tf.saved_model.load('dynamic_translator')
# # _ = reloaded.translate(tf.constant(inputs)) #warmup


load_options = tf.saved_model.LoadOptions(
    experimental_io_device='/job:localhost'
)

reloaded = tf.saved_model.load('dynamic_translator', options=load_options)

result = reloaded.translate(tf.constant(inputs))





print(result[0].numpy().decode())
print(result[1].numpy().decode())
print(result[2].numpy().decode())
print()