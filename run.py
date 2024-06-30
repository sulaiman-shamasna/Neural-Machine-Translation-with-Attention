import tensorflow as tf
from preprocessing import target_raw, context_raw
from train import model

# from load import (tf_lower_and_split_punct, masked_loss, masked_acc)

def run():
    # from train import model

    # custom_objects = {
    #                 'tf_lower_and_split_punct': tf_lower_and_split_punct,
    #                 'masked_loss': masked_loss,
    #                 'masked_acc': masked_acc
    #             }

    # model = tf.keras.models.load_model('trained_models/big_model11', custom_objects=custom_objects)
    # model.summary()

    import textwrap

    print('Expected output:\n', '\n'.join(textwrap.wrap(target_raw[-1])))

    inputs = [
        'Hace mucho frio aqui.', # "It's really cold here."
        'Esta es mi vida.', # "This is my life."
        'Su cuarto es un desastre.', # "His room is a mess"
        'Hola.'
    ]

    for t in inputs:
        print(model.translate([t])[0].numpy().decode())

        print()
        result = model.translate(inputs)

        print(result[0].numpy().decode())
        print(result[1].numpy().decode())
        print(result[2].numpy().decode())
        print(result[3].numpy().decode())
    #     print()

if __name__ == "__main__":
    run()
    