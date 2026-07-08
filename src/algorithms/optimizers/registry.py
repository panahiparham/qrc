import optax

def getOptimizerBuilder(name: str):
    name = name.lower()
    buffers = {
        'adam': optax.adam,
        'sgd': optax.sgd,
        'rmsprop': optax.rmsprop,
    }

    return buffers[name]
