import optax
from typing import Dict

def buildOptimizer(params: Dict) -> optax.GradientTransformation:
    name = params.get('name', 'adam').lower()

    if name == 'adam':
        return optax.adam(
            learning_rate=params['learning_rate'],
            b1=params.get('b1', 0.9),
            b2=params.get('b2', 0.999),
            eps=params.get('eps', 1e-8),
        )

    if name == 'sgd':
        return optax.sgd(
            learning_rate=params['learning_rate'],
            momentum=params.get('momentum', None),
            nesterov=params.get('nesterov', False),
        )

    if name == 'rmsprop':
        return optax.rmsprop(
            learning_rate=params['learning_rate'],
            decay=params.get('decay', 0.9),
            eps=params.get('eps', 1e-8),
            momentum=params.get('momentum', None),
        )

    raise NotImplementedError(f'Unknown optimizer: {name}')
