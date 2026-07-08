from .iid import IIDBuffer, IIDConfig


def getBufferBuilder(name: str):
    buffers = {
        'iid': (IIDBuffer, IIDConfig),
    }
    if name not in buffers:
        raise ValueError(f'Unknown buffer type: {name}')
    return buffers[name]
