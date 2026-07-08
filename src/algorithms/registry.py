from typing import Type
from algorithms.BaseAgent import BaseAgent

from algorithms.nn.DQN import DQN
from algorithms.nn.QRC import QRC
from algorithms.nn.QRCPostAdam import QRCPostAdam


def getAgent(name: str) -> Type[BaseAgent]:
    if name == 'DQN':
        return DQN
    if name == 'QRC':
        return QRC
    if name == 'QRCPostAdam':
        return QRCPostAdam
    raise ValueError(f'Unknown algorithm: {name}')
