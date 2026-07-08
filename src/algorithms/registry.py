from typing import Type
from algorithms.BaseAgent import BaseAgent

from algorithms.nn.DQN import DQN
from algorithms.nn.QRC import QRC
from algorithms.nn.QRCPostAdam import QRCPostAdam
from algorithms.nn.QRCTarget import QRCTarget


def getAgent(name: str) -> Type[BaseAgent]:
    if name == 'DQN':
        return DQN
    if name == 'QRC':
        return QRC
    if name == 'QRCPostAdam':
        return QRCPostAdam
    if name == 'QRCTarget':
        return QRCTarget
    raise ValueError(f'Unknown algorithm: {name}')
