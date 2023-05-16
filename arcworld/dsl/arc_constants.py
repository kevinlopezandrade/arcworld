from arcworld.dsl.arc_types import *

ZERO = 0
ONE = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5
SIX = 6
SEVEN = 7
EIGHT = 8
NINE = 9
TEN = 10
F = False
T = True

NEG_ONE = -1

ORIGIN = (0, 0)
UNITY = (1, 1)
DOWN = (1, 0)
RIGHT = (0, 1)
UP = (-1, 0)
LEFT = (0, -1)

NEG_TWO = -2
NEG_UNITY = (-1, -1)
UP_RIGHT = (-1, 1)
DOWN_LEFT = (1, -1)

ZERO_BY_TWO = (0, 2)
TWO_BY_ZERO = (2, 0)
TWO_BY_TWO = (2, 2)
THREE_BY_THREE = (3, 3)


arc_constants_mapper = {
    'ZERO': {'type': Integer, 'value': ZERO},
    'ONE': {'type': Integer, 'value': ONE},
    'TWO': {'type': Integer, 'value': TWO},
    'THREE': {'type': Integer, 'value': THREE},
    'FOUR': {'type': Integer, 'value': FOUR},
    'FIVE': {'type': Integer, 'value': FIVE},
    'SIX': {'type': Integer, 'value': SIX},
    'SEVEN': {'type': Integer, 'value': SEVEN},
    'EIGHT': {'type': Integer, 'value': EIGHT},
    'NINE': {'type': Integer, 'value': NINE},
    'TEN': {'type': Integer, 'value': TEN},
    'F': {'type': Boolean, 'value': F},
    'T': {'type': Boolean, 'value': T},
    'NEG_ONE': {'type': Integer, 'value': NEG_ONE},
    'ORIGIN': {'type': IntegerTuple, 'value': ORIGIN},
    'UNITY': {'type': IntegerTuple, 'value': UNITY},
    'DOWN': {'type': IntegerTuple, 'value': DOWN},
    'RIGHT': {'type': IntegerTuple, 'value': RIGHT},
    'UP': {'type': IntegerTuple, 'value': UP},
    'LEFT': {'type': IntegerTuple, 'value': LEFT},
    'NEG_TWO': {'type': Integer, 'value': NEG_TWO},
    'NEG_UNITY': {'type': IntegerTuple, 'value': NEG_UNITY},
    'UP_RIGHT': {'type': IntegerTuple, 'value': UP_RIGHT},
    'DOWN_LEFT': {'type': IntegerTuple, 'value': DOWN_LEFT},
    'TWO_BY_TWO': {'type': IntegerTuple, 'value': TWO_BY_TWO},
    'THREE_BY_THREE': {'type': IntegerTuple, 'value': THREE_BY_THREE},
    'ZERO_BY_TWO': {'type': IntegerTuple, 'value': ZERO_BY_TWO},
    'TWO_BY_ZERO': {'type': IntegerTuple, 'value': TWO_BY_ZERO}
}
