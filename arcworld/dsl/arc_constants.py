import arcworld.dsl.arc_types as ARC_TYPE

# TODO: Maybe setting the __all__ attribute in arc_types moduel
# works better and we avoid defining ARC_TYPE and use the star import.

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
    "ZERO": {"type": ARC_TYPE.Integer, "value": ZERO},
    "ONE": {"type": ARC_TYPE.Integer, "value": ONE},
    "TWO": {"type": ARC_TYPE.Integer, "value": TWO},
    "THREE": {"type": ARC_TYPE.Integer, "value": THREE},
    "FOUR": {"type": ARC_TYPE.Integer, "value": FOUR},
    "FIVE": {"type": ARC_TYPE.Integer, "value": FIVE},
    "SIX": {"type": ARC_TYPE.Integer, "value": SIX},
    "SEVEN": {"type": ARC_TYPE.Integer, "value": SEVEN},
    "EIGHT": {"type": ARC_TYPE.Integer, "value": EIGHT},
    "NINE": {"type": ARC_TYPE.Integer, "value": NINE},
    "TEN": {"type": ARC_TYPE.Integer, "value": TEN},
    "F": {"type": ARC_TYPE.Boolean, "value": F},
    "T": {"type": ARC_TYPE.Boolean, "value": T},
    "NEG_ONE": {"type": ARC_TYPE.Integer, "value": NEG_ONE},
    "ORIGIN": {"type": ARC_TYPE.IntegerTuple, "value": ORIGIN},
    "UNITY": {"type": ARC_TYPE.IntegerTuple, "value": UNITY},
    "DOWN": {"type": ARC_TYPE.IntegerTuple, "value": DOWN},
    "RIGHT": {"type": ARC_TYPE.IntegerTuple, "value": RIGHT},
    "UP": {"type": ARC_TYPE.IntegerTuple, "value": UP},
    "LEFT": {"type": ARC_TYPE.IntegerTuple, "value": LEFT},
    "NEG_TWO": {"type": ARC_TYPE.Integer, "value": NEG_TWO},
    "NEG_UNITY": {"type": ARC_TYPE.IntegerTuple, "value": NEG_UNITY},
    "UP_RIGHT": {"type": ARC_TYPE.IntegerTuple, "value": UP_RIGHT},
    "DOWN_LEFT": {"type": ARC_TYPE.IntegerTuple, "value": DOWN_LEFT},
    "TWO_BY_TWO": {"type": ARC_TYPE.IntegerTuple, "value": TWO_BY_TWO},
    "THREE_BY_THREE": {"type": ARC_TYPE.IntegerTuple, "value": THREE_BY_THREE},
    "ZERO_BY_TWO": {"type": ARC_TYPE.IntegerTuple, "value": ZERO_BY_TWO},
    "TWO_BY_ZERO": {"type": ARC_TYPE.IntegerTuple, "value": TWO_BY_ZERO},
}
