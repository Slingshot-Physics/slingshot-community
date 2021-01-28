from enum import Enum

class ReductionType(Enum):
   TOTAL_FRAME_TIME = 1
   TIME_PER_BODY_PAIR = 2
   NUM_CALLS_PER_FRAME = 3
   TIME_PER_CALL = 4

class ReductionName:
   names = {
      ReductionType.TOTAL_FRAME_TIME: "total frame time",
      ReductionType.TIME_PER_BODY_PAIR: "time per body pair",
      ReductionType.NUM_CALLS_PER_FRAME: "num calls per frame",
      ReductionType.TIME_PER_CALL: "time per call",
   }
