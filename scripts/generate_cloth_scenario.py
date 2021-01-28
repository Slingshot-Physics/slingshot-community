from scenario_data_types import *

import argparse
import json
from typing import List, Tuple

import numpy as np

def generate_bodies(num_rows: int, num_cols: int) -> List[data_rigidbody_t]:
   delta_h: float = 0.5
   delta_w: float = 0.5

   start_pos: np.ndarray = np.array(
      [delta_w * (num_rows - 1) / 2.0, 0.0, delta_h * (num_cols - 1) / 2.0]
   )

   rigid_bodies = []
   for i in range(num_rows):
      for j in range(num_cols):
         body_id: int = num_rows * i + j

         pos: np.ndarray = start_pos - np.array(
            # [delta_w * i, 0, delta_h * j]
            [delta_w * j, 0, delta_h * i]
         )

         rigid_bodies.append(
            data_rigidbody_t(
               body_id,
               data_matrix33_t([0.0025, 0., 0., 0., 0.0025, 0., 0., 0., 0.0025]),
               data_vector3_t(v=pos.tolist()),
               data_vector3_t(),
               data_vector3_t(),
               data_quaternion_t(),
               data_vector3_t(),
               data_vector3_t(),
               0,
               0.125
            )
         )

   return rigid_bodies

def generate_springs_and_dampers(num_rows: int, num_cols: int) -> Tuple[
   List[data_forceSpring_t], List[data_forceVelocityDamper_t]
]:
   springs: List[data_forceSpring_t] = []
   dampers: List[data_forceVelocityDamper_t] = []

   # Horizontal springs
   for i in range(num_rows):
      for j in range(num_cols - 1):
         parent_body_id: int = num_rows * i + j
         child_body_id: int = num_rows * i + j + 1

         print("horizontal spring between", parent_body_id, child_body_id)

         springs.append(
            data_forceSpring_t(
               parent_body_id,
               child_body_id,
               data_vector3_t(),
               data_vector3_t(),
               0.5,
               -40.0
            )
         )

         dampers.append(
            data_forceVelocityDamper_t(
               parent_body_id,
               child_body_id,
               data_vector3_t(),
               data_vector3_t(),
               -0.125
            )
         )

   # Vertical springs
   for i in range(num_rows - 1):
      for j in range(num_cols):
         parent_body_id: int = num_rows * i + j
         child_body_id: int = num_rows * (i + 1) + j

         print("vertical spring between", parent_body_id, child_body_id)

         springs.append(
            data_forceSpring_t(
               parent_body_id,
               child_body_id,
               data_vector3_t(),
               data_vector3_t(),
               0.5,
               -40.0
            )
         )

         dampers.append(
            data_forceVelocityDamper_t(
               parent_body_id,
               child_body_id,
               data_vector3_t(),
               data_vector3_t(),
               -0.125
            )
         )

   # Cross 1 springs
   for i in range(num_rows - 1):
      for j in range(num_cols - 1):
         parent_body_id: int = num_rows * i + j
         child_body_id: int = num_rows * (i + 1) + j + 1

         print("cross 1 spring between", parent_body_id, child_body_id)

         springs.append(
            data_forceSpring_t(
               parent_body_id,
               child_body_id,
               data_vector3_t(),
               data_vector3_t(),
               0.7071,
               -40.0
            )
         )

         dampers.append(
            data_forceVelocityDamper_t(
               parent_body_id,
               child_body_id,
               data_vector3_t(),
               data_vector3_t(),
               -0.125
            )
         )

   # Cross 2 springs
   for i in range(num_rows - 1):
      for j in range(1, num_cols):
         parent_body_id: int = num_rows * i + j
         child_body_id: int = num_rows * (i + 1) + j - 1

         print("cross 2 spring between", parent_body_id, child_body_id)

         springs.append(
            data_forceSpring_t(
               parent_body_id,
               child_body_id,
               data_vector3_t(),
               data_vector3_t(),
               0.7071,
               -40.0
            )
         )

         dampers.append(
            data_forceVelocityDamper_t(
               parent_body_id,
               child_body_id,
               data_vector3_t(),
               data_vector3_t(),
               -0.125
            )
         )

   return springs, dampers

def generate_colliders(num_rows: int, num_cols: int) -> List[
   data_isometricCollider_t
]:
   colliders = []
   for body_id in range(num_rows * num_cols):
      colliders.append(
         data_isometricCollider_t(
            body_id,
            int(ShapeTypes.CUBE.value),
            0.1,
            0.75,
            1
         )
      )

   return colliders

def generate_shapes(num_rows: int, num_cols: int) -> List[
   data_shape_t
]:
   shapes: List[data_shape_t] = []

   for body_id in range(num_rows * num_cols):
      temp_shape = data_shape_t(
         shapeType=int(ShapeTypes.CUBE.value),
         bodyId=body_id
      )

      setattr(temp_shape, "cube", shapeCube(0.25, 0.25, 0.25))
      shapes.append(temp_shape)

   return shapes

def generate_cloth_scenario(args):
   bodies: List[data_rigidbody_t] = generate_bodies(args.n, args.m)
   springs, dampers = generate_springs_and_dampers(args.n, args.m)
   colliders = generate_colliders(args.n, args.m)
   shapes = generate_shapes(args.n, args.m)
   constant_forces: List[data_forceConstant_t] = []
   for body in bodies:
      constant_forces.append(
         data_forceConstant_t(
            parentId=body.id,
            parentLinkPoint=data_vector3_t(v=[0.0, 0.0, 0.0]),
            force=data_vector3_t(v=[0.0, 0.0, -9.8]),
            frame=1
         )
      )

   # Corner springs
   springs.append(
      data_forceSpring_t(
         -1,
         0,
         data_vector3_t(
            [bodies[0].linpos.v[0] + 3, bodies[0].linpos.v[1], bodies[0].linpos.v[2] + 3]
         ),
         data_vector3_t(),
         1.4 * 3.0,
         -100.0
      )
   )

   springs.append(
      data_forceSpring_t(
         -1,
         args.m - 1,
         data_vector3_t(
            [bodies[args.m - 1].linpos.v[0] - 3, bodies[args.m - 1].linpos.v[1], bodies[args.m - 1].linpos.v[2] + 3]
         ),
         data_vector3_t(),
         1.4 * 3.0,
         -100.0
      )
   )

   dampers.append(
      data_forceVelocityDamper_t(
         parentId=-1,
         childId=0,
         parentLinkPoint=data_vector3_t(),
         childLinkPoint=data_vector3_t(),
         damperCoeff=-10.1
      )
   )

   dampers.append(
      data_forceVelocityDamper_t(
         parentId=-1,
         childId=args.m - 1,
         parentLinkPoint=data_vector3_t(),
         childLinkPoint=data_vector3_t(),
         damperCoeff=-10.1
      )
   )

   scenario = data_scenario_t(
      numBodies=(args.n * args.m),
      numIsometricColliders=(args.n * args.m),
      numShapes=(args.n * args.m),
      numConstantForces=(args.n * args.m),
      numSpringForces=len(springs),
      numVelocityDamperForces=len(dampers),
      bodies=bodies,
      constantForces=constant_forces,
      springForces=springs,
      velocityDamperForces=dampers,
      shapes=shapes,
      isometricColliders=colliders
   )

   with open(args.filename, "w") as file_out:
      json.dump(scenario, file_out, default=lambda o: o.__dict__, indent=2)

def main() -> None:
   parser = argparse.ArgumentParser(description="NxM cloth scenario generator")

   parser.add_argument(
      "n", default=4, type=int, help="Number of rows of cloth nodes"
   )

   parser.add_argument(
      "m", default=4, type=int, help="Number of columns of cloth nodes"
   )

   parser.add_argument("--filename", required=True)

   args = parser.parse_args()

   generate_cloth_scenario(args)

if __name__ == "__main__":
   main()
