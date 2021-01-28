"""
Generates a randomized scenario with only collision constraints enabled.
Generates static and moving bodies. Initial orientations are all the same.
"""

from scenario_data_types import *

import argparse
import json

import numpy as np

def remap(x_min, x_max, y_min, y_max, val) -> float:
   """
   Linearly remap x values to y values given some value 'val' in the x domain.
   """
   return ((y_max - y_min)/(x_max - x_min)) * (val - x_min) + y_min

def vec3_from_ranges(x_range, y_range, z_range) -> data_vector3_t:
   """
   Uses ranges from the arg parser to generate a random 3-vector.
   """
   return data_vector3_t(
      [
         remap(0., 1., *x_range, np.random.rand()),
         remap(0., 1., *y_range, np.random.rand()),
         remap(0., 1., *z_range, np.random.rand()),
      ]
   )

def rando_quat() -> data_quaternion_t:
   n = np.random.rand(3)
   n = n / np.linalg.norm(n)

   theta = np.random.rand() * 2 * np.pi
   quat = data_quaternion_t([np.cos(theta/2), *list(np.sin(theta/2) * n)])

   return quat

def recalculate_moments(scenario: data_scenario_t) -> None:
   shapes: List[data_shape_t] = scenario.shapes

   for shape in shapes:
      inertia_diag = inertia(shape)
      for body in scenario.bodies:
         if body.id == shape.bodyId:
            body.J.m[0] = body.mass * inertia_diag[0]
            body.J.m[4] = body.mass * inertia_diag[1]
            body.J.m[8] = body.mass * inertia_diag[2]
            break

def add_floor(scenario: data_scenario_t, args) -> None:
   scenario.numBodies += 1
   scenario.numShapes += 1
   scenario.numIsometricColliders += 1

   floor_id: int = scenario.numBodies + 1

   scenario.bodies.append(
      data_rigidbody_t(
         floor_id,
         data_matrix33_t([1., 0., 0., 0., 1., 0., 0., 0., 1.]),
         data_vector3_t([0.0, 0.0, -5]),
         data_vector3_t([0.0, 0.0, 0.]),
         data_vector3_t([0.0, 0.0, 0.]),
         data_quaternion_t([1., 0.0, 0.0, 0.]),
         data_vector3_t([0.0, 0.0, 0.]),
         data_vector3_t([0.0, 0.0, 0.]),
         1,
         5.,
         data_vector3_t([0.0, 0.0, 0.]),
      )
   )

   scenario.isometricColliders.append(
      data_isometricCollider_t(
         floor_id,
         int(ShapeTypes.CUBE.value),
         0.75,
         0.75,
         1
      )
   )

   floor_shape = data_shape_t(
      int(ShapeTypes.CUBE.value),
      floor_id
   )

   setattr(
      floor_shape,
      "cube",
      shapeCube(
         length=1.5 * (max(args.range_x_position) - min(args.range_x_position)),
         width=1.5 * (max(args.range_y_position) - min(args.range_y_position)),
      )
   )

   scenario.shapes.append(
      floor_shape
   )

def generate_shapes(scenario: data_scenario_t) -> None:
   scenario.numShapes = scenario.numBodies

   for collider in scenario.isometricColliders:
      shape_type = ShapeTypes(collider.shapeName)

      shape = data_shape_t(
         bodyId = collider.bodyId,
         shapeType = shape_type.value
      )

      if shape_type == ShapeTypes.CAPSULE:
         setattr(shape, "capsule", shapeCapsule())
      elif shape_type == ShapeTypes.CYLINDER:
         setattr(shape, "cylinder", shapeCylinder())
      elif shape_type == ShapeTypes.SPHERE:
         setattr(shape, "sphere", shapeSphere())
      elif shape_type == ShapeTypes.CUBE:
         setattr(shape, "cube", shapeCube())

      scenario.shapes.append(shape)

def generate_colliders(
   body_ids: List[int],
   shape_types: List[int] = None
) -> List[data_isometricCollider_t]:
   colliders = []
   for body_id in body_ids:
      colliders.append(
         data_isometricCollider_t(
            body_id,
            int(
               np.random.choice(
                  [
                     ShapeTypes.CUBE.value,
                     ShapeTypes.CYLINDER.value,
                     ShapeTypes.SPHERE.value,
                     ShapeTypes.CAPSULE.value
                  ] if shape_types is None else shape_types
               )
            ),
            float(np.random.rand()),
            float(np.random.rand()),
            1
         )
      )

   return colliders

def generate_bodies(
   body_ids: List[int], static: bool, args
) -> List[data_rigidbody_t]:
   bodies = []
   for body_id in body_ids:
      bodies.append(
         data_rigidbody_t(
            id = body_id,
            J = data_matrix33_t([1., 0., 0., 0., 1., 0., 0., 0., 1.]),
            linpos= vec3_from_ranges(
               args.range_x_position,
               args.range_y_position,
               args.range_z_position
            ),
            linvel = vec3_from_ranges(
               args.range_x_velocity,
               args.range_y_velocity,
               args.range_z_velocity
            ),
            orientation = rando_quat(),
            rotvel = vec3_from_ranges(
               args.range_x_angular_velocity,
               args.range_y_angular_velocity,
               args.range_z_angular_velocity
            ),
            stationary = int(static),
            mass = np.random.rand() * 5 + 1.0
         )
      )

   return bodies

def generate_scenario_file(args):
   moving_body_ids = list(range(1, args.num_moving_bodies + 1))
   static_body_ids = list(
      range(
         args.num_moving_bodies + 1,
         args.num_static_bodies + args.num_moving_bodies + 1
      )
   )

   num_bodies = args.num_moving_bodies + args.num_static_bodies

   moving_colliders = generate_colliders(moving_body_ids, args.shapes)
   moving_bodies = generate_bodies(moving_body_ids, False, args)

   constant_forces: List[data_forceConstant_t] = []
   for body in moving_bodies:
      constant_forces.append(
         data_forceConstant_t(
            body.id,
            data_vector3_t([0.0, 0.0, 0.0]),
            data_vector3_t([0.0, 0.0, -9.8]),
            1
         )
      )

   static_colliders = generate_colliders(static_body_ids)
   static_bodies = generate_bodies(static_body_ids, True, args)

   scenario = data_scenario_t(
      numBodies=num_bodies,
      numIsometricColliders=len(moving_colliders) + len(static_colliders),
      numShapes=num_bodies,
      numConstantForces=len(constant_forces),
      constantForces=constant_forces,
      isometricColliders=moving_colliders + static_colliders,
      bodies=moving_bodies + static_bodies,
   )

   generate_shapes(scenario)

   if args.add_floor:
      add_floor(scenario, args)

   recalculate_moments(scenario)

   with open(args.scenario_filename, "w") as file_out:
      if (args.indent):
         json.dump(scenario, file_out, default=lambda o: o.__dict__, indent=3)
      else:
         json.dump(scenario, file_out, default=lambda o: o.__dict__)

def generate_viz_file(args):
   pass

def main():
   parser = argparse.ArgumentParser(description="Randomized scenario generator")

   parser.add_argument("--scenario_filename", default="new_scenario.json")

   parser.add_argument("--viz_config_filename", default="new_viz_config.json")

   parser.add_argument(
      "--shapes",
      help="The types of shapes to include in the moving bodies",
      type=int,
      required=False,
      default=None,
      nargs='*',
      choices=[
         ShapeTypes.CUBE.value,
         ShapeTypes.SPHERE.value,
         ShapeTypes.CAPSULE.value,
         ShapeTypes.CYLINDER.value,
      ]
   )

   parser.add_argument(
      "--add_floor",
      help="Adds a cube floor to the scenario with the dimensions of the scenario area's position distribution",
      action="store_true",
   )

   parser.add_argument(
      "--enable_logging",
      help="Enable logging from the scenario",
      action="store_true",
   )

   parser.add_argument(
      "--num_moving_bodies",
      help="The number of moving bodies to add",
      type=int,
      required=False,
      default=np.random.randint(1, 1025),
      choices=range(1, 1025),
   )

   parser.add_argument(
      "--num_static_bodies",
      help="The number of non-moving bodies to add",
      type=int,
      required=False,
      default=np.random.randint(0, 65),
      choices=range(0, 65),
   )

   parser.add_argument(
      "--indent",
      action="store_true"
   )

   ranged_vector3s: Dict[str, List[int]] = {
      "position": [-20, 20],
      "velocity": [-5, 5],
      "angular_velocity": [-10, 10],
   }

   for field_name, field_range in ranged_vector3s.items():
      for axis in ["x", "y", "z"]:
         parser.add_argument(
            f"--range_{axis}_{field_name}",
            help="Array of min and max {axis}-component {field_name} values",
            nargs=2,
            type=float,
            required=False,
            default=field_range
         )

   args = parser.parse_args()
   args.range_z_position = [0, max(ranged_vector3s["position"])]

   generate_scenario_file(args)

   generate_viz_file(args)

if __name__ == "__main__":
   main()
