from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

class Frame(Enum):
   BODY = 0
   GLOBAL = 1

@dataclass
class shapeCapsule:
   radius: float = 1
   height: float = 2.0
   _typename: str = "data_shapeCapsule_t"

@dataclass
class shapeCylinder:
   radius: float = 1.0
   height: float = 2.0
   _typename: str = "data_shapeCylinder_t"

@dataclass
class shapeSphere:
   radius: float = 1.0
   _typename: str = "data_shapeSphere_t"

@dataclass
class shapeCube:
   length: float = 2.0
   width: float = 2.0
   height: float = 2.0
   _typename: str = "data_shapeCube_t"

class ShapeTypes(Enum):
   CUBE = 0
   CYLINDER = 1
   SPHERE = 4
   CAPSULE = 7

@dataclass
class data_shape_t:
   shapeType: int
   bodyId: int = -1
   _typename: str = "data_shape_t"

def inertia(shape: data_shape_t) -> List[float]:
   inertia_diag = [1.0, 1.0, 1.0]

   if shape.shapeType == ShapeTypes.CUBE.value:
      if not hasattr(shape, "cube"):
         print("no cube attr")
         return inertia_diag
      inertia_diag[0] = (1 / 12) * (shape.cube.width ** 2 + shape.cube.height ** 2)
      inertia_diag[1] = (1 / 12) * (shape.cube.length ** 2 + shape.cube.height ** 2)
      inertia_diag[2] = (1 / 12) * (shape.cube.width ** 2 + shape.cube.length ** 2)
   elif shape.shapeType == ShapeTypes.SPHERE.value:
      if not hasattr(shape, "sphere"):
         print("no sphere attr")
         return inertia_diag
      for i in range(3):
         inertia_diag[i] = (2 / 5) * (shape.sphere.radius ** 2)
   elif shape.shapeType == ShapeTypes.CAPSULE.value:
      if not hasattr(shape, "capsule"):
         print("no capsule attr")
         return inertia_diag
      r2: float = shape.capsule.radius ** 2
      h2: float = shape.capsule.height ** 2
      rh: float = shape.capsule.radius * shape.capsule.height
      inertia_diag[0] = 2 * ((r2 / 10) + (h2 / 4) + (3 * rh / 16)) + \
         (1 / 12) * (3 * r2 + h2)
      inertia_diag[1] = inertia_diag[0]
      inertia_diag[2] = (7 / 10) * r2
   elif shape.shapeType == ShapeTypes.CYLINDER.value:
      if not hasattr(shape, "cylinder"):
         print("no cylinder attr")
         return inertia_diag
      r2: float = shape.cylinder.radius ** 2
      h2: float = shape.cylinder.height ** 2
      inertia_diag[0] = (1 / 12) * (3 * r2 + h2)
      inertia_diag[1] = inertia_diag[0]
      inertia_diag[2] = r2 / 2

   return inertia_diag

@dataclass
class data_quaternion_t:
   q: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
   _typename: str = "data_quaternion_t"

@dataclass
class data_vector4_t:
   v: List[float]
   _typename: str = "data_vector4_t"

@dataclass
class data_vector3_t:
   v: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
   _typename: str = "data_vector3_t"

@dataclass
class data_matrix33_t:
   m: List[float]
   _typename: str = "data_matrix33_t"

@dataclass
class data_loggerConfig_t:
   logDir: str = "."
   loggingType: int = 0
   _typename: str = "data_loggerConfig_t"

@dataclass
class data_rigidbody_t:
   id: int
   J: data_matrix33_t
   linpos: data_vector3_t
   linvel: data_vector3_t
   orientation: data_quaternion_t
   rotvel: data_vector3_t
   stationary: int
   mass: float
   gravity: data_vector3_t = data_vector3_t([0.0, 0.0, -9.8])
   _typename: str = "data_rigidbody_t"

@dataclass
class data_isometricTransform_t:
   rotate: data_matrix33_t
   translate: data_vector3_t
   _typename: str = "data_isometricTransform_t"

@dataclass
class data_transform_t:
   scale: data_matrix33_t
   rotate: data_matrix33_t
   translate: data_vector3_t
   _typename: str = "data_transform_t"

@dataclass
class data_isometricCollider_t:
   bodyId: int
   shapeName: int
   restitution: float
   mu: float
   enabled: int
   _typename: str = "data_isometricCollider_t"

@dataclass
class data_forceVelocityDamper_t:
   parentId: int
   childId: int
   parentLinkPoint: data_vector3_t
   childLinkPoint: data_vector3_t
   damperCoeff: float
   _typename: str = "data_forceVelocityDamper_t"

@dataclass
class data_forceSpring_t:
   parentId: int
   childId: int
   parentLinkPoint: data_vector3_t
   childLinkPoint: data_vector3_t
   restLength: float
   springCoeff: float
   _typename: str = "data_forceSpring_t"

@dataclass
class data_forceConstant_t:
   childId: int
   childLinkPoint: data_vector3_t
   acceleration: data_vector3_t
   frame: Frame
   _typename: str = "data_forceConstant_t"

@dataclass
class data_forceDrag_t:
   childId: int
   linearDragCoeff: float
   quadraticDragCoeff: float
   _typename: str = "data_forceDrag_t"

@dataclass
class data_torqueDrag_t:
   childId: int
   linearDragCoeff: float
   quadraticDragCoeff: float
   _typename: str = "data_torqueDrag_t"

@dataclass
class data_scenario_t:
   numBodies: int
   bodies: List[data_rigidbody_t]
   numIsometricColliders: int = 0
   numShapes: int = 0
   numBalljointConstraints: int = 0
   numGearConstraints: int = 0
   numRevoluteJointConstraints: int = 0
   numRevoluteMotorConstraints: int = 0
   numRotation1dConstraints: int = 0
   numTranslation1dConstraints: int = 0
   numConstantForces: int = 0
   numDragForces: int = 0
   numSpringForces: int = 0
   numDragTorques: int = 0
   numVelocityDamperForces: int = 0
   isometricColliders: List[data_isometricCollider_t] = field(default_factory=lambda: [])
   shapes: List[data_shape_t] = field(default_factory=lambda: [])
   balljointConstraints: List = field(default_factory=lambda: [])
   gearConstraints: List = field(default_factory=lambda: [])
   revoluteJointConstraints: List = field(default_factory=lambda: [])
   revoluteMotorConstraints: List = field(default_factory=lambda: [])
   rotation1dConstraints: List = field(default_factory=lambda: [])
   translation1dConstraints: List = field(default_factory=lambda: [])
   constantForces: List[data_forceConstant_t] = field(default_factory=lambda: [])
   dragForces: List[data_forceDrag_t] = field(default_factory=lambda: [])
   springForces: List[data_forceSpring_t] = field(default_factory=lambda: [])
   velocityDamperForces: List[data_forceVelocityDamper_t] = field(default_factory=lambda: [])
   dragTorques: List[data_torqueDrag_t] = field(default_factory=lambda: [])
   logger: data_loggerConfig_t = data_loggerConfig_t()
   _typename: str = "data_scenario_t"

@dataclass
class data_vizMeshProperties_t:
   bodyId: int
   color: data_vector4_t
   _typename: str ="data_vizMeshProperties_t"

@dataclass
class data_vizConfig_t:
   numMeshProps: int

   maxFps: int = 60
   realtime: int = 0
   windowWidth: int = 800
   windowWidth: int = 600
   cameraPos: data_vector3_t = data_vector3_t([5, 15, 5])
   cameraPoint: data_vector3_t = data_vector3_t([0, 5, 0])
   mousePick: int = 1
   _typename: str = "data_vizConfig_t"
