#ifndef SHAPE_WIDGETS_HEADER
#define SHAPE_WIDGETS_HEADER

#include "geometry_types.hpp"

bool cube_widget(geometry::types::shape_t & shape);

bool sphere_widget(geometry::types::shape_t & shape);

bool capsule_widget(geometry::types::shape_t & shape);

bool cylinder_widget(geometry::types::shape_t & shape);

// Returns the diagonal from the moment of inertia tensor of the given shape
// type, per unit mass. (e.g. multiply the result by the body's mass to get
// the actual moment of inertia tensor's diagonal).
Vector3 inertia(const geometry::types::shape_t & shape);

#endif
