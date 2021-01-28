#include "shape_widgets.hpp"

#include "imgui.h"

#include "inertia.hpp"

#include <algorithm>

bool cube_widget(geometry::types::shape_t & shape)
{
   bool widget_modified = false;
   ImGui::TableNextRow();
   ImGui::TableSetColumnIndex(0);
   ImGui::Text("cube parameters");

   ImGui::TableNextRow();
   ImGui::TableSetColumnIndex(0);
   ImGui::Text("length");
   ImGui::TableSetColumnIndex(1);
   ImGui::PushItemWidth(-1);
   widget_modified |= ImGui::DragFloat("##1length", &shape.cube.length, 0.1f, 0.1f, 128.f);
   ImGui::PopItemWidth();


   ImGui::TableNextRow();
   ImGui::TableSetColumnIndex(0);
   ImGui::Text("width");
   ImGui::TableSetColumnIndex(1);
   ImGui::PushItemWidth(-1);
   widget_modified |= ImGui::DragFloat("##2width", &shape.cube.width, 0.1f, 0.1f, 128.f);
   ImGui::PopItemWidth();


   ImGui::TableNextRow();
   ImGui::TableSetColumnIndex(0);
   ImGui::Text("height");
   ImGui::TableSetColumnIndex(1);
   ImGui::PushItemWidth(-1);
   widget_modified |= ImGui::DragFloat("##3height", &shape.cube.height, 0.1f, 0.1f, 128.f);
   ImGui::PopItemWidth();

   shape.cube.length = std::max(shape.cube.length, 0.1f);
   shape.cube.width = std::max(shape.cube.width, 0.1f);
   shape.cube.height = std::max(shape.cube.height, 0.1f);

   return widget_modified;
}

bool sphere_widget(geometry::types::shape_t & shape)
{
   bool widget_modified = false;
   ImGui::TableNextRow();
   ImGui::TableSetColumnIndex(0);
   ImGui::Text("sphere params");

   ImGui::TableNextRow();
   ImGui::TableSetColumnIndex(0);
   ImGui::Text("radius");
   ImGui::TableSetColumnIndex(1);
   ImGui::PushItemWidth(-1);
   widget_modified |= ImGui::DragFloat("##1radius", &shape.sphere.radius, 0.1f, 0.1f, 128.f);
   ImGui::PopItemWidth();

   return widget_modified;
}

bool capsule_widget(geometry::types::shape_t & shape)
{
   bool widget_modified = false;
   ImGui::TableNextRow();
   ImGui::TableSetColumnIndex(0);
   ImGui::Text("capsule params");

   ImGui::TableNextRow();
   ImGui::TableSetColumnIndex(0);
   ImGui::Text("radius");
   ImGui::TableSetColumnIndex(1);
   ImGui::PushItemWidth(-1);
   widget_modified |= ImGui::DragFloat("##2radius", &shape.capsule.radius, 0.1f, 0.1f, 128.f);
   ImGui::PopItemWidth();


   ImGui::TableNextRow();
   ImGui::TableSetColumnIndex(0);
   ImGui::Text("height");
   ImGui::TableSetColumnIndex(1);
   ImGui::PushItemWidth(-1);
   widget_modified |= ImGui::DragFloat("##2height", &shape.capsule.height, 0.1f, 0.1f, 128.f);
   ImGui::PopItemWidth();

   shape.capsule.height = std::max(shape.capsule.radius, shape.capsule.height);

   return widget_modified;
}

bool cylinder_widget(geometry::types::shape_t & shape)
{
   bool widget_modified = false;
   ImGui::TableNextRow();
   ImGui::TableSetColumnIndex(0);
   ImGui::Text("cylinder params");

   ImGui::TableNextRow();
   ImGui::TableSetColumnIndex(0);
   ImGui::Text("radius");
   ImGui::TableSetColumnIndex(1);
   ImGui::PushItemWidth(-1);
   widget_modified |= ImGui::DragFloat("##3radius", &shape.cylinder.radius, 0.1f, 0.1f, 128.f);
   ImGui::PopItemWidth();

   ImGui::TableNextRow();
   ImGui::TableSetColumnIndex(0);
   ImGui::Text("length");
   ImGui::TableSetColumnIndex(1);
   ImGui::PushItemWidth(-1);
   widget_modified |= ImGui::DragFloat("##3height", &shape.cylinder.height, 0.1f, 0.1f, 128.f);
   ImGui::PopItemWidth();

   return widget_modified;
}

Vector3 inertia(const geometry::types::shape_t & shape)
{
   Vector3 result;
   Matrix33 inertia_tensor = geometry::inertiaTensor(shape);

   result[0] = inertia_tensor(0, 0);
   result[1] = inertia_tensor(1, 1);
   result[2] = inertia_tensor(2, 2);

   return result;
}
