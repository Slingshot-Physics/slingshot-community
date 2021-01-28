#include "component_widget_interface.hpp"

#include "imgui.h"

#include <string>

void IComponentWidget::bodyIdComboBox(
   const std::string & combo_box_label,
   const trecs::uid_t excluded_body_uid,
   const std::unordered_set<trecs::uid_t> & rigid_body_entities,
   trecs::uid_t & current_body_uid
) const
{
   char preview_uid_chars[32];
   snprintf(preview_uid_chars, 32, "body id %i", static_cast<int>(current_body_uid));

   viz::color_t body_color = (
      (current_body_uid != -1) ? *allocator_.getComponent<viz::color_t>(current_body_uid) : gray
   );

   ImGui::PushStyleColor(
      ImGuiCol_Text,
      ImVec4(body_color[0], body_color[1], body_color[2], body_color[3])
   );

   if (ImGui::BeginCombo(combo_box_label.c_str(), preview_uid_chars))
   {
      ImGui::PopStyleColor();

      for (const auto body_uid : rigid_body_entities)
      {
         if (body_uid == excluded_body_uid)
         {
            continue;
         }

         const bool is_selected = (current_body_uid == body_uid);

         std::string body_id_string("body id ");
         body_id_string += std::to_string(body_uid);

         if (ImGui::Selectable(body_id_string.c_str(), is_selected))
         {
            current_body_uid = body_uid;
         }

         if (is_selected)
         {
            ImGui::SetItemDefaultFocus();
         }
      }

      const bool is_selected = (current_body_uid == -1);

      std::string body_id_string("body id ");
      body_id_string += std::to_string(-1);

      if (ImGui::Selectable(body_id_string.c_str(), is_selected))
      {
         current_body_uid = -1;
      }

      if (is_selected)
      {
         ImGui::SetItemDefaultFocus();
      }

      ImGui::EndCombo();
   }
   else
   {
      ImGui::PopStyleColor();
   }
}

geometry::types::isometricTransform_t IComponentWidget::getTransform(
   trecs::uid_t body_entity
)
{
   oy::types::rigidBody_t * body_a = allocator_.getComponent<oy::types::rigidBody_t>(body_entity);

   geometry::types::isometricTransform_t trans_A_to_W = {
      identityMatrix(), {0.f, 0.f, 0.f}
   };

   if (body_a != nullptr)
   {
      trans_A_to_W.rotate = body_a->ql2b.rotationMatrix().transpose();
      trans_A_to_W.translate = body_a->linPos;
   }

   return trans_A_to_W;
}
