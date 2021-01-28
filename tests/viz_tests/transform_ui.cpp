#include "transform_ui.hpp"

#include "attitudeutils.hpp"

int TransformUi::counter_ = 0;

bool TransformUi::no_window(void)
{
   bool ui_modified = false;
   ImGui::SetNextItemOpen(true);
   if (ImGui::TreeNode(name_.c_str()))
   {
      bool pos_changed = ImGui::DragFloat3(
         "pos", &(trans_B_to_W_.translate[0]), 0.025f, -10.f, 10.f, "%0.7f"
      );

      Vector3 rpy = frd2NedMatrixToRpy(trans_B_to_W_.rotate);
      bool rpy_changed = ImGui::DragFloat3(
         "roll, pitch, yaw", &(rpy[0]), 0.025f, -1.f * M_PI, M_PI, "%0.7f"
      );

      if (rpy_changed)
      {
         rpy[1] = std::max(
            std::min(rpy[1], (float )M_PI/2.f),
            (float )-M_PI/2.f
         );
         trans_B_to_W_.rotate = frd2NedMatrix(rpy);
      }

      bool scale_changed = false;
      Vector3 scale;
      if (force_diagonal_scale_)
      {
         float scalar_scale = trans_B_to_W_.scale(0, 0);
         scale_changed = ImGui::DragFloat(
            "scale", &(scalar_scale), 0.025f, 0.001f, 10.f, "%0.7f"
         );
         scale.Initialize(
            scalar_scale,
            scalar_scale,
            scalar_scale
         );
      }
      else
      {
         scale.Initialize(
            trans_B_to_W_.scale(0, 0),
            trans_B_to_W_.scale(1, 1),
            trans_B_to_W_.scale(2, 2)
         );
         scale_changed = ImGui::DragFloat3(
            "scale", &(scale[0]), 0.025f, 0.001f, 10.f, "%0.7f"
         );
      }

      if (scale_changed)
      {
         for (int i = 0; i < 3; ++i)
         {
            trans_B_to_W_.scale(i, i) = scale[i];
         }
      }

      ui_modified = scale_changed | rpy_changed | pos_changed;

      ImGui::TreePop();
   }

   return ui_modified;
}
