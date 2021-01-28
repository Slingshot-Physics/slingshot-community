#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in vec3 aNormal;

// Interface block
out VS_OUT
{
   vec3 FragPos;
   vec3 Normal;
   vec3 Color;
   vec4 FragPosLightSpace;
} vs_out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat4 lightSpaceMatrix;

void main()
{
   vs_out.Color = vec3(aColor);
   vs_out.FragPos = vec3(model * vec4(aPos, 1.0));
   vs_out.Normal = transpose(inverse(mat3(model))) * aNormal;
   // Calculates the fragment location in the directional light's frame
   vs_out.FragPosLightSpace = lightSpaceMatrix * vec4(vs_out.FragPos, 1.0);
   gl_Position = projection * view * vec4(vs_out.FragPos, 1.0);
}
