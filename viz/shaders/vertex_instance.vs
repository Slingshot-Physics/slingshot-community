#version 330 core

// Vertex position relative to the center.
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in vec3 aNormal;
// The indexed center vector.
layout (location = 3) in float aOffset;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec4 vertColor;
out vec3 vertNormal;
out vec3 vertPos;

void main()
{
   gl_Position = projection * view * model * vec4(aPos + aOffset*aNormal, 1.0);
   vertColor = aColor;
   vertNormal = mat3(transpose(inverse(model))) * aNormal;
   vertPos = (model * vec4(aPos, 1.0)).xyz;
}
