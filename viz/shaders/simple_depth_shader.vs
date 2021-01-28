#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in vec3 aNormal;

uniform mat4 light_space_matrix;
uniform mat4 model;

void main()
{
   gl_Position = light_space_matrix * model * vec4(aPos, 1.0);
}
