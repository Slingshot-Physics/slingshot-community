#version 330 core
out vec4 FragColor;

// Interface block
in VS_OUT
{
   vec3 FragPos;
   vec3 Normal;
   vec3 Color;
   vec4 FragPosLightSpace;
} fs_in;

uniform sampler2D depthMap;

// The direction of the shadow light in world coordinates. This is technically
// the negative of the direction the ray is shining.
uniform vec3 shadowLightDir;

// The direction of the ambient directional light in world coordinates. This is
// technically the negative of the direction the light is shining.
uniform vec3 diffuseLightDir;

// The color of the ambient light emitted from the diffuse light.
uniform vec3 ambientLightColor;

// The strength of the ambient light in the scene.
uniform float ambientLightStrength;

float ShadowCalculation(vec4 fragPosLightSpace, float bias)
{
   // The position of the fragment in light space, but with the perspective
   // division removed. Perspective division is only a problem if the light
   // casting a shadow isn't purely directional.
   vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;

   // Get the coordinates in the range [0, 1].
   projCoords = projCoords * 0.5 + 0.5;

   // The closest depth from the light's POV in normalized depth coordinates.
   float closestDepth = texture(depthMap, projCoords.xy).r;
   float currentDepth = projCoords.z;

   float shadow = 0.0;

   // Percentage-closest filtering
   vec2 texelSize = 1.0 / textureSize(depthMap, 0);
   for (int i = -1; i <= 1; ++i)
   {
      for (int j = -1; j <= 1; ++j)
      {
         float pcfDepth = texture(depthMap, projCoords.xy + vec2(i, j) * texelSize).r;
         shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
      }
   }

   shadow /= 9.0;

   if (projCoords.z > 1.0)
   {
      shadow = 0.0;
   }

   return shadow;
}

void main()
{
   vec3 shapeColor = fs_in.Color;
   vec3 normal = normalize(fs_in.Normal);

   vec3 diffuseLightDirUnit = normalize(diffuseLightDir);
   float diffuseStrength = max(dot(diffuseLightDirUnit, normal), 0.0);
   vec3 ambientLight = ambientLightStrength * ambientLightColor;
   vec3 diffuseLight = diffuseStrength * ambientLightColor;

   float bias = max(
      0.05 * (
         1.0 - dot(normal, normalize(shadowLightDir))
      ),
      0.005
   );

   float shadow = ShadowCalculation(fs_in.FragPosLightSpace, bias);
   vec3 lighting = (ambientLight + (1.0 - shadow) * diffuseLight) * shapeColor;

   FragColor = vec4(lighting, 1.0);
}
