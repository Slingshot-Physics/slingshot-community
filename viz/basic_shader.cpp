#include "basic_shader.hpp"

BasicShader::BasicShader(
   const char * vShaderLoc, const char * fShaderLoc
)
{
   std::cout << "loading vertex shader from " << vShaderLoc << "\n";
   std::cout << "loading fragment shader from " << fShaderLoc << "\n";
   unsigned int vShaderId = compileShader(
      vShaderLoc, GL_VERTEX_SHADER
   );

   unsigned int fShaderId = compileShader(
      fShaderLoc, GL_FRAGMENT_SHADER
   );

   id = glCreateProgram();
   glAttachShader(id, vShaderId);
   glAttachShader(id, fShaderId);
   glLinkProgram(id);

   glDeleteShader(vShaderId);
   glDeleteShader(fShaderId);
}

BasicShader::BasicShader(
   const char * vShaderLoc, const char * gShaderLoc, const char * fShaderLoc
)
{
   unsigned int vShaderId = compileShader(
      vShaderLoc, GL_VERTEX_SHADER
   );

   unsigned int fShaderId = compileShader(
      fShaderLoc, GL_FRAGMENT_SHADER
   );

   unsigned int gShaderId = compileShader(
      gShaderLoc, GL_GEOMETRY_SHADER
   );
   id = glCreateProgram();
   glAttachShader(id, vShaderId);
   glAttachShader(id, gShaderId);
   glAttachShader(id, fShaderId);
   glLinkProgram(id);

   glDeleteShader(vShaderId);
   glDeleteShader(fShaderId);
   glDeleteShader(gShaderId);
}

void BasicShader::use(void) const
{
   glUseProgram(id);
}

void BasicShader::setUniformFloat(const std::string & name, const float & val)
{
   unsigned int uniformLoc = getUniformLocation(name);
   glUniform1fv(uniformLoc, 1, &val);
}

void BasicShader::setUniformVec2(const std::string & name, const glm::vec2 & vec)
{
   unsigned int uniformLoc = getUniformLocation(name);
   glUniform2fv(uniformLoc, 1, &(vec[0]));
}

void BasicShader::setUniformVec3(const std::string & name, const glm::vec3 & vec)
{
   unsigned int uniformLoc = getUniformLocation(name);
   glUniform3fv(uniformLoc, 1, &(vec[0]));
}

void BasicShader::setUniformVec4(const std::string & name, const glm::vec4 & vec)
{
   unsigned int uniformLoc = getUniformLocation(name);
   glUniform4fv(uniformLoc, 1, &(vec[0]));
}

void BasicShader::setUniformMatrix2(const std::string & name, const glm::mat2 & mat)
{
   unsigned int uniformLoc = getUniformLocation(name);
   glUniformMatrix2fv(uniformLoc, 1, GL_FALSE, &(mat[0][0]));
}

void BasicShader::setUniformMatrix3(const std::string & name, const glm::mat3 & mat)
{
   unsigned int uniformLoc = getUniformLocation(name);
   glUniformMatrix3fv(uniformLoc, 1, GL_FALSE, &(mat[0][0]));
}

void BasicShader::setUniformMatrix4(const std::string & name, const glm::mat4 & mat)
{
   unsigned int uniformLoc = getUniformLocation(name);
   glUniformMatrix4fv(uniformLoc, 1, GL_FALSE, &(mat[0][0]));
}
