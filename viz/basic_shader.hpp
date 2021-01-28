#ifndef BASIC_SHADER_HEADER
#define BASIC_SHADER_HEADER

#include <iostream>
#include <map>

// This is necessary for Mac - the latest version of GLM deprecates calls with
// radians.
#define GLM_FORCE_RADIANS

#include <glm/glm.hpp>
#include "gl_common.hpp"

#include "util.hpp"

// A simple shader class that loads in a fragment and vertex shader program,
// links them to a GL program, and keeps track of the program ID. Also provides
// several helper functions for uploading uniforms to the shader program.
class BasicShader
{
   public:

      unsigned int id;

      BasicShader(const char * vShaderLoc, const char * fShaderLoc);

      BasicShader(
         const char * vShaderLoc,
         const char * gShaderLoc,
         const char * fShaderLoc
      );

      void use(void) const;

      void setUniformFloat(const std::string & name, const float & val);

      void setUniformVec2(const std::string & name, const glm::vec2 & vec);

      void setUniformVec3(const std::string & name, const glm::vec3 & vec);

      void setUniformVec4(const std::string & name, const glm::vec4 & vec);

      void setUniformMatrix2(const std::string & name, const glm::mat2 & mat);

      void setUniformMatrix3(const std::string & name, const glm::mat3 & mat);

      void setUniformMatrix4(const std::string & name, const glm::mat4 & mat);

   private:

      std::map<const std::string, unsigned int> uniformNameHash_;

      unsigned int getUniformLocation(const std::string & uniformName)
      {
         unsigned int retVal = 0;
         if (uniformNameHash_.find(uniformName) != uniformNameHash_.end())
         {
            retVal = uniformNameHash_[uniformName];
         }
         else
         {
            retVal = glGetUniformLocation(id, uniformName.c_str());
            uniformNameHash_[uniformName] = retVal;
         }
         return retVal;
      }

      static unsigned int compileShader(
         const char * shaderLoc, GLuint64 shaderType
      )
      {
         GLint success = -1;
         char infoLog[512];
         std::string shaderSource;

         readFileToString(shaderLoc, shaderSource);
         const char * shaderSourcePtr = shaderSource.c_str();

         unsigned int shaderId = glCreateShader(shaderType);
         glShaderSource(shaderId, 1, &shaderSourcePtr, nullptr);
         glCompileShader(shaderId);

         glGetShaderiv(shaderId, GL_COMPILE_STATUS, &success);
         if (!success)
         {
            glGetShaderInfoLog(shaderId, 512, nullptr, infoLog);
            std::cout << "Error compiling shader source " << shaderLoc << std::endl;
         }

         return shaderId;
      }
};

#endif
