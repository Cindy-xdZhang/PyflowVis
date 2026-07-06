#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 projMat;
uniform mat4 viewMat;
uniform mat4 modelMat;

out vec3 vNormalWS;

void main()
{
    vec4 worldPos = modelMat * vec4(aPos, 1.0);
    gl_Position = projMat * viewMat * worldPos;
    vNormalWS = mat3(modelMat) * aNormal;
}
