#version 460 core
layout (location = 0) in vec3 aPos;   // unit-cube corner in [0,1]^3 (proxy geometry)

uniform mat4 projMat;
uniform mat4 viewMat;
uniform mat4 modelMat;
uniform vec3 volMin;
uniform vec3 volMax;

out vec3 vWorldPos;

void main()
{
    vec3 worldPos = volMin + aPos * (volMax - volMin);
    vWorldPos = worldPos;
    gl_Position = projMat * viewMat * modelMat * vec4(worldPos, 1.0);
}
