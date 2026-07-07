#version 460 core
// General mesh material (ported from optimal-connection BlingPhoneColorMap.vert).
// Shared by the GeneralMeshRenderer's curves (tubes), surfaces, and prototypes (spheres/arrows).
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

uniform mat4 projMat;
uniform mat4 viewMat;
uniform mat4 modelMat;

out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec2 vUV;

void main()
{
    vec4 wp = modelMat * vec4(aPosition, 1.0);
    vWorldPos = wp.xyz;
    vWorldNormal = normalize(mat3(transpose(inverse(modelMat))) * aNormal);
    vUV = aUV;
    gl_Position = projMat * viewMat * wp;
}
