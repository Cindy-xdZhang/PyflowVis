#version 460 core
// DVR proxy-cube vertex stage (aligned with optimal-connection src/shaders/DVR.vert).
// The proxy geometry is a unit cube in [0,1]^3; it is mapped to the volume's object-space
// bounding box and then to world space by modelMat. Only the world position is passed on;
// the fragment stage recovers the object-space ray via inverse(modelMat), so the ray/box
// intersection is done in object space exactly like the C++ engine.
layout (location = 0) in vec3 aPos;   // unit-cube corner in [0,1]^3

uniform mat4 projMat;
uniform mat4 viewMat;
uniform mat4 modelMat;
uniform vec3 uVolumeBoundsMin;
uniform vec3 uVolumeBoundsMax;

out vec3 vWorldPos;

void main()
{
    vec3 objectPos = uVolumeBoundsMin + aPos * (uVolumeBoundsMax - uVolumeBoundsMin);
    vec4 worldPos  = modelMat * vec4(objectPos, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projMat * viewMat * worldPos;
}
