#version 460 core
// Instanced arrow glyph.
// Template attributes (per-vertex, divisor 0): a unit arrow along +Z, z in [0,1], unit radius.
layout (location = 0) in vec3 aPos;      // template position
layout (location = 1) in vec3 aNormal;   // template normal
// Per-instance attributes (divisor 1):
layout (location = 2) in vec3 iPos;      // sample position in world space
layout (location = 3) in vec3 iVec;      // sampled vector (direction * speed)

uniform mat4 projMat;
uniform mat4 viewMat;
uniform mat4 modelMat;
uniform float scale;     // user overall scale
uniform float radius;    // template radius scale
uniform float height;    // arrow length per unit speed
uniform float maxSpeed;  // for speed normalization / coloring

out float vSpeedN;
out vec3 vNormalWS;

void main()
{
    float speed = length(iVec);
    vec3 dir = speed > 1e-8 ? iVec / speed : vec3(0.0, 0.0, 1.0);

    // Build an orthonormal basis whose z-axis is the vector direction.
    vec3 ref = abs(dir.z) < 0.99 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 xAxis = normalize(cross(ref, dir));
    vec3 yAxis = cross(dir, xAxis);

    float len = max(height * speed * scale, 1e-6);
    float rad = radius * scale;

    // Scale template (xy by radius, z by length) then rotate into the basis.
    vec3 local = vec3(aPos.x * rad, aPos.y * rad, aPos.z * len);
    vec3 worldPos = iPos + xAxis * local.x + yAxis * local.y + dir * local.z;

    gl_Position = projMat * viewMat * modelMat * vec4(worldPos, 1.0);

    vec3 n = xAxis * aNormal.x + yAxis * aNormal.y + dir * aNormal.z;
    vNormalWS = normalize(n);
    vSpeedN = maxSpeed > 1e-8 ? clamp(speed / maxSpeed, 0.0, 1.0) : 0.0;
}
