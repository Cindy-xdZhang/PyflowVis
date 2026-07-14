#version 460 core
// 3D flowline vertex stage (ported from optimal-connection illuminated3DLinesVertex.glsl).
// KEEPS the full 3D position — the 2D flowline shader flattened z to 0 (vec3(pos2d,0) + uplifting),
// which forced every 3D streamline/pathline onto the z=0 plane. The tube is assembled in VIEW space
// by the geometry shader (so its frame stays stable relative to the camera), then projected there.
layout(location = 0) in vec3 in_position;   // full 3D position (x, y, z)
layout(location = 1) in vec2 in_attribs;    // (normalized_time, attrib2)

uniform mat4 modelMat;
uniform mat4 viewMat;
uniform mat4 projMat;   // unused here; the geometry shader applies it after building the tube

out VS_OUT {
    vec3 viewPos;
    float attrib;
    float attrib2;
} vs_out;

void main()
{
    vec4 viewPos = viewMat * modelMat * vec4(in_position, 1.0);
    gl_Position = viewPos;              // view space; geometry shader builds the tube and applies projMat
    vs_out.viewPos = viewPos.xyz;
    vs_out.attrib = in_attribs.x;
    vs_out.attrib2 = in_attribs.y;
}
