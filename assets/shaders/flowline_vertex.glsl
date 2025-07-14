#version 460 core
layout(location = 0) in vec4 in_position;
layout(location = 1) in vec2 in_attribs;


uniform mat4 modelMat;
uniform mat4 viewMat;
uniform mat4 projMat;

out VS_OUT {
    vec3 position;
    float attrib;
    float attrib2;
    float opacity;
} vs_out;

void main() {
    vs_out.position = vec3(modelMat * vec4(in_position, 1.0));
    vs_out.attrib = in_attrib;
    vs_out.attrib2 = in_attrib2;
    vs_out.opacity = in_opacity;
    gl_Position = projMat * viewMat * modelMat * vec4(in_position, 1.0);
}