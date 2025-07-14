#version 460 core
in vec3 position;
in vec3 tangent;
in float attrib;
in float attrib2;
in float opacity;

uniform sampler1D colorMap;
out vec4 FragColor;

void main() {
    vec3 color = texture(colorMap, attrib).rgb;
    FragColor = vec4(color, opacity);
}