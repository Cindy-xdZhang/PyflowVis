#version 460 core
in vec3 vNormalWS;

uniform vec3 color;

out vec4 FragColor;

void main()
{
    vec3 N = normalize(vNormalWS);
    // Two-sided Lambert (abs) + a soft second light so cavities stay readable without culling.
    vec3 L1 = normalize(vec3(0.5, 0.8, 0.6));
    vec3 L2 = normalize(vec3(-0.4, -0.2, -0.7));
    float diff = 0.25 + 0.60 * abs(dot(N, L1)) + 0.15 * abs(dot(N, L2));
    FragColor = vec4(abs(color) * diff, 1.0);
}
