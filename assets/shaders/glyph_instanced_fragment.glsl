#version 460 core
in float vSpeedN;
in vec3 vNormalWS;

uniform vec3 color;   // user color (fast end of the ramp)

out vec4 FragColor;

void main()
{
    // Two-sided Lambert (abs) so back faces stay lit without face culling.
    vec3 L = normalize(vec3(0.4, 0.7, 0.6));
    float diff = 0.35 + 0.65 * abs(dot(normalize(vNormalWS), L));

    // Speed ramp: slow -> dark blue, fast -> user color.
    vec3 slow = vec3(0.15, 0.25, 0.70);
    vec3 c = mix(slow, abs(color), vSpeedN);

    FragColor = vec4(c * diff, 1.0);
}
