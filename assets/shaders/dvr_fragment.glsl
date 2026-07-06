#version 460 core
// Direct volume rendering by ray marching a 3D scalar texture, with the built-in
// 1D colormap array as the transfer function. Back faces of the proxy cube are
// rasterized (front-face culled); the ray/box segment is computed analytically so
// entry/exit do not depend on which face was rasterized. Output is premultiplied
// alpha, to be composited with glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA).
in vec3 vWorldPos;

uniform mat4 viewMat;
uniform vec3 volMin;
uniform vec3 volMax;
uniform sampler3D volumeTex;
uniform sampler1DArray colorMaps1Darray;
uniform int colorMap;
uniform float scalarMin;
uniform float scalarMax;
uniform float densityScale;
uniform int numSteps;

out vec4 FragColor;

void main()
{
    vec3 camPos = (inverse(viewMat) * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec3 rayDir = normalize(vWorldPos - camPos);

    // Ray/box intersection (slab method); handles rayDir components of 0 via +/-inf.
    vec3 invD = 1.0 / rayDir;
    vec3 t0s = (volMin - camPos) * invD;
    vec3 t1s = (volMax - camPos) * invD;
    vec3 tsmaller = min(t0s, t1s);
    vec3 tbigger = max(t0s, t1s);
    float tNear = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
    float tFar = min(min(tbigger.x, tbigger.y), tbigger.z);
    tNear = max(tNear, 0.0);
    if (tNear >= tFar)
        discard;

    int steps = max(numSteps, 1);
    float dt = (tFar - tNear) / float(steps);
    vec3 extent = volMax - volMin;
    float range = max(scalarMax - scalarMin, 1e-8);

    vec3 accumC = vec3(0.0);
    float accumA = 0.0;
    for (int i = 0; i < steps; ++i)
    {
        float t = tNear + (float(i) + 0.5) * dt;
        vec3 p = camPos + t * rayDir;
        vec3 uvw = (p - volMin) / extent;
        float s = texture(volumeTex, uvw).r;
        float sn = clamp((s - scalarMin) / range, 0.0, 1.0);
        vec3 col = texture(colorMaps1Darray, vec2(sn, float(colorMap))).rgb;
        float a = clamp(sn * densityScale * dt, 0.0, 1.0);   // opacity ~ value * density * segment length
        accumC += (1.0 - accumA) * a * col;                  // front-to-back, premultiplied
        accumA += (1.0 - accumA) * a;
        if (accumA > 0.99)
            break;
    }
    if (accumA < 0.003)
        discard;
    FragColor = vec4(accumC, accumA);
}
