#version 460 core
// 3D flowline geometry stage (ported from optimal-connection illuminatedLines3DGeometry.glsl):
// expands each adjacency segment into a round tube of TUBE_SLICES faces with per-vertex normals,
// built in VIEW space. This is what makes 3D flowlines read as real 3D tubes (vs the 2D
// screen-facing ribbon).
layout(lines_adjacency) in;
#define TUBE_SLICES 12
layout(triangle_strip, max_vertices = 26) out;   // 2 * (TUBE_SLICES + 1)

uniform mat4 projMat;
uniform float lineWeight;   // tube DIAMETER in view/world units (radius = 0.5 * lineWeight)

in VS_OUT {
    vec3 viewPos;
    float attrib;
    float attrib2;
} gs_in[];

out vec3 vViewPos;
out vec3 vNormal;
out float vAttrib;
out float vAttrib2;

const float PI = 3.14159265358979323846;

vec3 safeNormalize(vec3 v)
{
    float l2 = dot(v, v);
    if (l2 < 1e-20) return vec3(0.0, 0.0, 1.0);
    return v * inversesqrt(l2);
}

// Camera-stable tube frame: pick a reference axis from the view direction toward the camera.
void buildFrame(in vec3 T, in vec3 Pview, out vec3 N, out vec3 B)
{
    vec3 V = safeNormalize(-Pview);
    vec3 ref = (abs(dot(T, V)) < 0.95) ? V : vec3(0.0, 1.0, 0.0);
    N = safeNormalize(cross(T, ref));
    if (dot(N, N) < 1e-10) N = safeNormalize(cross(T, vec3(1.0, 0.0, 0.0)));
    B = safeNormalize(cross(T, N));
}

void main()
{
    float radius = 0.5 * lineWeight;

    vec3 Pm1 = gl_in[0].gl_Position.xyz;
    vec3 P0  = gl_in[1].gl_Position.xyz;   // segment start
    vec3 P1  = gl_in[2].gl_Position.xyz;   // segment end
    vec3 P2  = gl_in[3].gl_Position.xyz;

    vec3 dThis = P1 - P0;
    if (dot(dThis, dThis) < 1e-20)
        return;

    vec3 T0 = safeNormalize((P0 - Pm1) + dThis);
    vec3 T1 = safeNormalize(dThis + (P2 - P1));
    if (dot(T0, T0) < 1e-10) T0 = safeNormalize(dThis);
    if (dot(T1, T1) < 1e-10) T1 = safeNormalize(dThis);

    vec3 N0, B0; buildFrame(T0, P0, N0, B0);
    vec3 N1, B1; buildFrame(T1, P1, N1, B1);

    for (int i = 0; i <= TUBE_SLICES; ++i)
    {
        float a = 2.0 * PI * float(i) / float(TUBE_SLICES);
        float ca = cos(a);
        float sa = sin(a);
        vec3 dir0 = ca * N0 + sa * B0;
        vec3 dir1 = ca * N1 + sa * B1;

        // start ring vertex
        vViewPos = P0 + radius * dir0;
        vNormal  = dir0;
        vAttrib  = gs_in[1].attrib;
        vAttrib2 = gs_in[1].attrib2;
        gl_Position = projMat * vec4(vViewPos, 1.0);
        EmitVertex();

        // end ring vertex
        vViewPos = P1 + radius * dir1;
        vNormal  = dir1;
        vAttrib  = gs_in[2].attrib;
        vAttrib2 = gs_in[2].attrib2;
        gl_Position = projMat * vec4(vViewPos, 1.0);
        EmitVertex();
    }
    EndPrimitive();
}
