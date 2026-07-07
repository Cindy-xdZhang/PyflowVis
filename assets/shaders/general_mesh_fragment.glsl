#version 460 core
// General mesh material (ported from optimal-connection BlingPhoneColorMap.frag).
// Color: baseColor (rgba) when colorMap < 0, else the built-in 1D colormap array sampled at uv.x.
// Lighting: Blinn-Phong with a headlight (light co-located with the camera) because this Python
// engine has no scene-light uniforms; two-sided so open meshes stay lit from either side.
in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec2 vUV;

uniform sampler1DArray colorMaps1Darray;   // provided by the Material (texture0="builtIn")
uniform int   colorMap;                     // >=0: colormap layer by uv.x; <0: baseColor
uniform vec4  baseColor;                    // rgba
uniform vec3  camPos;                       // world-space camera position

out vec4 FragColor;

void main()
{
    vec3 albedo = baseColor.rgb;
    if (colorMap >= 0)
        albedo = texture(colorMaps1Darray, vec2(clamp(vUV.x, 0.0, 1.0), float(colorMap))).rgb;

    vec3 N = normalize(vWorldNormal);
    vec3 V = normalize(camPos - vWorldPos);
    if (dot(N, V) < 0.0) N = -N;            // two-sided
    vec3 L = V;                              // headlight
    vec3 H = normalize(L + V);

    const float ambient = 0.2;
    const float kd = 0.7;
    const float ks = 0.4;
    const float shininess = 64.0;
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), shininess);

    vec3 color = albedo * (ambient + kd * diff) + vec3(ks * spec);
    FragColor = vec4(color, baseColor.a);
}
