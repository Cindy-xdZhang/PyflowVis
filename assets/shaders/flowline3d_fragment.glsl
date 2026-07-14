#version 460 core
// 3D flowline fragment stage (ported from optimal-connection illuminatedLines3d.frag):
// colormap color from the selected attribute, with a view-space headlight so the round tube reads
// as 3D (the Python engine has no scene-light uniforms, so the light sits at the camera).
in vec3 vViewPos;
in vec3 vNormal;
in float vAttrib;    // normalized time
in float vAttrib2;   // selected source attribute

uniform int ColorCodingAttribute;              // 0 -> attrib (time), 1 -> attrib2
uniform int colorMap;                          // colormap-array layer
uniform sampler1DArray colorMaps1Darray;       // provided by the Material (texture0="builtIn")

out vec4 FragColor;

void main()
{
    float attr = (ColorCodingAttribute == 1) ? vAttrib2 : vAttrib;
    vec3 base = texture(colorMaps1Darray, vec2(clamp(attr, 0.0, 1.0), float(colorMap))).rgb;

    // Headlight in view space: the camera is the view-space origin, so the light direction toward
    // it is normalize(-viewPos). Two-sided so the far side of the tube stays lit.
    vec3 N = normalize(vNormal);
    vec3 L = normalize(-vViewPos);
    float diff = abs(dot(N, L));
    float ambient = 0.35;
    vec3 color = base * (ambient + 0.75 * diff);

    FragColor = vec4(color, 1.0);
}
