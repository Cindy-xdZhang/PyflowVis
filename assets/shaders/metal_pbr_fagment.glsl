#version 460 core
out vec4 FragColor;

in vec3 vPos;
in vec2 vTexCoord;
in vec3 vNormal;

uniform sampler2D albedoMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;

uniform vec3 cameraPos;

void main()
{
    vec3 albedo = texture(albedoMap, vTexCoord).rgb;
    float metallic = texture(metallicMap, vTexCoord).r;
    float roughness = texture(roughnessMap, vTexCoord).r;
    float ao = texture(aoMap, vTexCoord).r;

    vec3 N = normalize(vNormal);
    vec3 V = normalize(cameraPos - vPos);

    // Calculate lighting
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        vec3 L = vec3(0.0);
        float distance = length(L);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = vec3(0.0);

        // Calculate radiance
        radiance += vec3(0.3) * attenuation;

        // Calculate BRDF
        vec3 F0 = vec3(0.04);
        vec3 F = F0 + (1.0 - F0) * pow(1.0 - dot(N, V), 5.0);
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        vec3 nominator = kD * albedo / pi + kS * radiance;
        vec3 denominator = kD + kS;
        vec3 brdf = nominator / denominator;

        // Calculate Lo
        Lo += brdf * radiance;
    }

    // Calculate final color
    vec3 color = Lo * albedo;
    color *= ao;

    FragColor = vec4(color, 1.0);
}s