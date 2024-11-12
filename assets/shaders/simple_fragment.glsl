#version 460 core
out vec4 FragColor;

uniform vec3 color;


in vec3  vPos;

void main()
{

    FragColor = vec4(color,1);
    
}
