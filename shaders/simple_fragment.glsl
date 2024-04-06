#version 460 core
out vec4 FragColor;

in vec3 pos;




vec4 colorMap(vec3 pos){
	float domainSizeRadial = 10;
	float domainSizeVertical = 5;
	float radial = sqrt(pos.x*pos.x + pos.y*pos.y);
	float vertical = pos.z;
	radial /= domainSizeRadial;
	vertical /= domainSizeVertical;

	vertical = min(1.f, max(-1.f, vertical));
	radial = min(1.f, max(-1.f, radial));
	
	vec3 colorPositive = vec3(1.f,0.f,0.f);  
	vec3 colorZero = vec3(1.f,1.f,1.f);  
	vec3 colorNegative = vec3(0.f,0.f,1.f); 
	vec3 colorVertical;
	if(vertical<0){
		float interpolationWeight = -vertical;
		colorVertical = colorNegative * (interpolationWeight) + colorZero * (1.f-interpolationWeight);
	} else {
		float interpolationWeight = vertical;
		colorVertical = colorPositive * (interpolationWeight) + colorZero * (1.f-interpolationWeight);
	}
	
	float radialNormalized = 0.5f+ 0.5f*radial;
	return vec4(colorVertical*(1.f-radialNormalized),1.f);
};

void main()
{

     FragColor = colorMap(pos);
    
}
