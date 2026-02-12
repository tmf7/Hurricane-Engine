#version 450

/*
// for image read/write of a storage image in GLSL fragment shader:
layout (binding = 0, rgb8) uniform readonly image2D inputImage;
layout (binding = 1, rgb8) uniform writeonly image2D outputImage;
*/

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
	outColor = texture(texSampler, fragTexCoord);
	/*
	// for image read/write of a storage image in GLSL fragment shader:
	vec3 pixel = imageLoad(inputImage, ivec2(gl_GlobalInvocationID.xy)).rgb;
	imageStore(outputImage, ivec2(gl_GlobalInvocationID.xy), pixel);
	*/
}