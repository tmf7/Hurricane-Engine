#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 fragColor;

void main() {
	gl_PointSize = 5.0f; // required if VK_PRIMITIVE_TOPOLOGY_POINT_LIST is set in VkPipelineInputAssemblyStateCreateInfo
	gl_Position = vec4(inPosition, 0.0f,  1.0);
	fragColor = inColor;
}