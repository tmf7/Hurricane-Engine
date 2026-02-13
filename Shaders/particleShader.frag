#version 450

layout(location = 0) in vec4 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
	// force points to be circular
    if (length(gl_PointCoord - vec2(0.5)) > 0.5) {
        discard;
    }
	
	outColor = fragColor;
}