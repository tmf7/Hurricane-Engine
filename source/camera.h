#include "vk_types.h"
#include <SDL3/SDL_events.h>

// TODO (TF 19 MAR 2026): Only store the render parameters in the "engine camera"
// then have a "game camera" responsible for updating the relevant "engine camera" params
// after it moves/whatever
class Camera {
public:
	glm::vec3 position;
	glm::vec3 velocity;
	float pitch = 0.0f;
	float yaw = 0.0f;

	glm::mat4 GetViewMatrix();
	glm::mat4 GetRotationMatrix();
	void ProcessSDLEvent(SDL_Event& event);
	void Update();
};
