#include "camera.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>

// TODO (TF 19 MAR 2026): quick and dirty implementation of camera movement, needs upgrade
glm::mat4 Camera::GetViewMatrix()
{
    glm::mat4 cameraTranslation = glm::translate(glm::mat4{ 1.0f }, position);
    glm::mat4 cameraRotation = GetRotationMatrix();

    // move the world opposite direction to global camera, hence the inversion
    return glm::inverse(cameraTranslation * cameraRotation);
}

glm::mat4 Camera::GetRotationMatrix()
{
    glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3{ 1.0f, 0.0f, 0.0f });
    glm::quat yawRotation = glm::angleAxis(yaw, glm::vec3{ 0.0f, -1.0f, 0.0f });
    return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}

void Camera::ProcessSDLEvent(SDL_Event& event)
{
    SDL_Keycode eventKey = event.key.key;

    if (event.type == SDL_EVENT_KEY_DOWN)
    {
        if (eventKey == SDLK_W) { velocity.z = -1.0f; }
        if (eventKey == SDLK_S) { velocity.z = 1.0f; }
        if (eventKey == SDLK_A) { velocity.x = -1.0f; }
        if (eventKey == SDLK_D) { velocity.s = 1.0f; }
    }

    if (event.type == SDL_EVENT_KEY_UP)
    {
        if (eventKey == SDLK_W) { velocity.z = 0.0f; }
        if (eventKey == SDLK_S) { velocity.z = 0.0f; }
        if (eventKey == SDLK_A) { velocity.x = 0.0f; }
        if (eventKey == SDLK_D) { velocity.s = 0.0f; }
    }

    if (event.type == SDL_EVENT_MOUSE_MOTION)
    {
        yaw += event.motion.xrel / 200.0f;
        pitch -= event.motion.yrel / 200.0f;
    }
}

void Camera::Update()
{
    glm::mat4 cameraRotation = GetRotationMatrix();
    position += glm::vec3(cameraRotation * glm::vec4(velocity * 0.25f, 0.0f));
}
