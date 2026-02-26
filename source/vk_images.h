#pragma once 

#include <vulkan/vulkan.h>

namespace vkutil {

	void transition_image(VkCommandBuffer cmd, VkImage, VkImageLayout oldLayout, VkImageLayout newLayout);
};