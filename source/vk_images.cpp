#include "vk_images.h"
#include "vk_initializers.h"

void vkutil::transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkImageAspectFlags aspectMask = (newLayout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) 
                                    ? VK_IMAGE_ASPECT_DEPTH_BIT 
                                    : VK_IMAGE_ASPECT_COLOR_BIT;

	VkImageSubresourceRange imageSubresourceRange {
		.aspectMask = aspectMask,
		.baseMipLevel = 0,
		.levelCount = VK_REMAINING_MIP_LEVELS,
		.baseArrayLayer = 0,
		.layerCount = VK_REMAINING_ARRAY_LAYERS
	};

	VkImageMemoryBarrier2 imageBarrier {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
		.pNext = nullptr,
		.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, // FIXME (TF 25 FEB 2026): using this stage for transition is inefficient (stalls pipeline)
		.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
		.oldLayout = oldLayout,
		.newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, // FIXME (TF 25 FEB 2026): use 0?
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, // FIXME (TF 25 FEB 2026): use 0?
        .image = image,
        .subresourceRange = imageSubresourceRange
	};

	VkDependencyInfo depInfo {
		.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
		.pNext = nullptr,
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &imageBarrier
	};

	vkCmdPipelineBarrier2(cmd, &depInfo);
}

void vkutil::copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D srcSize, VkExtent2D dstSize)
{
	VkImageSubresourceLayers srcSubresource {
		.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		.mipLevel = 0,
		.baseArrayLayer = 0,
		.layerCount = 1
	};
	VkImageSubresourceLayers dstSubresource = srcSubresource;

	VkImageBlit2 blitRegion {
		.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
		.pNext = nullptr,
		.srcSubresource = srcSubresource,
		.srcOffsets = {{0,0,0}, {static_cast<int32_t>(srcSize.width), static_cast<int32_t>(srcSize.height), 1}},
		.dstSubresource = dstSubresource,
		.dstOffsets = {{0,0,0}, {static_cast<int32_t>(dstSize.width), static_cast<int32_t>(dstSize.height), 1}}
	};

	VkBlitImageInfo2 blitInfo{
		.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
		.pNext = nullptr,
		.srcImage = source,
		.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		.dstImage = destination,
		.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.regionCount = 1,
		.pRegions = &blitRegion,
		.filter = VK_FILTER_LINEAR
	};

	vkCmdBlitImage2(cmd, &blitInfo);
}

// TODO (TF 22 MAR 2026): KTX and DDS file formats are more performant for vulkan images, and
// support pre-generated mipmaps. Alternatively, generate mipmaps using a compute shader.
// DEBUG: assumes image is already in VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
void vkutil::generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D imageSize)
{
	uint32_t mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(imageSize.width, imageSize.height)))) + 1;
	for (uint32_t level = 0; level < mipLevels; ++level)
	{
		VkExtent2D halfSize{
			.width = static_cast<uint32_t>(imageSize.width * 0.5f),
			.height = static_cast<uint32_t>(imageSize.height * 0.5f)
		};

		VkImageSubresourceRange subresourceRange {
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = level,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = VK_REMAINING_ARRAY_LAYERS
		};

		VkImageMemoryBarrier2 imageBarrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
			.pNext = nullptr,
			.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
			.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
			.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, // FIXME (TF 22 MAR 2026): use 0?
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, // FIXME (TF 22 MAR 2026): use 0?
			.image = image,
			.subresourceRange = subresourceRange
		};

		VkDependencyInfo dependencyInfo{
			.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
			.pNext = nullptr,
			.dependencyFlags = 0,
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &imageBarrier
		};

		vkCmdPipelineBarrier2(cmd, &dependencyInfo);

		if (level < mipLevels - 1)
		{
			VkImageBlit2 blitRegion{
				.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
				.pNext = nullptr,
				.srcSubresource = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = level,
					.baseArrayLayer = 0,
					.layerCount = 1
				},
				.srcOffsets = {
					{ 0, 0, 0 },
					{ static_cast<int32_t>(imageSize.width), static_cast<int32_t>(imageSize.height), 1 }
				},
				.dstSubresource = {
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.mipLevel = level + 1,
					.baseArrayLayer = 0,
					.layerCount = 1
				},
				.dstOffsets = {
					{ 0, 0, 0 },
					{ static_cast<int32_t>(halfSize.width), static_cast<int32_t>(halfSize.height), 1 }
				}
			};

			VkBlitImageInfo2 blitInfo{
				.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
				.pNext = nullptr,
				.srcImage = image,
				.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				.dstImage = image,
				.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				.regionCount = 1,
				.pRegions = &blitRegion,
				.filter = VK_FILTER_LINEAR
			};

			vkCmdBlitImage2(cmd, &blitInfo);
			imageSize = halfSize;
		}
	}

	transition_image(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}
