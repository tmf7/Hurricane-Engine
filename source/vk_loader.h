#pragma once
#include "vk_types.h"
#include <unordered_map>
#include <filesystem>

#include "vk_descriptors.h"

struct GLTFMaterial {
	MaterialInstance data;
};

struct Bounds {
	glm::vec3 origin;
	float sphereRadius;
	glm::vec3 extents;
};

struct GeoSurface {
	uint32_t startIndex;
	uint32_t count;
	Bounds renderBounds;
	std::shared_ptr<GLTFMaterial> material;
};

struct MeshAsset {
	std::string name;
	std::vector<GeoSurface> surfaces;
	GPUMeshBuffers meshBuffers;
};

class VulkanEngine;

// ==================== BEGIN DEPRECATED ==============================
std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath);
// ==================== END DEPRECATED ==============================

struct LoadedGLTF : public IRenderable
{
public:
	VulkanEngine* engine;

	std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
	std::unordered_map<std::string, std::shared_ptr<Node>> nodes;
	std::unordered_map<std::string, AllocatedImage> images;
	std::unordered_map<std::string, std::shared_ptr<GLTFMaterial>> materials;

	std::vector<std::shared_ptr<Node>> rootNodes;
	std::vector<VkSampler> samplers;
	DescriptorAllocatorGrowable descriptorPool;
	AllocatedBuffer materialDataBuffer;

	// TODO (TF 20 MAR 2026): add runtim deletion either via VkQueueWait for resources to be available
	// or use the per-frame deletion queue lambda capture (similarly)
	~LoadedGLTF() { ClearAll(); }

	virtual void Draw(const glm::mat4& rootMatrix, DrawContext& ctx);

private:

	void ClearAll();
};

std::optional<std::shared_ptr<LoadedGLTF>> LoadGLTF(VulkanEngine* engine, std::filesystem::path filePath);
