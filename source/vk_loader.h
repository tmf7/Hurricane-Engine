#pragma once
#include "vk_types.h"
#include <unordered_map>
#include <filesystem>

#include "vk_descriptors.h"

struct GLTFMaterial {
	MaterialInstance data;
};

struct GeoSurface {
	uint32_t startIndex;
	uint32_t count;
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

	~LoadedGLTF() { ClearAll(); }

	virtual void Draw(const glm::mat4& rootMatrix, DrawContext& ctx);

private:

	void ClearAll();
};

std::optional<std::shared_ptr<LoadedGLTF>> LoadGLTF(VulkanEngine* engine, std::filesystem::path filePath);