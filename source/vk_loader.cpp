#include "vk_loader.h"
#include "stb_image.h"
#include <iostream>

#include "vk_engine.h"
#include "vk_initializers.h"
//#include "vk_types.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>

// ==================== BEGIN DEPRECATED ==============================
std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath)
{
	std::cout << "Loading GLTF: " << filePath << std::endl;

	fastgltf::Expected<fastgltf::GltfDataBuffer> gltfFile = fastgltf::GltfDataBuffer::FromPath(filePath);
	if (!gltfFile) {
		std::cout << "Failed to load glTF: " << fastgltf::to_underlying(gltfFile.error());
		return {};
	}

	constexpr fastgltf::Options gltfOptions = fastgltf::Options::LoadExternalBuffers;

	fastgltf::Asset gltf;
	fastgltf::Parser parser{};

	fastgltf::Expected<fastgltf::Asset> load = parser.loadGltfBinary(gltfFile.get(), filePath.parent_path(), gltfOptions);
	if (load) {
		gltf = std::move(load.get());
	} 
	else {
		std::cout << "Failed to load glTF: " << fastgltf::to_underlying(load.error());
		return {};
	}

	std::vector<std::shared_ptr<MeshAsset>> meshes;
	std::vector<uint32_t> indices;
	std::vector<Vertex> vertices;

	for (fastgltf::Mesh& mesh : gltf.meshes) {
		MeshAsset newMesh;

		newMesh.name = mesh.name;
		indices.clear();
		vertices.clear();

		for (auto&& primitive : mesh.primitives) {
			GeoSurface newSurface{
				.startIndex = static_cast<uint32_t>(indices.size()),
				.count = static_cast<uint32_t>(gltf.accessors[primitive.indicesAccessor.value()].count)
			};

			uint32_t initial_vtx = static_cast<uint32_t>(vertices.size());
			// load indexes
			{
				fastgltf::Accessor& indexAccessor = gltf.accessors[primitive.indicesAccessor.value()];
				indices.reserve(indices.size() + indexAccessor.count);

				fastgltf::iterateAccessor<uint32_t>(gltf, indexAccessor,
					[&](uint32_t idx) {
					indices.push_back(idx + initial_vtx);
				});
			}

			// load vertex positions
			{
				fastgltf::Attribute* positions = primitive.findAttribute("POSITION");
				fastgltf::Accessor& positionAccessor = gltf.accessors[positions->accessorIndex];
				vertices.resize(vertices.size() + positionAccessor.count);

				fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, positionAccessor,
					[&](glm::vec3 position, size_t index) {
					Vertex newVertex{
						.position = position,
						.uv_x = 0,
						.normal = {1.0f, 0.0f, 0.0f },
						.uv_y = 0,
						.color = glm::vec4{ 1.0f }
					};
					vertices[initial_vtx + index] = newVertex;
				});
			}

			// load vertex normals
			{
				fastgltf::Attribute* normals = primitive.findAttribute("NORMAL");
				if (normals != primitive.attributes.end()) {
					fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[normals->accessorIndex],
						[&](glm::vec3 normal, size_t index) {
						vertices[initial_vtx + index].normal = normal;
					});
				}
			}

			// load UVs
			{
				fastgltf::Attribute* uv = primitive.findAttribute("TEXCOORD_0");
				if (uv != primitive.attributes.end()) {
					fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[uv->accessorIndex],
						[&](glm::vec2 uv, size_t index) {
						vertices[initial_vtx + index].uv_x = uv.x;
						vertices[initial_vtx + index].uv_y = uv.y;
					});
				}
			}

			// load vertex colors
			{
				fastgltf::Attribute* colors = primitive.findAttribute("COLOR_0");
				if (colors != primitive.attributes.end()) {
					fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[colors->accessorIndex],
						[&](glm::vec4 color, size_t index) {
						vertices[initial_vtx + index].color = color;
					});
				}

			}
			newMesh.surfaces.push_back(newSurface);
		}
		// display vertex normals
		constexpr bool overrideColors = false;
		if (overrideColors) {
			for (Vertex& vertex : vertices) {
				vertex.color = glm::vec4(vertex.normal, 1.0f);
			}
		}
		newMesh.meshBuffers = engine->uploadMesh(indices, vertices);
		meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newMesh)));
	}

	return meshes;
}
// ==================== END DEPRECATED ==============================

void LoadedGLTF::Draw(const glm::mat4& rootMatrix, DrawContext& ctx)
{
	for (auto& node : rootNodes)
	{
		node->Draw(rootMatrix, ctx);
	}
}

void LoadedGLTF::ClearAll()
{
	// TODO (TF 19 FEB 2026): fill this function to destroy all allocations
}

VkFilter GetVulkanSamplerFilter(fastgltf::Filter filter)
{
	switch (filter)
	{
	case fastgltf::Filter::Nearest:
	case fastgltf::Filter::NearestMipMapNearest:
	case fastgltf::Filter::NearestMipMapLinear:
		return VK_FILTER_NEAREST;

	case fastgltf::Filter::Linear:
	case fastgltf::Filter::LinearMipMapNearest:
	case fastgltf::Filter::LinearMipMapLinear:
	default:
		return VK_FILTER_LINEAR;
	}
}

VkSamplerMipmapMode GetVulkanMipmapFilter(fastgltf::Filter filter)
{
	switch (filter)
	{
	case fastgltf::Filter::NearestMipMapNearest:
	case fastgltf::Filter::LinearMipMapNearest:
		return VK_SAMPLER_MIPMAP_MODE_NEAREST;

	case fastgltf::Filter::NearestMipMapLinear:
	case fastgltf::Filter::LinearMipMapLinear:
	default:
		return VK_SAMPLER_MIPMAP_MODE_LINEAR;
	}
}

std::optional<std::shared_ptr<LoadedGLTF>> LoadGLTF(VulkanEngine* engine, std::filesystem::path filePath)
{
	std::cout << "Loading GLTF: " << filePath << std::endl;

	std::shared_ptr<LoadedGLTF> scene = std::make_shared<LoadedGLTF>();
	scene->engine = engine;
	LoadedGLTF& loadedGLTF = *scene.get();

	fastgltf::Expected<fastgltf::GltfDataBuffer> gltfFile = fastgltf::GltfDataBuffer::FromPath(filePath);
	if (!gltfFile) {
		std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(gltfFile.error()) << std::endl;
		return {};
	}

	fastgltf::Asset gltfAsset;
	fastgltf::Parser parser{};

	constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember
								| fastgltf::Options::AllowDouble
								//| fastgltf::Options::LoadGLBBuffers // TODO (TF 19 MAR 2026): remove this, it's default behavior
								| fastgltf::Options::LoadExternalBuffers;

	fastgltf::GltfDataBuffer& data = gltfFile.get();
	auto fileType = fastgltf::determineGltfFileType(data);

	switch (fileType)
	{
		case fastgltf::GltfType::glTF:
		{
			fastgltf::Expected<fastgltf::Asset> parsedAsset = parser.loadGltf(data, filePath.parent_path(), gltfOptions);
			if (parsedAsset)
			{
				gltfAsset = std::move(parsedAsset.get());
			}
			else
			{
				std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(gltfFile.error()) << std::endl;
				return {};
			}
			break;
		}
		case fastgltf::GltfType::GLB:
		{
			fastgltf::Expected<fastgltf::Asset> parsedAsset = parser.loadGltfBinary(data, filePath.parent_path(), gltfOptions);
			if (parsedAsset)
			{
				gltfAsset = std::move(parsedAsset.get());
			}
			else
			{
				std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(gltfFile.error()) << std::endl;
				return {};
			}
			break;
		}
		default:
		{
			std::cerr << "Failed to determine glTF container type!" << std::endl;
			return {};
		}
	}

	// ======================== BEGIN LOADING MATERIALS =======================
	// DEBUG: rough estimate of necessary descriptors, can grow (but not change types)
	std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> poolSizes = {
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 }
	};
	loadedGLTF.descriptorPool.init(engine->_device, static_cast<uint32_t>(gltfAsset.materials.size()), poolSizes);
	
	for (fastgltf::Sampler& sampler : gltfAsset.samplers)
	{
		VkSamplerCreateInfo samplerInfo{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.pNext = nullptr,
			.flags = 0,
			.magFilter = GetVulkanSamplerFilter(sampler.magFilter.value_or(fastgltf::Filter::Nearest)),
			.minFilter = GetVulkanSamplerFilter(sampler.minFilter.value_or(fastgltf::Filter::Nearest)),
			.mipmapMode = GetVulkanMipmapFilter(sampler.minFilter.value_or(fastgltf::Filter::Nearest)),
			.minLod = 0,
			.maxLod = VK_LOD_CLAMP_NONE
		};

		VkSampler newSampler;
		vkCreateSampler(engine->_device, &samplerInfo, nullptr, &newSampler);
		loadedGLTF.samplers.push_back(newSampler);
	}

	// TODO (TF 19 MAR 2026): give these to a class so they can be cleared, instead of re-allocated on each load
	std::vector<AllocatedImage> images;
	std::vector<std::shared_ptr<GLTFMaterial>> materials;
	std::vector<std::shared_ptr<MeshAsset>> meshes;
	std::vector<std::shared_ptr<Node>> nodes;

	// DEBUG: everyting from the GLTF file must be loaded in order because all resources are indexed, not named
	for (fastgltf::Image& image : gltfAsset.images)
	{
		images.push_back(engine->_errorCheckerboardImage); // fallback
	}

	// TODO (TF 19 MAR 2026): make material extraction more broad/flexible, for now this can only load one type of material layout
	loadedGLTF.materialDataBuffer = engine->create_buffer(sizeof(GLTFMetallic_Roughness::MaterialConstants) * gltfAsset.materials.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	int dataIndex = 0;
	auto sceneMaterialConstants = static_cast<GLTFMetallic_Roughness::MaterialConstants*>(loadedGLTF.materialDataBuffer.info.pMappedData);

	for (fastgltf::Material& material : gltfAsset.materials)
	{
		std::shared_ptr<GLTFMaterial> newMaterial = std::make_shared<GLTFMaterial>();
		materials.push_back(newMaterial);
		loadedGLTF.materials[material.name.c_str()] = newMaterial;

		sceneMaterialConstants[dataIndex] = {
			.colorFactors = glm::vec4{ material.pbrData.baseColorFactor[0],
									   material.pbrData.baseColorFactor[1],
									   material.pbrData.baseColorFactor[2],
									   material.pbrData.baseColorFactor[3] },
			.metal_rough_factors = glm::vec4{ material.pbrData.metallicFactor,
											  material.pbrData.roughnessFactor,
											  0.0f,
											  0.0f }
			// padding to 256-byte alignment

		};;

		MaterialPass passType = MaterialPass::MainColor;
		if (material.alphaMode == fastgltf::AlphaMode::Blend)
		{
			passType = MaterialPass::Transparent;
		}

		GLTFMetallic_Roughness::MaterialResources materialResources{
			.colorImage = engine->_whiteImage,
			.colorSampler = engine->_defaultSamplerLinear,
			.metalRoughImage = engine->_whiteImage,
			.metalRoughSampler = engine->_defaultSamplerLinear,
			.dataBuffer = loadedGLTF.materialDataBuffer.buffer,
			.dataBufferOffset = dataIndex * sizeof(GLTFMetallic_Roughness::MaterialConstants),
		};

		if (material.pbrData.baseColorTexture.has_value())
		{
			size_t baseTextureIndex = material.pbrData.baseColorTexture.value().textureIndex;
			size_t imageIndex = gltfAsset.textures[baseTextureIndex].imageIndex.value();
			size_t samplerIndex = gltfAsset.textures[baseTextureIndex].samplerIndex.value();

			materialResources.colorImage = images[imageIndex];
			materialResources.colorSampler = loadedGLTF.samplers[samplerIndex];
		}

		newMaterial->data = engine->_metalRoughMaterial.write_material(engine->_device, passType, materialResources, loadedGLTF.descriptorPool);
		dataIndex++;
	}
	// ======================== END LOADING MATERIALS =======================
	
	// ======================== BEGIN LOADING MESHES =======================
	std::vector<uint32_t> indices;
	std::vector<Vertex> vertices;

	for (fastgltf::Mesh& mesh : gltfAsset.meshes)
	{
		std::shared_ptr<MeshAsset> newMesh = std::make_shared<MeshAsset>();
		meshes.push_back(newMesh);
		loadedGLTF.meshes[mesh.name.c_str()] = newMesh;
		newMesh->name = mesh.name;

		indices.clear();
		vertices.clear();

		for (auto&& primitive : mesh.primitives)
		{
			GeoSurface newSurface{
				.startIndex = static_cast<uint32_t>(indices.size()),
				.count = static_cast<uint32_t>(gltfAsset.accessors[primitive.indicesAccessor.value()].count)
			};

			size_t initialVertex = vertices.size();

			// load indexes
			{
				fastgltf::Accessor& indexAccessor = gltfAsset.accessors[primitive.indicesAccessor.value()];
				indices.reserve(indices.size() + indexAccessor.count);

				fastgltf::iterateAccessor<uint32_t>(gltfAsset, indexAccessor,
					[&](uint32_t index) {
					indices.push_back(index + initialVertex);
				});
			}

			// load vertex positions
			{
				fastgltf::Attribute* positions = primitive.findAttribute("POSITION");
				fastgltf::Accessor& positionAccessor = gltfAsset.accessors[positions->accessorIndex];
				vertices.resize(vertices.size() + positionAccessor.count);

				fastgltf::iterateAccessorWithIndex<glm::vec3>(gltfAsset, positionAccessor,
					[&](glm::vec3 position, uint32_t index) {
					Vertex newVertex{
						.position = position,
						.uv_x = 0.0f,
						.normal = {1.0f, 0.0f, 0.0f },
						.uv_y = 0.0f,
						.color = glm::vec4{ 1.0f }
					};
					vertices[index + initialVertex] = newVertex;
				});
			}

			// load vertex normals
			{
				fastgltf::Attribute* normals = primitive.findAttribute("NORMAL");
				if (normals != primitive.attributes.end()) // may not have normals
				{
					fastgltf::iterateAccessorWithIndex<glm::vec3>(gltfAsset, gltfAsset.accessors[normals->accessorIndex],
						[&](glm::vec3 normal, uint32_t index) {
						vertices[index + initialVertex].normal = normal;
					});
				}
			}

			// load UVs
			{
				fastgltf::Attribute* uvs = primitive.findAttribute("TEXCOORD_0");
				if (uvs != primitive.attributes.end()) // may not have UVs
				{
					fastgltf::iterateAccessorWithIndex<glm::vec2>(gltfAsset, gltfAsset.accessors[uvs->accessorIndex],
						[&](glm::vec2 uv, uint32_t index) {
						vertices[index + initialVertex].uv_x = uv.x;
						vertices[index + initialVertex].uv_y = uv.y;
					});
				}

			}

			// load vertex colors
			{
				fastgltf::Attribute* colors = primitive.findAttribute("COLOR_0");
				if (colors != primitive.attributes.end()) // may not have colors
				{
					fastgltf::iterateAccessorWithIndex<glm::vec4>(gltfAsset, gltfAsset.accessors[colors->accessorIndex],
						[&](glm::vec4 color, uint32_t index) {
						vertices[index + initialVertex].color = color;
					});
				}
			}

			// assign materials
			if (primitive.materialIndex.has_value())
			{
				newSurface.material = materials[primitive.materialIndex.value()];
			}
			else
			{
				newSurface.material = materials[0]; // DEBUG: fallback assumes at least one material in file
			}

			newMesh->surfaces.push_back(newSurface);
		}

		newMesh->meshBuffers = engine->uploadMesh(indices, vertices);
	}
	// ======================== END LOADING MESHES =======================

	// ======================== BEGIN LOADING NODES =======================
	for (fastgltf::Node& gltfNode : gltfAsset.nodes)
	{
		std::shared_ptr<Node> newNode;

		if (gltfNode.meshIndex.has_value())
		{
			newNode = std::make_shared<MeshNode>();
			static_cast<MeshNode*>(newNode.get())->mesh = meshes[*gltfNode.meshIndex];
		}
		else
		{
			newNode = std::make_shared<Node>();
		}

		nodes.push_back(newNode);
		loadedGLTF.nodes[gltfNode.name.c_str()] = newNode; // TODO(? TF 19 MAR 2026): leave unassigned?

		auto transformVisitor = fastgltf::visitor {
			[&](fastgltf::math::fmat4x4 matrix) {
				memcpy(&(newNode->localTransform), matrix.data(), sizeof(matrix));
			},
			[&](fastgltf::TRS transform) {
				glm::vec3 translation {
					transform.translation[0],
					transform.translation[1],
					transform.translation[2]
				};

				glm::quat rotation{
					transform.rotation[3],
					transform.rotation[0],
					transform.rotation[1],
					transform.rotation[2]
				};

				glm::vec3 scale{
					transform.scale[0],
					transform.scale[1],
					transform.scale[2]
				};

				glm::mat4 translationMatrix = glm::translate(glm::mat4{ 1.0f }, translation);
				glm::mat4 rotationMatrix = glm::toMat4(rotation);
				glm::mat4 scaleMatrix = glm::scale(glm::mat4{ 1.0f }, scale);
				newNode->localTransform = translationMatrix * rotationMatrix * scaleMatrix;
			} 
		};

		std::visit(transformVisitor, gltfNode.transform);
	}

	for (int i = 0; i < gltfAsset.nodes.size(); ++i)
	{
		fastgltf::Node& gltfNode = gltfAsset.nodes[i];
		std::shared_ptr<Node>& rootNode = nodes[i];

		for (auto& childIndex : gltfNode.children)
		{
			rootNode->children.push_back(nodes[childIndex]);
			nodes[childIndex]->parent = rootNode;
		}
	}

	for (auto& node : nodes)
	{
		if (node->parent.lock() == nullptr)
		{
			loadedGLTF.rootNodes.push_back(node);
			node->RefreshTransform(glm::mat4{ 1.0f });
		}
	}
	// ======================== END LOADING NODES =======================

	return scene; // loadedGLTF
}


