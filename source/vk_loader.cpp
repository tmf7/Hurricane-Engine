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
		constexpr bool overrideColors = true;
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
