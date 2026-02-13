// ===========  BEGIN TEMP CODE ===========  
//#define VK_USE_PLATFORM_WIN32_KHR
// ===========  END TEMP CODE ===========  
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

// ===========  BEGIN TEMP CODE ===========  
//#define GLFW_EXPOSE_NATIVE_WIN32
//#include <GLFW/glfw3native.h>
// ===========  END TEMP CODE ===========  

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <map>
#include <optional>
#include <set>

#include <cstdint> // for uint32_t
#include <limits> // for std::numeric_limits
#include <algorithm> // for std::clamp
#include <fstream>
#include <format>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <array>

#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <unordered_map>

#include <random>

struct Particle {
	glm::vec2 position;
	glm::vec2 velocity;
	glm::vec4 color;

	// TODO (TF 13 FEB 2026): pass in binding == 0 to indicate the singlar vertex buffer binding's index
	static VkVertexInputBindingDescription GetBindingDescription(uint32_t binding) {
		VkVertexInputBindingDescription bindingDescription{
			binding,					// binding
			sizeof(Particle),			// stride
			VK_VERTEX_INPUT_RATE_VERTEX // inputRate // TODO (TF 6 FEB 2026): experiment with instanced rendering
		};

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 2> GetAttributeDescriptions(uint32_t binding, uint32_t location) {
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

		attributeDescriptions[0].binding = binding;
		attributeDescriptions[0].location = location;
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Particle, position);

		attributeDescriptions[1].binding = binding;
		attributeDescriptions[1].location = location + 1; // DEBUG: prior attribute only takes up one 32-bit slot
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Particle, color);

		return attributeDescriptions;
	}
};

struct TransformUBO {
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
	//alignas(16) float time;
};

struct TimeUBO {
	alignas(4) float time;
};

struct Vertex {
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;

	bool operator==(const Vertex& other) const {
		return pos == other.pos
			&& color == other.color
			&& texCoord == other.texCoord;
	}

	static VkVertexInputBindingDescription GetBindingDescription(uint32_t binding) {
		VkVertexInputBindingDescription bindingDescription {
			binding,					// binding
			sizeof(Vertex),				// stride
			VK_VERTEX_INPUT_RATE_VERTEX // inputRate // TODO (TF 6 FEB 2026): experiment with instanced rendering
		};

		return bindingDescription;
	}

	// TODO (TF 6 FEB 2026): pass in binding == 0 to indicate the singlar vertex buffer binding's index
	// similarly set location == 0 so attributes are referenced correctly
	// ALTERNATIVELY: split this into distinct function calls like GetPositionAttributeDescription(binding), etc
	static std::array<VkVertexInputAttributeDescription, 3> GetAttributeDescriptions(uint32_t binding, uint32_t location) {
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

		attributeDescriptions[0].binding = binding;
		attributeDescriptions[0].location = location;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = binding;
		attributeDescriptions[1].location = location + 1; // DEBUG: prior attribute only takes up one 32-bit slot
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		attributeDescriptions[2].binding = binding;
		attributeDescriptions[2].location = location + 2; // DEBUG: prior attribute only takes up one 32-bit slot
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		return attributeDescriptions;
	}
};

// https://en.cppreference.com/w/cpp/utility/hash.html
namespace std {
	template<>
	struct hash<Vertex> {
		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^
				(hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
				(hash<glm::vec2>()(vertex.texCoord) << 1);
 		}
	};
}

const uint32_t WINDOW_WIDTH = 800;
const uint32_t WINDOW_HEIGHT = 600;
const int MAX_FRAMES_IN_FLIGHT = 2;

// DEBUG: ensure PARTICLE_COUNT equals the
// maxComputeWorkGroupInvocations, given [Wx, Wy, Wz] [Lx, Ly, Lz]
// so gl_GlobalInvocationID doe not exceed the SSBO array size
// eg: dispatch of [64, 1, 1] w/local of [32, 32, 1] = 64 x 32 x32 = 65,536 invocations
// so PARTICLE_COUNT of 4069 requires that many invocations, hence the current setup of
// dispatch [(4096 / 256), 1, 1] local [256, 1, 1] = 4096 total invocations
const int PARTICLE_COUNT = 4096; 
const int WORKGROUP_SIZE_X = PARTICLE_COUNT / 256;

const std::string TEST_TEXTURE_PATH = "Textures/texture.jpg";
const std::string MODEL_PATH = "Models/viking_room.obj";
const std::string MODEL_TEXTURE_PATH = "Textures/viking_room.png";

const std::vector<const char*> VALIDATION_LAYERS = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> DEVICE_EXTENSIONS = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

VkResult CreateDebugUtilsMessengerEXT(
	VkInstance instance, 
	const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, 
	const VkAllocationCallbacks* pAllocator, 
	VkDebugUtilsMessengerEXT* pDebugMessenger) {
	
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(
	VkInstance instance, 
	VkDebugUtilsMessengerEXT debugMessenger, 
	const VkAllocationCallbacks* pAllocator) {
	
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

class HelloTriangleApplication {
private:
	struct QueueFamilyIndices {
		// DEBUG: vk guarantees VK_QUEUE_GRAPHICS_BIT family supports VK_QUEUE_TRANSFER_BIT
		// DEBUG: vk gurantees at least ONE queue family supports both GRAPHICS and COMPUTE BITs
		std::optional<uint32_t> graphicsAndComputeFamily; 
		std::optional<uint32_t> presentFamily;
		std::optional<uint32_t> transferFamily;

		bool IsComplete() const {
			return graphicsAndComputeFamily.has_value()
				&& presentFamily.has_value() 
				&& transferFamily.has_value();
		}
	};

	struct SwapChainSupportDetails {
		VkSurfaceCapabilitiesKHR capabilities;
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

public:
	void Run() {
		InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT);
		InitVulkan();
		MainLoop();
		Cleanup();
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData) {
		
		std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;
	
		// VK_TRUE would indicate invalid call
		// should be aborted with VK_ERROR_VALIDATION_FAILED_EXT error.
		// However, that typically for testing layers themselves.
		return VK_FALSE; 
	}

	static std::vector<char> ReadFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error(std::format("failed to open file [{0}]!", filename));
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);
		file.seekg(0);
		file.read(buffer.data(), fileSize);
		file.close();

		return buffer;
	}

	static void FramebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

private:
	void InitWindow(uint32_t WindowWidth, uint32_t WindowHeight) {
		// TODO (TF 30 JAN 2026): experiment with creating a window using 
		// the the Windows window vulkan extension directly		
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		//glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WindowWidth, WindowHeight, "Vulkan Hurricane", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, FramebufferResizeCallback);
	}

	bool CheckValidationLayerSupport() {
		uint32_t layerCount;

		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		// confirm ALL required validation layers are available
		for (const char* layerName : VALIDATION_LAYERS) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (std::string(layerName) == layerProperties.layerName) {
					layerFound = true;
					std::cout << layerName << " FOUND." << std::endl;
					break;
				}
			}

			if (!layerFound) {
				std::cout << layerName << " NOT FOUND." << std::endl;
				return false;
			}
		}

		return true;
	}

	bool IsExtensionSupported(const char* requiredExtension, const std::vector<VkExtensionProperties>& supportedExtensions) {
		for (const auto& supportedExtension : supportedExtensions) {
			if (std::string(supportedExtension.extensionName) == requiredExtension) {
				std::cout << " [" << supportedExtension.extensionName << "] ";
				return true;
			}
		}

		return false;
	}

	std::vector<const char*> GetRequiredInstanceExtensions() {
		// get which windowing extensions are required
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensionNames;

		glfwExtensionNames = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		// cross-check which extensions this computer supports
		uint32_t supportedExtensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &supportedExtensionCount, nullptr);
		std::vector<VkExtensionProperties> supportedExtensions(supportedExtensionCount);

		vkEnumerateInstanceExtensionProperties(nullptr, &supportedExtensionCount, supportedExtensions.data());

		std::cout << "glfw required instance extensions (windows):\n";
		for (int i = 0; i < glfwExtensionCount; ++i) {
			std::cout << "\t" << glfwExtensionNames[i] << (IsExtensionSupported(glfwExtensionNames[i], supportedExtensions) ? " SUPPORTED" : " NOT SUPPORTED") << "\n";
		}

		std::cout << "available instance extensions:\n";
		for (const auto& supportedExtension : supportedExtensions) {
			std::cout << "\t" << supportedExtension.extensionName << "\n";
		}

		std::vector<const char*> requiredExtensions(glfwExtensionNames, glfwExtensionNames + glfwExtensionCount);

		// append validation debug logging extension
		if (enableValidationLayers) {
			requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return requiredExtensions;
	}

	void CreateSurface() {
		VkResult result = glfwCreateWindowSurface(instance, window, nullptr, &surface);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surace!");
		}

		// ===========  BEGIN TEMP CODE ===========  
		//VkWin32SurfaceCreateInfoKHR createInfo {
		//	VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,	// sType
		//	nullptr,											// pNext
		//	0,													// flags
		//	GetModuleHandle(nullptr),							// hinstance // DEBUG: throws if module freed before handle use
		//	glfwGetWin32Window(window)							// hwnd
		//};

		//// technically a WSI extension function, but so common the Vulkan loader includes it
		//// so vkGetInstanceProcAddr isn't needed
		//VkResult resultTEMP = vkCreateWin32SurfaceKHR(instance, &createInfo, nullptr, &surface);
		//if (resultTEMP != VK_SUCCESS) {
		//	throw std::runtime_error("failed to create window surface!");
		//}
		// ===========  END TEMP CODE ===========  
	}

	void CreateInstance() {
		if (enableValidationLayers && !CheckValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available!");
		}

		uint32_t enabledLayerCount = 0;
		const char* const* enabledLayerNames = nullptr;
		
		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		void* pInstanceNext = nullptr;

		if (enableValidationLayers) {
			enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
			enabledLayerNames = VALIDATION_LAYERS.data();

			// attach debug messenger to instance creation itself,
			// as well as instance actions (see SetupDebugMessenger)
			PopulateDebugMessengerCreateInfo(debugCreateInfo);
			pInstanceNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}

		VkApplicationInfo appInfo
		{
			VK_STRUCTURE_TYPE_APPLICATION_INFO, // sType
			nullptr,							// pNext
			"Hello Triangle",					// pApplicationName
			VK_MAKE_VERSION(1, 0, 0),			// applicationVersion
			"No Engine",						// pEngineName
			VK_MAKE_VERSION(1, 0, 0),			// engineVersion
			VK_API_VERSION_1_0					// apiVersion
		};

		auto requiredExtensions = GetRequiredInstanceExtensions();

		VkInstanceCreateInfo instanceCreateInfo
		{
			VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,			 // sType
			pInstanceNext,									 // pNext
			0,												 // flags
			&appInfo,										 // pApplicationInfo
			enabledLayerCount,								 // enabledLayerCount
			enabledLayerNames,								 // ppEnabledLayerNames
			static_cast<uint32_t>(requiredExtensions.size()),// enabledExtensionCount
			requiredExtensions.data()						 // ppEnabledExtensionNames
		};

		VkResult result = vkCreateInstance(&instanceCreateInfo, nullptr, &instance);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create vulkan instance!");
		}
	}

	// DEBUG: used with 2D test texture
	void PopulateTestVerticies() {
		vertices = {
			{{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0, 0.0f}, {0.0f, 0.0f}},
			{{0.5f, -0.5f, 0.0f}, {0.0f, 1.0, 0.0f}, {1.0f, 0.0f}},
			{{0.5f, 0.5f, 0.0f}, {0.0f, 0.0, 1.0f}, {1.0f, 1.0f}},
			{{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0, 1.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0, 0.0f}, {0.0f, 0.0f}},
			{{0.5f, -0.5f, -0.5f}, {0.0f, 1.0, 0.0f}, {1.0f, 0.0f}},
			{{0.5f, 0.5f, -0.5f}, {0.0f, 0.0, 1.0f}, {1.0f, 1.0f}},
			{{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0, 1.0f}, {0.0f, 1.0f}}
		};

		indices = {
			0, 1, 2, 2, 3, 0,
			4, 5, 6, 6, 7, 4
		};
	}

	void InitVulkan() {
		CreateInstance();
		SetupDebugMessenger();

		// DEBUG: create window surface after instance and prior to device
		// because it influences device selection (eg: multiple screens, etc)
		CreateSurface(); 
		PickPhysicalDevice();
		CreateLogicalDevice();

		// swapchain images are implicitly created when swapchain is created
		CreateSwapChain();
		CreateSwapChainImageViews();

		CreateRenderPass();
		CreateDescriptorSetLayout(0); // TODO (TF 8 FEB 2026): make descriptorSetLayout creation dynamic
		CreateComputeDescriptorSetLayout(0);
		CreateGraphicsPipeline();
		CreateComputePipeline();
		CreateCommandPools();

		CreateColorResources();
		CreateDepthResources();
		CreateFramebuffers(); // DEBUG: relies on depthImageView

		//CreateTextureImage(MODEL_TEXTURE_PATH.c_str());
		//CreateTextureImageView();
		//CreateTextureSampler();

		//LoadModel();
		//CreateVertexBuffer();
		//CreateIndexBuffer();

		InitializeParticlePositions();
		CreateShaderStorageBuffers();

		//CreateTransformUniformBuffers();
		CreateTimeUniformBuffers();
		//CreateModelShaderDescriptorPool();
		CreateComputeDescriptorPool();
		//CreateModelShaderDescriptorSets(0); // TODO (TF 8 FEB 2026): make descriptorSets creation dynamic
		CreateComputeDescriptorSets(0);
		CreateGraphicsAndComputeCommandBuffers();
		
		CreateSyncObjects();
	}

	void InitializeParticlePositions() {
		std::default_random_engine rndEngine((unsigned)time(nullptr));
		std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);
		particles.resize(PARTICLE_COUNT);

		for (auto& particle : particles) {
			float r = 0.25f * sqrt(rndDist(rndEngine));
			float theta = rndDist(rndEngine) * 2 * glm::pi<float>();
			float x = (r * cos(theta) * WINDOW_HEIGHT) / WINDOW_WIDTH;
			float y = r * sin(theta);
			particle.position = glm::vec2(x, y);
			particle.velocity = glm::normalize(glm::vec2(x, y)) * 0.000005f;
			particle.color = glm::vec4(rndDist(rndEngine),
									   rndDist(rndEngine),
									   rndDist(rndEngine),
									   1.0f);
		}
	}

	void CreateShaderStorageBuffers() {
		shaderStorageBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		shaderStorageBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);

		VkDeviceSize bufferSize = sizeof(Particle) * particles.size();
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		std::cout << "CREATING stagingBufferMemory (for shaderStorageBuffers): ";
		CreateBuffer(bufferSize, 
					 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
					 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
					 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					 stagingBuffer,
					 stagingBufferMemory);

		void* data;
		vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, particles.data(), (size_t)bufferSize);
		vkUnmapMemory(logicalDevice, stagingBufferMemory);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			std::cout << "CREATING shaderStorageBuffers[" << i << "]: ";
			CreateBuffer(bufferSize,
						 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
						 | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
						 | VK_BUFFER_USAGE_TRANSFER_DST_BIT, // vertex store for rendering, and host->GPU transfer dst buffer
						 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
						 shaderStorageBuffers[i],
						 shaderStorageBuffersMemory[i]);
			CopyBuffer(stagingBuffer, shaderStorageBuffers[i], bufferSize);
		}

		vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
		vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
	}

	void CreateColorResources() {
		VkFormat colorFormat = swapChainImageFormat;

		std::cout << "CREATING colorImageMemory: 0x";
		CreateImage(swapChainExtent.width, 
					swapChainExtent.height, 
					1, msaaSamples, 
					colorFormat,
					VK_IMAGE_TILING_OPTIMAL,
					VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT
					| VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					colorImage,
					colorImageMemory);
		colorImageView = CreateImageView(colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
	}

	VkSampleCountFlagBits GetMaxUsableSampleCount() {
		VkPhysicalDeviceProperties physicalDeviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

		VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts
									& physicalDeviceProperties.limits.framebufferDepthSampleCounts;
		if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
		if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
		if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
		if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
		if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
		if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

		return VK_SAMPLE_COUNT_1_BIT;
	}

	void LoadModel() {
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn;
		std::string err;

		// DEBUG: triangulation enabled by default
		bool success = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str());
		if (!success) {
			throw std::runtime_error(err);
		}

		std::unordered_map<Vertex, uint32_t> uniqueVertices{};
		for (const auto& shape : shapes) {
			for (const auto& index : shape.mesh.indices) {
				Vertex vertex{};

				// DEBUG: tinyobj verticies array is raw floats,
				// so multiply by 3 (2, etc) is necessary to get the correct vertex_index (etc) attribute
				vertex.pos = {
					attrib.vertices[3 * index.vertex_index + 0],
					attrib.vertices[3 * index.vertex_index + 1],
					attrib.vertices[3 * index.vertex_index + 2]
				};

				// flip y-axis to match vulkan's 0 == top of image system (obj setup for openGL 0 == bottom)
				vertex.texCoord = {
					attrib.texcoords[2 * index.texcoord_index + 0],
					1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
				};

				vertex.color = { 1.0f, 1.0f, 1.0f };

				if (uniqueVertices.count(vertex) == 0) {
					uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
					vertices.push_back(vertex);
				}

				// DEBUG: pick out the unique verticies and index triangles accordingly
				indices.push_back(uniqueVertices[vertex]);
				//vertices.push_back(vertex);
				//indices.push_back(indices.size());
			}
		}
	}

	VkFormat FindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR
				&& (props.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL
				&& (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}

		throw std::runtime_error("failed to find supported format!");
	}

	bool HasStencilComponent(VkFormat format) {
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT
			|| format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	VkFormat FindDepthFormat() {
		std::vector<VkFormat> formatCandidates = {
			VK_FORMAT_D32_SFLOAT,
			VK_FORMAT_D32_SFLOAT_S8_UINT,
			VK_FORMAT_D24_UNORM_S8_UINT
		};

		VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
		VkFormatFeatureFlags desiredFeatures = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;

		return FindSupportedFormat(formatCandidates, tiling, desiredFeatures);
	}

	void CreateDepthResources() {
		VkFormat depthFormat = FindDepthFormat();

		std::cout << "CREATING depthImageMemory: 0x";
		CreateImage(swapChainExtent.width, swapChainExtent.height, 
					1, msaaSamples,
					depthFormat,
					VK_IMAGE_TILING_OPTIMAL,
					VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					depthImage,
					depthImageMemory);

		depthImageView = CreateImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);

		// DEBUG: TransitionImageLayout() is not used here
		// to transition from UNDEFINED to DEPTH_STENCIL_ATTACHMENT_OPTIMAL
		// because the transition is implicitly handled in the renderpass setup
		//TransitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
	}

	void CreateTextureSampler() {
		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);
		float maxAnisotropy = properties.limits.maxSamplerAnisotropy; // guaranteed support of 1.0f

		VkSamplerCreateInfo samplerInfo {
			VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,		// sType
			nullptr,									// pNext
			0,											// flags
			VK_FILTER_LINEAR,							// magFilter
			VK_FILTER_LINEAR,							// minFilter
			VK_SAMPLER_MIPMAP_MODE_LINEAR,				// mipmapMode
			VK_SAMPLER_ADDRESS_MODE_REPEAT,				// addressModeU
			VK_SAMPLER_ADDRESS_MODE_REPEAT,				// addressModeV
			VK_SAMPLER_ADDRESS_MODE_REPEAT,				// addressModeW
			0.0f,										// mipLodBias
			VK_TRUE,									// anisotropyEnable
			maxAnisotropy,								// maxAnisotropy
			VK_FALSE,									// compareEnable
			VK_COMPARE_OP_ALWAYS,						// compareOp
			0.0f,										// minLod // TODO (TF 10 FEB 2026): test static_cast<float>(mipLevels / 2)
			VK_LOD_CLAMP_NONE,							// maxLod
			VK_BORDER_COLOR_INT_OPAQUE_BLACK,			// borderColor
			VK_FALSE									// unnormalizedCoordinates
		};

		VkResult result = vkCreateSampler(logicalDevice, &samplerInfo, nullptr, &textureSampler);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	VkImageView CreateImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, uint32_t mipLevels) {
		VkComponentMapping componentMapping {
			VK_COMPONENT_SWIZZLE_IDENTITY, // r
			VK_COMPONENT_SWIZZLE_IDENTITY, // g
			VK_COMPONENT_SWIZZLE_IDENTITY, // b
			VK_COMPONENT_SWIZZLE_IDENTITY  // a
		};

		VkImageSubresourceRange subresourceRange{
			aspectFlags,				// aspectMask
			0,							// baseMipLevel
			mipLevels,					// levelCount
			0,							// baseArraylayer
			1							// layerCount // TODO (TF 3 FEB 2026): experiment with multiple layers for stereoscopic 3D application
		};

		VkImageViewCreateInfo viewInfo {
			VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, // sType
			nullptr,								  // pNext
			0,										  // flags
			image,									  // image
			VK_IMAGE_VIEW_TYPE_2D,					  // viewType
			format,									  // format // TODO (TF 3 FEB 2026): experiment with different view formats
			componentMapping,						  // components // TODO (TF 3 FEB 2026): experiment with monochrome components, or const channels
			subresourceRange						  // subresourceRange
		};

		VkImageView imageView;
		VkResult result = vkCreateImageView(logicalDevice, &viewInfo, nullptr, &imageView);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("faild to create texture image view!");
		}
		
		return imageView;
	}

	void CreateTextureImageView() {
		textureImageView = CreateImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, mipLevels);
	}

	// creates a concurrent shared 2D image of depth == 1 extent
	void CreateImage(uint32_t width, uint32_t height, uint32_t mipLevels, 
					VkSampleCountFlagBits numSamples, VkFormat format, 
					VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags memoryPropertyFlags,
					VkImage& image, VkDeviceMemory& imageMemory) {
		//QueueFamilyIndices indices = FindQueueFamilies(physicalDevice);
		//std::vector<uint32_t> queueFamilyIndices = {
		//	indices.graphicsFamily.value(),
		//	indices.transferFamily.value()
		//};

		//if (indices.presentFamily != indices.graphicsFamily) {
		//	queueFamilyIndices.push_back(indices.presentFamily.value());
		//}

		VkExtent3D imageExtent{
			width,	// width
			height,	// height
			1		// depth
		};

		VkImageCreateInfo imageInfo {
			VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,				// sType
			nullptr,											// pNext
			0,													// flags // TODO (TF 9 FEB 2026): experiment with sparse image flags to avoid need for entire image to be memory-backed (eg: 3D voxel grid with air)
			VK_IMAGE_TYPE_2D,									// imageType
			format,												// format // TODO (TF 9 FEB 2026): account for device not supporting this/that format and react (don't crash)
			imageExtent,										// extent
			mipLevels,											// mipLevels
			1,													// arrayLayers
			numSamples,											// samples
			tiling,												// tiling
			usage,												// usage
			VK_SHARING_MODE_EXCLUSIVE,							// sharingMode
			0,													// queueFamilyIndexCount
			nullptr,											// pQueueFamilyIndices
			VK_IMAGE_LAYOUT_UNDEFINED							// initialLayout
		};

		VkResult createImageResult = vkCreateImage(logicalDevice, &imageInfo, nullptr, &image);
		if (createImageResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(logicalDevice, image, &memRequirements);

		uint32_t memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, memoryPropertyFlags);

		VkMemoryAllocateInfo allocInfo{
			VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,		// sType
			nullptr,									// pNext
			memRequirements.size,						// allocationSize
			memoryTypeIndex
		};

		VkResult memoryAllocResult = vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &imageMemory);
		if (memoryAllocResult != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(logicalDevice, image, imageMemory, 0);
		std::cout << std::hex << reinterpret_cast<uint64_t>(imageMemory) << std::dec << std::endl;
	}

	// IMPORTANT NOTES:
	// -> Mipmaps should be generated offline and stored in the texture tile alongside the base mip level, for best performance.
	// -> if runtime mip generation is necessary, but no format can be found which supports linear blitting,
	//		then generate mip levels in software (eg: stb_image_resize https://github.com/nothings/stb/tree/master)
	//		and load each mip level into the image the same way the base level texture is loaded.
	void GenerateMipMaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
		// check if the optimal tiling formatted image supports linear blitting
		VkFormatProperties formatProperties;
		vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

		if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
			throw std::runtime_error("texture image format does not support linear blitting!");
		}
		
		VkCommandBuffer oneTimeCommandBuffer = BeginSingleTimeCommands(graphicsAndComputeCommandPool);

		VkImageSubresourceRange subResourceRange {
			VK_IMAGE_ASPECT_COLOR_BIT,		// aspectMask 
			0,								// baseMipLevel
			1,								// levelCount
			0,								// baseArraylayer
			1								// layerCount
		};

		VkImageMemoryBarrier barrier {
			VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,		// sType
			nullptr,									// pNext
			0,											// srcAccessMask 
			0,											// dstAccessMask
			VK_IMAGE_LAYOUT_UNDEFINED,					// oldLayout
			VK_IMAGE_LAYOUT_UNDEFINED,					// newLayout
			VK_QUEUE_FAMILY_IGNORED,					// srcQueueFamilyIndex // FIXME: using transferQueue
			VK_QUEUE_FAMILY_IGNORED,					// dstQueueFamilyIndex // FIXME: using transferQueue
			image,										// image
			subResourceRange							// subresourceRange
		};

		int32_t mipWidth = texWidth;
		int32_t mipHeight = texHeight;

		for (uint32_t i = 1; i < mipLevels; ++i) {
			barrier.subresourceRange.baseMipLevel = i - 1;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			vkCmdPipelineBarrier(oneTimeCommandBuffer,		// commandBuffer
				VK_PIPELINE_STAGE_TRANSFER_BIT,				// srcStageMask
				VK_PIPELINE_STAGE_TRANSFER_BIT,				// dstStageMask
				0,											// dependencyFlags
				0,											// memoryBarrierCount
				nullptr,									// pMemoryBarriers
				0,											// bufferMemoryBarrierCount
				nullptr,									// pBufferMemoryBarriers
				1,											// imageMemoryBarrierCount
				&barrier									// pImageMemoryBarriers
			);

			VkImageSubresourceLayers srcSubresourceLayers {
				VK_IMAGE_ASPECT_COLOR_BIT,	// aspectMask
				i - 1,						// mipLevel
				0,							// baseArrayLayer
				1							// layerCount
			};

			VkImageSubresourceLayers dstSubresourceLayers {
				VK_IMAGE_ASPECT_COLOR_BIT,
				i,
				0,
				1
			};

			VkImageBlit blit {
				srcSubresourceLayers,								// srcSubresource
				{{0, 0, 0}, {mipWidth, mipHeight, 1}},				// srcOffsets[2]
				dstSubresourceLayers,								// dstSubresource
				{{ 0, 0, 0 }, {mipWidth > 1 ? mipWidth / 2 : 1,
							   mipHeight > 1 ? mipHeight / 2 : 1,
								1}}	// dstOffsets[2]
			};

			// DEBUG: vkCmdBlitImage is not supported on all platforms (image format must support linear filtering)
			vkCmdBlitImage(oneTimeCommandBuffer,
				image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				1, &blit,
				VK_FILTER_LINEAR);

			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			vkCmdPipelineBarrier(oneTimeCommandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				0,
				0, nullptr,
				0, nullptr,
				1, &barrier);

			if (mipWidth > 1) {
				mipWidth /= 2;
			}

			if (mipHeight > 1) {
				mipHeight /= 2;
			}
		}

		barrier.subresourceRange.baseMipLevel = mipLevels - 1;
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		vkCmdPipelineBarrier(oneTimeCommandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		EndSingleTimeCommands(graphicsAndComputeQueue, graphicsAndComputeCommandPool, oneTimeCommandBuffer);
	}

	void CreateTextureImage(const char* path) {
		int texWidth;
		int texHeight;
		int texChannels;

		stbi_uc* pixels = stbi_load(path, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		VkDeviceSize imageSize = texWidth * texHeight * 4;

		if (pixels == nullptr) {
			throw std::runtime_error("failed to load texture image!");
		}
		
		mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
		
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		std::cout << "CREATING stagingBufferMemory (for textureImage): ";
		CreateBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
								VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
								| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
								stagingBuffer,
								stagingBufferMemory);

		void* data;
		vkMapMemory(logicalDevice, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(logicalDevice, stagingBufferMemory);
		stbi_image_free(pixels);
		
		std::cout << "CREATING textureImageMemory: 0x";
		CreateImage(texWidth, texHeight, 
					mipLevels, VK_SAMPLE_COUNT_1_BIT,
					VK_FORMAT_R8G8B8A8_SRGB, 
					VK_IMAGE_TILING_OPTIMAL, 
					VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
					textureImage,
					textureImageMemory);

		TransitionImageLayout(textureImage, 
			VK_FORMAT_R8G8B8A8_SRGB, 
			VK_IMAGE_LAYOUT_UNDEFINED, 
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			mipLevels
		);

		CopyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
	
		//TransitionImageLayout(textureImage,
		//	VK_FORMAT_R8G8B8A8_SRGB,
		//	VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		//	VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		//	mipLevels
		//);
		GenerateMipMaps(textureImage, VK_FORMAT_R8G8B8A8_SRGB, texWidth, texHeight, mipLevels);

		vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
		vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
	}

	void CreateComputeDescriptorSets(uint32_t binding) {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, computeDescriptorSetLayout);

		VkDescriptorSetAllocateInfo allocInfo{
			VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,		// sType
			nullptr,											// pNext
			computeDescriptorPool,								// descriptorPool
			static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),		// descriptorSetCount
			layouts.data()										// pSetLayouts
		};

		computeDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		VkResult result = vkAllocateDescriptorSets(logicalDevice, &allocInfo, computeDescriptorSets.data());
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate compute descriptor sets!");
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			VkDescriptorBufferInfo uniformBufferInfo{
				timeUniformBuffers[i],			// buffer
				0,								// offset
				sizeof(TimeUBO)					// range
			};

			VkDescriptorBufferInfo storageBufferInfoLastFrameInfo{
				shaderStorageBuffers[(i - 1) % MAX_FRAMES_IN_FLIGHT],
				0,
				sizeof(Particle) * PARTICLE_COUNT
			};

			VkDescriptorBufferInfo storageBufferInfoCurrentFrameInfo{
				shaderStorageBuffers[i],
				0,
				sizeof(Particle) * PARTICLE_COUNT
			};

			VkWriteDescriptorSet uniformBufferDescriptorWrite{
				VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,		// sType
				nullptr,									// pNext
				computeDescriptorSets[i],					// dstSet
				binding,									// dstBinding
				0,											// dstArrayElement
				1,											// descriptorCount
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,			// descriptorType
				nullptr,									// pImageInfo
				&uniformBufferInfo,							// pBufferInfo
				nullptr										// pTexelBufferView
			};

			VkWriteDescriptorSet storageBufferInfoLastFrameWrite{
				VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,		// sType
				nullptr,									// pNext
				computeDescriptorSets[i],					// dstSet
				binding + 1,								// dstBinding
				0,											// dstArrayElement
				1,											// descriptorCount
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,			// descriptorType
				nullptr,									// pImageInfo
				&storageBufferInfoLastFrameInfo,			// pBufferInfo
				nullptr										// pTexelBufferView
			};

			VkWriteDescriptorSet storageBufferInfoCurrentFrameWrite{
				VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,		// sType
				nullptr,									// pNext
				computeDescriptorSets[i],					// dstSet
				binding + 2,								// dstBinding
				0,											// dstArrayElement
				1,											// descriptorCount
				VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,			// descriptorType
				nullptr,									// pImageInfo
				&storageBufferInfoCurrentFrameInfo,			// pBufferInfo
				nullptr										// pTexelBufferView
			};

			std::array<VkWriteDescriptorSet, 3> descriptorWrites{
				uniformBufferDescriptorWrite,
				storageBufferInfoLastFrameWrite,
				storageBufferInfoCurrentFrameWrite
			};

			vkUpdateDescriptorSets(logicalDevice, 
				static_cast<uint32_t>(descriptorWrites.size()), 
				descriptorWrites.data(), 
				0, 
				nullptr);
		}
	}

	void CreateModelShaderDescriptorSets(uint32_t binding) {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);

		VkDescriptorSetAllocateInfo allocInfo {
			VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,		// sType
			nullptr,											// pNext
			descriptorPool,										// descriptorPool
			static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),		// descriptorSetCount
			layouts.data()										// pSetLayouts
		};

		descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		VkResult result = vkAllocateDescriptorSets(logicalDevice, &allocInfo, descriptorSets.data());
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		// TODO (TF 9 FEB 2026): make the number and type of bindings dynamic
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			VkDescriptorBufferInfo bufferInfo {
				transformUniformBuffers[i],		// buffer
				0,								// offset
				sizeof(TransformUBO),			// range // DEBUG: VK_WHOLE_SIZE also works here
			};

			VkDescriptorImageInfo imageInfo {
				textureSampler,								// sampler
				textureImageView,							// imageView
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL	// imageLayout
			};

			VkWriteDescriptorSet bufferDescriptorWrite {
				VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,		// sType
				nullptr,									// pNext
				descriptorSets[i],							// dstSet
				binding,									// dstBinding		// TODO (TF 8 FEB 2026): make dstBinding dynamic
				0,											// dstArrayElement	// TODO (TF 8 FEB 2026): make dstArrayElement dynamic
				1,											// descriptorCount
				VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,			// descriptorType
				nullptr,									// pImageInfo
				&bufferInfo,								// pBufferInfo
				nullptr										// pTexelBufferView
			};

			VkWriteDescriptorSet imageDescriptorWrite {
				VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,		// sType
				nullptr,									// pNext
				descriptorSets[i],							// dstSet
				binding + 1,								// dstBinding
				0,											// dstArrayElement
				1,											// descriptorCount
				VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,	// descriptorType
				&imageInfo,									// pImageInfo
				nullptr,									// pBufferInfo
				nullptr										// pTexelBufferView
			};

			std::array<VkWriteDescriptorSet, 2> descriptorWrites {
				bufferDescriptorWrite,
				imageDescriptorWrite
			};

			vkUpdateDescriptorSets(logicalDevice, 
				static_cast<uint32_t>(descriptorWrites.size()), 
				descriptorWrites.data(),
				0, 
				nullptr);
		}
	}

	void CreateComputeDescriptorPool() {
		VkDescriptorPoolSize uniformBufferPoolSize {
			VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,			// type
			static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)	// descriptorCount
		};
		VkDescriptorPoolSize ssboPoolSize {
			VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,				// type
			static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 2	// descriptorCount // DEBUG: account for the 2x SSBOs for hendling prior/current frame data
		};

		std::array<VkDescriptorPoolSize, 2> poolSizes {
			uniformBufferPoolSize,
			ssboPoolSize
		};

		VkDescriptorPoolCreateInfo poolInfo {
			VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,	// sType
			nullptr,										// pNext
			0,												// flags // TODO (TF 8 FEB 2026): experiment with VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
			static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT) * 2,// maxSets	
			static_cast<uint32_t>(poolSizes.size()),		// poolSizeCount
			poolSizes.data()								// pPoolsizes
		};

		VkResult result = vkCreateDescriptorPool(logicalDevice, &poolInfo, nullptr, &computeDescriptorPool);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create compute descriptor pool!");
		}
	}

	void CreateModelShaderDescriptorPool() {
		VkDescriptorPoolSize uniformBufferPoolSize {
			VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,			// type
			static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)	// descriptorCount
		};
		VkDescriptorPoolSize textureSamplerPoolSize {
			VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,	// type
			static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT)	// descriptorCount
		};

		std::array<VkDescriptorPoolSize, 2> poolSizes {
			uniformBufferPoolSize,
			textureSamplerPoolSize
		};

		VkDescriptorPoolCreateInfo poolInfo {
			VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,	// sType
			nullptr,										// pNext
			0,												// flags // TODO (TF 8 FEB 2026): experiment with VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT
			static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT),	// maxSets
			static_cast<uint32_t>(poolSizes.size()),		// poolSizeCount
			poolSizes.data()								// pPoolsizes
		};

		VkResult result = vkCreateDescriptorPool(logicalDevice, &poolInfo, nullptr, &descriptorPool);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	void CreateTransformUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(TransformUBO);

		transformUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		transformUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		transformUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			std::cout << "CREATING mapped transformUniformBuffersMemory[" << i << "]: ";
			CreateBuffer(bufferSize,
						 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
						 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
						 | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
						 transformUniformBuffers[i],
						 transformUniformBuffersMemory[i]);

			vkMapMemory(logicalDevice, transformUniformBuffersMemory[i], 0, bufferSize, 0, &transformUniformBuffersMapped[i]);
		}
	}

	void CreateTimeUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(TimeUBO);

		timeUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		timeUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
		timeUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			std::cout << "CREATING mapped timeUniformBuffersMemory[" << i << "]: ";
			CreateBuffer(bufferSize,
				VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
				| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				timeUniformBuffers[i],
				timeUniformBuffersMemory[i]);

			vkMapMemory(logicalDevice, timeUniformBuffersMemory[i], 0, bufferSize, 0, &timeUniformBuffersMapped[i]);
		}
	}

	// DEBUG: two storage buffers are used to allow particle positions to be updated
	// on a single frame by reading from one and writing to the other without a write-after-read hazard
	void CreateComputeDescriptorSetLayout(uint32_t binding) {
		VkDescriptorSetLayoutBinding uboLayoutBinding {
			binding,							// binding
			VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,	// descriptorType
			1,									// descriptorCount l transforms)
			VK_SHADER_STAGE_COMPUTE_BIT,		// stageFlags
			nullptr								// pImmutableSamplers
		};

		VkDescriptorSetLayoutBinding ssboLayoutBinding_A {
			binding + 1,						// binding
			VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,	// descriptorType
			1,									// descriptorCount
			VK_SHADER_STAGE_COMPUTE_BIT,		// stageFlags
			nullptr								// pImmutableSamplers
		};

		VkDescriptorSetLayoutBinding ssboLayoutBinding_B {
			binding + 2,						// binding
			VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,	// descriptorType
			1,									// descriptorCount
			VK_SHADER_STAGE_COMPUTE_BIT,		// stageFlags
			nullptr								// pImmutableSamplers
		};

		std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
			uboLayoutBinding,
			ssboLayoutBinding_A,
			ssboLayoutBinding_B
		};

		VkDescriptorSetLayoutCreateInfo layoutInfo {
			VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,	// sType
			nullptr,												// pNext
			0,														// flags
			static_cast<uint32_t>(bindings.size()),					// bindingCount
			bindings.data()											// pBindings
		};

		VkResult result = vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &computeDescriptorSetLayout);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create compute descriptor set layout!");
		}
	}

	void CreateDescriptorSetLayout(uint32_t binding) {
		VkDescriptorSetLayoutBinding uboLayoutBinding {
			binding,							// binding
			VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,	// descriptorType
			1,									// descriptorCount // TODO (TF 8 FEB 2026): experiment with multiple descriptors (eg: for skeletal transforms)
			VK_SHADER_STAGE_VERTEX_BIT,			// stageFlags
			nullptr								// pImmutableSamplers
		};

		VkDescriptorSetLayoutBinding samplerLayoutBinding {
			binding + 1,								// binding
			VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,	// descriptorType
			1,											// descriptorCount
			VK_SHADER_STAGE_FRAGMENT_BIT,				// stageFlags
			nullptr										// pImmutableSamplers
		};

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
			uboLayoutBinding,
			samplerLayoutBinding
		};

		VkDescriptorSetLayoutCreateInfo layoutInfo {
			VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,	// sType
			nullptr,												// pNext
			0,														// flags
			static_cast<uint32_t>(bindings.size()),					// bindingCount
			bindings.data()											// pBindings
		};

		VkResult result = vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr, &descriptorSetLayout);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
			if (typeFilter & (1 << i)
				&& (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
		VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {

		QueueFamilyIndices indices = FindQueueFamilies(physicalDevice);
		std::vector<uint32_t> queueFamilyIndices = {
			indices.graphicsAndComputeFamily.value(),
			indices.transferFamily.value()
		};

		if (indices.presentFamily != indices.graphicsAndComputeFamily) {
			queueFamilyIndices.push_back(indices.presentFamily.value());
		}

		// DEBUG: exclusive is more performant but requires explicit
		// transfer of resource ownership
		VkBufferCreateInfo bufferInfo{
			VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,				// sType
			nullptr,											// pnext
			0,													// flags  // DEBUG: not a sparce buffer for now
			size,												// size
			usage,												// usage
			VK_SHARING_MODE_CONCURRENT,							// sharingMode // DEBUG: used by graphicsQueue and transferQueue
			static_cast<uint32_t>(queueFamilyIndices.size()),	// queueFamilyIndexCount
			queueFamilyIndices.data()							// pQueueFamilyIndices
		};

		VkResult bufferResult = vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &buffer);
		if (bufferResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create vertex buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(logicalDevice, buffer, &memRequirements);

		uint32_t memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, properties);

		VkMemoryAllocateInfo allocInfo{
			VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,		// sType
			nullptr,									// pNext
			memRequirements.size,						// allocationSize
			memoryTypeIndex								// memoryTypeIndex
		};

		// TODO (TF 6 FEB 2026): use an external memory allocator/manager to limit number of runtime allocations
		// so it doesn't  hit the maxMemoryAllocationCount (ie: don't call vkAllocateMemory for every new buffer)
		// ...this is only okay for now because its only a few allocations at startup
		VkResult memoryResult = vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &bufferMemory);
		if (memoryResult != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate vertex buffer memory!");
		}

		// DEBUG: singlar vertex buffer and memory binding so offset is 0
		vkBindBufferMemory(logicalDevice, buffer, bufferMemory, 0);
		std::cout << "0x" << std::hex << reinterpret_cast<uint64_t>(bufferMemory) << std::dec << std::endl;
	}

	void CreateVertexBuffer() {
		VkDeviceSize bufferSize = sizeof(vertices[0])* vertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		std::cout << "CREATING stagingBufferMemory (for vertexBuffer): ";
		CreateBuffer(bufferSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer,
			stagingBufferMemory);

		// copy vertex data from the host to space accessible to gpu as of the next vkQueueSubmit call
		void* data;
		vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(logicalDevice, stagingBufferMemory);

		std::cout << "CREATING vertexBufferMemory: ";
		CreateBuffer(bufferSize,
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			vertexBuffer,
			vertexBufferMemory);

		CopyBuffer(stagingBuffer, vertexBuffer, bufferSize);

		vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
		vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
	}

	void CreateIndexBuffer() {
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		std::cout << "CREATING stagingBufferMemory (for indexBuffer): ";
		CreateBuffer(bufferSize,
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			stagingBuffer,
			stagingBufferMemory);

		// copy index data from the host to space accessible to gpu as of the next vkQueueSubmit call
		void* data;
		vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(logicalDevice, stagingBufferMemory);

		std::cout << "CREATING indexBufferMemory: ";
		CreateBuffer(bufferSize,
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			indexBuffer,
			indexBufferMemory);

		CopyBuffer(stagingBuffer, indexBuffer, bufferSize);

		vkDestroyBuffer(logicalDevice, stagingBuffer, nullptr);
		vkFreeMemory(logicalDevice, stagingBufferMemory, nullptr);
	}
	
	// TODO (TF 9 FEB 2026): create SetupCommandBuffer and FlushCommandBuffer helper functions
	// to allow execution of multiple commands asynchronously in a single commandBuffer (CopyBuffer, CopyBufferToImage, TransitionImage)
	VkCommandBuffer BeginSingleTimeCommands(VkCommandPool commandPool) {
		VkCommandBufferAllocateInfo oneTimeAllocInfo {
			VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,		// sType
			nullptr,											// pNext
			commandPool,										// commandPool
			VK_COMMAND_BUFFER_LEVEL_PRIMARY,					// level
			1													// commandBufferCount
		};

		VkCommandBuffer oneTimeCommandBuffer;
		VkResult transferAllocateResult = vkAllocateCommandBuffers(logicalDevice, &oneTimeAllocInfo, &oneTimeCommandBuffer);
		if (transferAllocateResult != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate transfer command buffer!");
		}

		VkCommandBufferBeginInfo beginInfo{
			VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,	// sType
			nullptr,										// pNext
			VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,	// flags // DEBUG: tell driver to wait until buffer copy is done before returning
			nullptr											// pInheritanceInfo
		};

		vkBeginCommandBuffer(oneTimeCommandBuffer, &beginInfo);

		return oneTimeCommandBuffer;
	}

	void EndSingleTimeCommands(VkQueue queue, VkCommandPool commandPool, VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{
			VK_STRUCTURE_TYPE_SUBMIT_INFO,		// sType
			nullptr,							// pNext
			0,									// waitSemaphoreCount
			nullptr,							// pWaitSemaphores
			nullptr,							// pWaitDstStageMask
			1,									// commandBufferCount
			&commandBuffer,						// pCommandBuffers
			0,									// signalSemaphoreCount
			nullptr								// pSignalSemaphores
		};

		vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(queue); // TODO (TF 6 FEB 2026): use a fence or semaphore to sync more precisely (and allow parallel transfers)
		vkFreeCommandBuffers(logicalDevice, commandPool, 1, &commandBuffer);
	}

	void TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) {
		VkCommandBuffer oneTimeCommandBuffer = BeginSingleTimeCommands(graphicsAndComputeCommandPool); // DEBUG: must use graphicsCommandPool because it can support TRANSFER_BIT and FRAGMENT_BIT image transition commands

		VkAccessFlags sourceAccessMask;
		VkAccessFlags destinationAccessMask;
		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;
		VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
		
		if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
			aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT;

			if (HasStencilComponent(format)) {
				aspectFlags |= VK_IMAGE_ASPECT_STENCIL_BIT;
			}
		}

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			sourceAccessMask = 0;
			destinationAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			sourceAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			destinationAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
			sourceAccessMask = 0;
			destinationAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
									| VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		}
		else {
			throw std::runtime_error("unsupported layout transition!");
		}

		VkImageSubresourceRange subResourceRange {
			aspectFlags,					// aspectMask // TODO (TF 9 FEB 2026): experiment with other aspectMask values (stencil, depth, etc)
			0,								// baseMipLevel
			mipLevels,						// levelCount
			0,								// baseArraylayer
			1								// layerCount
		};

		// FIXME (? TF 10 FEB 2026): use src/dst QueueFamilyIndex to transfer ownership as needed (using 2+ queues)
		//QueueFamilyIndices indices = FindQueueFamilies(physicalDevice);

		VkImageMemoryBarrier barrier {
			VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,		// sType
			nullptr,									// pNext
			sourceAccessMask,							// srcAccessMask 
			destinationAccessMask,						// dstAccessMask
			oldLayout,									// oldLayout
			newLayout,									// newLayout
			VK_QUEUE_FAMILY_IGNORED,					// srcQueueFamilyIndex // FIXME: using transferQueue
			VK_QUEUE_FAMILY_IGNORED,					// dstQueueFamilyIndex // FIXME: using transferQueue
			image,										// image
			subResourceRange							// subresourceRange
		};

		vkCmdPipelineBarrier(oneTimeCommandBuffer,		// commandBuffer
			sourceStage,				// srcStageMask
			destinationStage,			// dstStageMask
			0,							// dependencyFlags // TODO (TF 9 FEB 2026): experiment with dependencyFlags (eg: to allow read of already written regions)
			0,							// memoryBarrierCount
			nullptr,					// pMemoryBarriers
			0,							// bufferMemoryBarrierCount
			nullptr,					// pBufferMemoryBarriers
			1,							// imageMemoryBarrierCount
			&barrier					// pImageMemoryBarriers
		);

		EndSingleTimeCommands(graphicsAndComputeQueue, graphicsAndComputeCommandPool, oneTimeCommandBuffer);
	}

	void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
		// DEBUG: must use graphicsAndComputeCommandPool 
		// because it can support TRANSFER_BIT and FRAGMENT_BIT image transition commands
		VkCommandBuffer oneTimeCommandBuffer = BeginSingleTimeCommands(graphicsAndComputeCommandPool); 

		VkImageSubresourceLayers subresourceLayers {
			VK_IMAGE_ASPECT_COLOR_BIT,		// aspectMask
			0,								// mipLevel
			0,								// baseArraylayer
			1								// layerCount
		};

		VkOffset3D offset = { 0, 0, 0 }; // xyz
		VkExtent3D extent {
			width,		// width
			height,		// height
			1			// depth
		};

		VkBufferImageCopy region {
			0,						// bufferOffset
			0,						// rowLength		 // DEBUG: 0 indicates bits are tightly packed
			0,						// bufferImageHeight // DEBUG: 0 indicates bits are tightly packed
			subresourceLayers,		// imageSubresource
			offset,					// imageOffset
			extent					// imageExtent
		};

		vkCmdCopyBufferToImage(oneTimeCommandBuffer, 
			buffer, 
			image, 
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&region
		);

		EndSingleTimeCommands(graphicsAndComputeQueue, graphicsAndComputeCommandPool, oneTimeCommandBuffer);
	}

	void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		VkCommandBuffer oneTimeCommandBuffer = BeginSingleTimeCommands(transferCommandPool);

		VkBufferCopy copyRegion {
			0,			// srcOffset
			0,			// dstOffset
			size		// size
		};

		vkCmdCopyBuffer(oneTimeCommandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
		EndSingleTimeCommands(transferQueue, transferCommandPool, oneTimeCommandBuffer);
	}

	void CreateSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
		computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT); // FIXME (? TF 13 FEB 2026): index to swapChainImages.size as needed (simlar to renderFinishedSemaphores per frame

		// DEBUG: ensure the rendering semaphores are available to be signaled by indexing to swapChain images directly
		// ie: vkQueuePresentKHR doesn't have pSignalSemaphores like vkQueueSubmit does, so this DrawFrame function
		// relies on a Fence to assume the gpu semaphores are ready, which is not guaranted when only indexing by frames-in-flight
		renderFinishedSemaphores.resize(swapChainImages.size());

		VkSemaphoreCreateInfo semaphoreInfo {
			VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,	// sType
			nullptr,									// pNext
			0											// flags
		};

		// DEBUG: start fence in signaled state so it doesn't block on first frame
		VkFenceCreateInfo fenceInfo {
			VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,	// sType
			nullptr,								// pNext
			VK_FENCE_CREATE_SIGNALED_BIT			// flags
		};

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			VkResult imageSempaphoreResult = vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]);
			VkResult frameFenceResult = vkCreateFence(logicalDevice, &fenceInfo, nullptr, &inFlightFences[i]);
			VkResult computeFrameFenceResult = vkCreateFence(logicalDevice, &fenceInfo, nullptr, &computeInFlightFences[i]);

			if (imageSempaphoreResult != VK_SUCCESS
				|| frameFenceResult != VK_SUCCESS
				|| computeFrameFenceResult != VK_SUCCESS) {
				throw std::runtime_error("failed to create render|compute synchronization objects for a frame!");
			}

			VkResult computeFinishedSemaphoreResult = vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &computeFinishedSemaphores[i]);
			if (computeFinishedSemaphoreResult != VK_SUCCESS) {
				throw std::runtime_error("failed to create compute synchronization objects for a frame!");
			}
		}

		for (size_t i = 0; i < swapChainImages.size(); ++i) {
			VkResult finishedSemaphoreResult = vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]);
			if (finishedSemaphoreResult != VK_SUCCESS) {
				throw std::runtime_error("failed to create render synchronization objects for a frame!");
			}

		}
	}

	void CreateGraphicsAndComputeCommandBuffers() {
		graphicsAndComputeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		// TODO (TF 5 FEB 2026): expierment with seconday command buffers
		VkCommandBufferAllocateInfo graphicsAllocInfo {
			VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,					// sType
			nullptr,														// pNext
			graphicsAndComputeCommandPool,									// commandPool
			VK_COMMAND_BUFFER_LEVEL_PRIMARY,								// level
			static_cast<uint32_t>(graphicsAndComputeCommandBuffers.size())	// commandBufferCount
		};

		VkResult graphicsAllocateResult = vkAllocateCommandBuffers(logicalDevice, &graphicsAllocInfo, graphicsAndComputeCommandBuffers.data());
		if (graphicsAllocateResult != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate graphics command buffers!");
		}
	}

	void CreateCommandPools() {
		QueueFamilyIndices queueFamilyIndices = FindQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo graphicsAndComputePoolInfo {
			VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,			// sType
			nullptr,											// pNext
			VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,	// flags
			queueFamilyIndices.graphicsAndComputeFamily.value()	// queueFamilyIndex
		};

		VkResult graphicsPoolResult = vkCreateCommandPool(logicalDevice, &graphicsAndComputePoolInfo, nullptr, &graphicsAndComputeCommandPool);
		if (graphicsPoolResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics command pool!");
		}

		VkCommandPoolCreateInfo transferPoolInfo {
			VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,			// sType
			nullptr,											// pNext
			VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
			| VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,				// flags // DEBUG: used for quick memory transfers from staging buffers
			queueFamilyIndices.transferFamily.value()			// queueFamilyIndex
		};

		VkResult transferPoolResult = vkCreateCommandPool(logicalDevice, &transferPoolInfo, nullptr, &transferCommandPool);
		if (transferPoolResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create transfer command pool!");
		}
	}

	void CreateFramebuffers() {
		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (size_t i = 0; i < swapChainImageViews.size(); ++i) {
			std::array<VkImageView, 3> attachments = {
				colorImageView, // DEBUG; only one subpass is running at a time due to semaphore usage, so colorImageView, and depthImageView can be reused for each swapChainImageView
				depthImageView,
				swapChainImageViews[i]
			};

			VkFramebufferCreateInfo framebufferInfo {
				VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,	// sType
				nullptr,									// pNext
				0,											// flags
				renderPass,									// renderPass
				static_cast<uint32_t>(attachments.size()),	// attachmentCount
				attachments.data(),							// pAttachments
				swapChainExtent.width,						// width
				swapChainExtent.height,						// height
				1											// layers
			};

			VkResult result = vkCreateFramebuffer(logicalDevice, &framebufferInfo, nullptr, &swapChainFramebuffers[i]);
			if (result != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	void CreateRenderPass() {
		VkAttachmentDescription colorAttachment {
			0,												// flags
			swapChainImageFormat,							// format
			msaaSamples,									// samples
			VK_ATTACHMENT_LOAD_OP_CLEAR,					// loadOp // TODO (TF 4 FEB 2026): experiment with not clearning the image before rendering to it
			VK_ATTACHMENT_STORE_OP_STORE,					// storeOp
			VK_ATTACHMENT_LOAD_OP_DONT_CARE,				// stencilLoadOp
			VK_ATTACHMENT_STORE_OP_DONT_CARE,				// stencilStoreOp
			VK_IMAGE_LAYOUT_UNDEFINED,						// initialLayout
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL		// finalLayout
		};

		VkAttachmentDescription colorAttachmentResolve {
			0,												// flags
			swapChainImageFormat,							// format
			VK_SAMPLE_COUNT_1_BIT,							// samples
			VK_ATTACHMENT_LOAD_OP_DONT_CARE,				// loadOp 
			VK_ATTACHMENT_STORE_OP_STORE,					// storeOp
			VK_ATTACHMENT_LOAD_OP_DONT_CARE,				// stencilLoadOp
			VK_ATTACHMENT_STORE_OP_DONT_CARE,				// stencilStoreOp
			VK_IMAGE_LAYOUT_UNDEFINED,						// initialLayout
			VK_IMAGE_LAYOUT_PRESENT_SRC_KHR					// finalLayout 
		};

		VkAttachmentDescription depthAttachment {
			0,												// flags
			FindDepthFormat(),								// format
			msaaSamples,									// samples
			VK_ATTACHMENT_LOAD_OP_CLEAR,					// loadOp 
			VK_ATTACHMENT_STORE_OP_DONT_CARE,				// storeOp
			VK_ATTACHMENT_LOAD_OP_DONT_CARE,				// stencilLoadOp
			VK_ATTACHMENT_STORE_OP_DONT_CARE,				// stencilStoreOp
			VK_IMAGE_LAYOUT_UNDEFINED,						// initialLayout
			VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL// finalLayout 
		};

		VkAttachmentReference colorAttachmentRef {
			0,											// attachment DEBUG: aka the index referenced in "layout(location = 0) out vec4 outColor" of a shader
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL	// layout
		};

		VkAttachmentReference depthAttachmentRef {
			1,													// attachment
			VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL	// layout
		};

		VkAttachmentReference colorAttachmentResolveRef {
			2,											// attachment
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL	// layout
		};

		// TODO (TF 4 FEB 2026): experiment with input, resolve, depthStencil, and preserve attachments
		VkSubpassDescription subpass {
			0,									// flags
			VK_PIPELINE_BIND_POINT_GRAPHICS,	// pipelineBindPoint
			0,									// inputAttachmentCount
			nullptr,							// pInputAttachments
			1,									// colorAttachmentCount
			&colorAttachmentRef,				// pColorAttachments
			&colorAttachmentResolveRef,			// pResolveAttachments // matches the number in colorAttachmentCount (for mulitsampling)
			&depthAttachmentRef,				// pDepthStencilAttachment // only one
			0,									// preserveAttachmentCount
			nullptr								// pPreserveAttachments
		};

		// TODO (TF 5 FEB 2026): experiment with other VkDependencyFlagBits
		VkSubpassDependency dependency {
			VK_SUBPASS_EXTERNAL,								// srcSubpass
			0,													// dstSubpass

			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
			| VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,		// srcStageMask

			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
			| VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,		// dstStageMask

			VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
			| VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,				// srcAccessMask

			VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
			| VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,		// dstAccessmask

			0													// dependencyFlags
		};

		std::array<VkAttachmentDescription, 3> attachments = {
			colorAttachment,
			depthAttachment,
			colorAttachmentResolve
		};

		VkRenderPassCreateInfo renderPassInfo {
			VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,	// sType
			nullptr,									// pNext
			0,											// flags
			static_cast<uint32_t>(attachments.size()),	// attachmentCount
			attachments.data(),							// pAttachments
			1,											// subpassCount
			&subpass,									// pSubpasses
			1,											// dependencyCount
			&dependency									// pDependencies
		};

		VkResult result = vkCreateRenderPass(logicalDevice, &renderPassInfo, nullptr, &renderPass);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}
	}

	VkShaderModule CreateShaderModule(const std::vector<char>& code) {
		// DEBUG: std::vector default allocator ensures data satisfies worst-case alignment requirements
		// so reinterpret_cast from char to uint32_t does not adversly affect byte alignment here
		VkShaderModuleCreateInfo createInfo{
			VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,	// sType
			nullptr,										// pNext
			0,												// flags
			code.size(),									// codeSize
			reinterpret_cast<const uint32_t*>(code.data())	// pCode
		};

		VkShaderModule shaderModule;
		VkResult result = vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &shaderModule);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule; // code buffer no longer needed for this module to function
	}

	void CreateComputePipeline() {
		auto computeShaderCode = ReadFile("Shaders/compute.spv");

		VkShaderModule computeShaderModule = CreateShaderModule(computeShaderCode);

		VkPipelineShaderStageCreateInfo computeShaderStageInfo {
			VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, // sType
			nullptr,											 // pNext
			0,													 // flags
			VK_SHADER_STAGE_COMPUTE_BIT,						 // stage
			computeShaderModule,								 // module
			"main",												 // pName // TODO (TF 4 FEB 2026): experiment with multiple shaders with multiple entry points in one module
			nullptr												 // pSpecializationinfo // TODO (TF 4 FEB 2026): experiment with specialization info (push constants, etc)
		};

		// TODO (TF 4 FEB 2026): experiment with uniforms and push constants
		VkPipelineLayoutCreateInfo pipelineLayoutInfo{
			VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,	// sType
			nullptr,										// pNext
			0,												// flags
			1,												// setLayoutCount
			&computeDescriptorSetLayout,					// pSetLayouts
			0,												// pushConstantRangeCount
			nullptr,										// pPushConstantRanges
		};

		VkResult layoutResult = vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &computePipelineLayout);

		VkComputePipelineCreateInfo pipelineInfo{
			VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,		// sType
			nullptr,											// pNext
			0,													// flags
			computeShaderStageInfo,								// stage
			computePipelineLayout,								// layout
			VK_NULL_HANDLE,										// basePipelineHandle
			-1													// basePipelineIndex
		};

		VkResult pipelineResult = vkCreateComputePipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline);
		if (pipelineResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create compute pipeline!");
		}

		// shader modules no longer needed for this pipeline to function
		vkDestroyShaderModule(logicalDevice, computeShaderModule, nullptr);
	}

	void CreateGraphicsPipeline() {
		// TODO (TF 4 FEB 2026): remove hardcoded shader loading dependency
		//auto vertShaderCode = ReadFile("Shaders/vert.spv");
		//auto fragShaderCode = ReadFile("Shaders/frag.spv");
		auto vertShaderCode = ReadFile("Shaders/particleVert.spv");
		auto fragShaderCode = ReadFile("Shaders/particleFrag.spv");

		VkShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);

		// FIXME (TF 13 FEB 2026): a "PointSize" variable must be written to in pStages array
		// if the pipeline topology is VK_PRIMITIVE_TOPOLOGY_POINT_LIST
		VkPipelineShaderStageCreateInfo vertShaderStageInfo {
			VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, // sType
			nullptr,											 // pNext
			0,													 // flags
			VK_SHADER_STAGE_VERTEX_BIT,							 // stage
			vertShaderModule,									 // module
			"main",												 // pName // TODO (TF 4 FEB 2026): experiment with multiple shaders with multiple entry points in one module
			nullptr												 // pSpecializationinfo // TODO (TF 4 FEB 2026): experiment with specialization info (push constants, etc)
		};

		VkPipelineShaderStageCreateInfo fragShaderStageInfo {
			VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, // sType
			nullptr,											 // pNext
			0,													 // flags
			VK_SHADER_STAGE_FRAGMENT_BIT,						 // stage
			fragShaderModule,									 // module
			"main",												 // pName // TODO (TF 4 FEB 2026): experiment with multiple shaders with multiple entry points in one module
			nullptr												 // pSpecializationinfo // TODO (TF 4 FEB 2026): experiment with specialization info (push constants, etc)
		};

		VkPipelineShaderStageCreateInfo shaderStages[] = {
			vertShaderStageInfo,
			fragShaderStageInfo
		};

		// FIXME (TF 13 FEB 2026): the attributeDescrptions is borked for particles (duplicates?)
		// TODO (TF 6 FEB 2026): automate creating more than one binding
		auto bindingDescription = Particle::GetBindingDescription(0); // Vertex::GetBindingDescription(0);
		auto attributeDescription = Particle::GetAttributeDescriptions(0, 0); // Vertex::GetAttributeDescriptions(0, 0);

		// TODO (TF 4 FEB 2026): use vertex and index buffers instead of hardcoding verticies in shaders
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{
			VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,	// sType
			nullptr,													// pNext
			0,															// flags
			1,															// vertexBindingDescriptionCount
			&bindingDescription,										// pVertexBindingDescriptions
			static_cast<uint32_t>(attributeDescription.size()),			// vertexAttributeDescriptionCount
			attributeDescription.data()									// pVertexAttributeDescriptions
		};

		VkPipelineInputAssemblyStateCreateInfo inputAssembly {
			VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, // sType
			nullptr,													 // pNext
			0,															 // flags
			VK_PRIMITIVE_TOPOLOGY_POINT_LIST,							 // topology // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
			VK_FALSE													 // primitiveRestartEnable;
		};

		// use entire framebuffer area via the swapChainExtent
		VkViewport viewport {
			0,								// x
			0,								// y
			(float)swapChainExtent.width,	// width
			(float)swapChainExtent.height,	// height
			0.0f,							// minDepth
			1.0f							// maxDepth
		};

		// only discard outsize swapChainExtent (ie: keep entire framebuffer)
		VkRect2D scissor {
			{0, 0},			// offset
			swapChainExtent	// extent
		};

		// TODO (TF 4 FEB 2026): experiment with a dynamic state pipeline and using commands to set viewport and scissor
		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineDynamicStateCreateInfo dynamicState {
			VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,	// sType
			nullptr,												// pNext
			0,														// flags
			static_cast<uint32_t>(dynamicStates.size()),			// dynamicStateCount
			dynamicStates.data()									// pDynamicStates
		};

		VkPipelineViewportStateCreateInfo viewportState {
			VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,	// sType
			nullptr,												// pNext
			0,														// flags
			1,														// viewportCount
			&viewport,												// pViewPorts
			1,														// scissorCount
			&scissor												// pScissors
		};

		VkPipelineRasterizationStateCreateInfo rasterizer {
			VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,	// sType
			nullptr,													// pNext
			0,															// flags
			VK_FALSE,													// depthClampEnable // TODO (TF 4 FEB 2026): experiemnt with enabled depth clipping (eg: for shadow maps) (reqires enabled GPU feature)
			VK_FALSE,													// rasterizerDiscardEnable // if true, then geo never hits rasterizer
			VK_POLYGON_MODE_FILL,										// polygonMode // VK_POLYGON_MODE_FILL
			VK_CULL_MODE_BACK_BIT,										// cullMode // VK_CULL_MODE_BACK_BIT
			VK_FRONT_FACE_CLOCKWISE,									// frontFace // VK_FRONT_FACE_COUNTER_CLOCKWISE
			VK_FALSE,													// depthBiasEnable // TODO (TF 4 FEB 2026): experiment with depth bias for shadow maps / z-fighting resolution
			0.0f,														// depthBiasConstantFactor
			0.0f,														// depthBiasClamp
			0.0f,														// depthBiasSlopeFactor
			1.0f														// lineWidth // TODO (TF 4 FEB 2026): experiment with wideLines GPU feature
		};

		// PERF: sampleShadingEnabled degrades performance, but improves internal aliasing
		VkPipelineMultisampleStateCreateInfo multisampling {
			VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, // sType
			nullptr,												  // pNext
			0,														  // flags
			msaaSamples,											  // rasterizationSamples
			VK_TRUE,												  // sampleShadingEnable
			0.2f,													  // minSampleShading // closer to 1.0f is smoother
			nullptr,												  // pSampleMask
			VK_FALSE,												  // alphaToCoverageEnable
			VK_FALSE												  // alphaToOneEnable
		};

		// TODO (TF 4 FEB 2026): experiment with depth and stencil testing
		VkPipelineDepthStencilStateCreateInfo depthStencil {
			VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,	// sType
			nullptr,													// pNext
			0,															// flags
			VK_TRUE,													// depthTestEnable
			VK_TRUE,													// depthWriteEnable
			VK_COMPARE_OP_LESS,											// depthCompareOp;
			VK_FALSE,													// depthBoundsTestEnable
			VK_FALSE,													// stencilTestEnable
			{},															// front (stencilOP)
			{},															// back (stencilOP)
			0.0f,														// minDepthBounds
			1.0f														// maxDepthBounds
		};

		// TODO (TF 4 FEB 2026): experiment with color blending operations
		// per-framebuffer config (alpha mixing old and new values)
		VkPipelineColorBlendAttachmentState colorBlendAttachment {
			VK_FALSE,					// blendEnable
			VK_BLEND_FACTOR_ONE,		// srcColorBlendFactor
			VK_BLEND_FACTOR_ZERO,		// dstColorBlendFactor
			VK_BLEND_OP_ADD,			// colorBlendOp
			VK_BLEND_FACTOR_ONE,		// srcAlphaBlendFactor
			VK_BLEND_FACTOR_ZERO,		// dstAlphaBlendFactor
			VK_BLEND_OP_ADD,			// alphaBlendOp
			VK_COLOR_COMPONENT_R_BIT 
			| VK_COLOR_COMPONENT_G_BIT 
			| VK_COLOR_COMPONENT_B_BIT 
			| VK_COLOR_COMPONENT_A_BIT	// colorWriteMask
		};

		// TODO (TF 4 FEB 2026): experiment with global color blending operations
		// global color blending config (bitwise combination of old and new values)
		// DEBUG: logicOpEnable == VK_TRUE forces all VkPipelineColorBlendAttachmentState::blendEnable == VK_FALSE,
		// however the colorWriteMasks will still be used
		VkPipelineColorBlendStateCreateInfo colorBlending {
			VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,	// sType
			nullptr,													// pNext
			0,															// flags
			VK_FALSE,													// logicOpEnable
			VK_LOGIC_OP_COPY,											// logicOp
			1,															// attachmentCount
			&colorBlendAttachment,										// pAttachments
			{0.0f, 0.0f, 0.0f, 0.0f}									// blendConstants
		};

		// TODO (TF 4 FEB 2026): experiment with uniforms and push constants
		VkPipelineLayoutCreateInfo pipelineLayoutInfo {
			VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,	// sType
			nullptr,										// pNext
			0,												// flags
			1,												// setLayoutCount
			&descriptorSetLayout,							// pSetLayouts
			0,												// pushConstantRangeCount
			nullptr,										// pPushConstantRanges
		};

		VkResult pipelineLayoutCreateResult = vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo, nullptr, &pipelineLayout);
		if (pipelineLayoutCreateResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		// TODO (TF 4 FEB 2026): experiment with VK_PIPELINE_CREATE_DERIVATIVE_BIT
		// and using alternate subpasses https://docs.vulkan.org/spec/latest/chapters/renderpass.html#renderpass-compatibility
		VkGraphicsPipelineCreateInfo pipelineInfo {
			VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,	// sType
			nullptr,											// pNext
			0,													// flags
			2,													// stageCount
			shaderStages,										// pStages
			&vertexInputInfo,									// pVertexInputState
			&inputAssembly,										// pInputAsssemblyState
			nullptr,											// pTessellationState
			&viewportState,										// pViewportState
			&rasterizer,										// pRasterizationState
			&multisampling,										// pMultisampleState
			&depthStencil,										// pDepthStencilState
			&colorBlending,										// pColorBlendState
			&dynamicState,										// pDynamicState
			pipelineLayout,										// layout
			renderPass,											// renderPass
			0,													// subpass
			VK_NULL_HANDLE,										// basePipelineHandle
			-1													// basePipelineIndex
		};

		// TODO (TF 4 FEB 2026): experiment with creating multiple pipelines at once
		// and with using a VkPipelineCache across multiple application runs via file cache
		VkResult pipelineCreateResult = vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline);
		if (pipelineCreateResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		// shader modules no longer needed for this pipeline to function
		vkDestroyShaderModule(logicalDevice, vertShaderModule, nullptr);
		vkDestroyShaderModule(logicalDevice, fragShaderModule, nullptr);
	}

	void CleanupSwapChain() {
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			vkDestroySemaphore(logicalDevice, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(logicalDevice, inFlightFences[i], nullptr);
			vkDestroyFence(logicalDevice, computeInFlightFences[i], nullptr);
			vkDestroySemaphore(logicalDevice, computeFinishedSemaphores[i], nullptr);
		}

		for (size_t i = 0; i < swapChainImages.size(); ++i) {
			vkDestroySemaphore(logicalDevice, renderFinishedSemaphores[i], nullptr);
		}

		vkDestroyImageView(logicalDevice, colorImageView, nullptr);
		vkDestroyImage(logicalDevice, colorImage, nullptr);
		vkFreeMemory(logicalDevice, colorImageMemory, nullptr);

		vkDestroyImageView(logicalDevice, depthImageView, nullptr);
		vkDestroyImage(logicalDevice, depthImage, nullptr);
		vkFreeMemory(logicalDevice, depthImageMemory, nullptr);

		for (auto framebuffer : swapChainFramebuffers) {
			vkDestroyFramebuffer(logicalDevice, framebuffer, nullptr);
		}

		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(logicalDevice, imageView, nullptr);
		}

		// swapchain images are implicitly cleaned up when the swapchain is destroyed
		vkDestroySwapchainKHR(logicalDevice, swapChain, nullptr);
	}

	void RecreateSwapChain() {
		int width = 0;
		int height = 0;
		glfwGetFramebufferSize(window, &width, &height);

		// TODO (TF 6 FEB 2026): do more than a brute force wait until window is 
		// in forground/non-zero size again
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		// TODO (TF 6 FEB 2026): explicitly wait for the oldSwapChain to be available and destroy that
		// meanwhile create a new SwapChain to start rendering to the resized/new window properties
		// ie: pass oldSwapChain into vkCreateSwapchainKHR
		vkDeviceWaitIdle(logicalDevice);

		CleanupSwapChain();

		// DEBUG: the renderpass would also need to be recreated if the swapchain imageformat
		// changed, for example, when application window changes monitors
		CreateSwapChain();
		CreateSwapChainImageViews();
		CreateColorResources();
		CreateDepthResources();
		CreateFramebuffers();
		CreateSyncObjects();
	}

	void CreateSwapChain() {
		SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = ChooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = ChooseSwapExtent(swapChainSupport.capabilities);

		// ensure renderer doesn't need to wait on driver to complete internal operations
		// before acquiring next image to render to (ie: minimum of double buffering)
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

		// maxImageCount == 0 implies no limit
		if (swapChainSupport.capabilities.maxImageCount > 0
			&& imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		QueueFamilyIndices indices = FindQueueFamilies(physicalDevice);
		std::vector<uint32_t> queueFamilyIndices = {
			indices.graphicsAndComputeFamily.value(),
			indices.transferFamily.value()
		};

		if (indices.presentFamily != indices.graphicsAndComputeFamily) {
			queueFamilyIndices.push_back(indices.presentFamily.value());
		}

		// DEBUG: exclusive is best performance option for sharing, 
		// but **requires explicit sync/transfer of ownership**
		VkSwapchainCreateInfoKHR createInfo {
			VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,		// sType
			nullptr,											// pNext
			0,													// flags
			surface,											// surface
			imageCount,											// minImageCount
			surfaceFormat.format,								// imageFormat
			surfaceFormat.colorSpace,							// imageColorSpace
			extent,												// imageExtent
			1,													// imageArrayLayers // TODO (TF 3 FEB 2026): make more than 1 for stereoscopic 3D
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,				// imageUsage // TODO (TF 3 FEB 2026): test VK_IMAGE_USAGE_TRANSFER_DST_BIT here for post-processing effects into the swapchain images
			VK_SHARING_MODE_CONCURRENT,							// imageSharingMode
			static_cast<uint32_t>(queueFamilyIndices.size()),	// queueFamilyIndexCount
			queueFamilyIndices.data(),							// pQueueFamilyIndices
			swapChainSupport.capabilities.currentTransform,		// preTransform // TODO (TF 3 FEB 2026): adapt this for rotated and/or flipped setups
			VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,					// compositeAlpha // DEBUG: opaque to avoid using alpha to blen with other windows
			presentMode,										// presentMode
			VK_TRUE,											// clipped // TODO (TF 3 FEB 2026): disable clipping to ensure pixels are readable even if another window obscures the application
			VK_NULL_HANDLE										// oldSwapChain // TODO (TF 3 FEB 2026): pass in the old/invalid swapchain if window is resized
		};

		VkResult result = vkCreateSwapchainKHR(logicalDevice, &createInfo, nullptr, &swapChain);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		// cache swapchain images
		vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(logicalDevice, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
		// TODO(? TF 3 FEB 2026): cache the surfaceFormat.colorspace too, for XR applications
	}

	void CreateSwapChainImageViews() {
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); ++i) {
			swapChainImageViews[i] = CreateImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
		}
	}

	bool CheckDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());
		for (const auto& extension : availableExtensions) {
			uint32_t erasedCount = requiredExtensions.erase(extension.extensionName);
			if (erasedCount > 0) {
				std::cout << "found required device extension [" << extension.extensionName << "]" << std::endl;
			}
		}

		// all extensions supported if empty
		return requiredExtensions.empty();
	}

	// TODO (TF 2 FEB 2026): modify the ranking to fit a unique application
	// ...generally most hosts will have 1 or 2 GPUs, so
	// at minimum if any device is returned then it does support Vulkan.
	int RateDeviceSuitability(VkPhysicalDevice device) {
		// Basic device properties like the name, 
		// type and supported Vulkan version can be queried using this
		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(device, &deviceProperties);

		// The support for optional features like texture compression, 
		// 64 bit floats and multi viewport rendering (useful for VR) can be queried using this
		VkPhysicalDeviceFeatures deviceFeatures;
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

		int score = 0;

		if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			score += 1000;
		}

		// maximum possible size of textures affects graphics quality
		score += deviceProperties.limits.maxImageDimension2D;

		if (!deviceFeatures.geometryShader) {
			return 0;
		}

		if (!deviceFeatures.samplerAnisotropy) {
			return 0;
		}

		// must support at least graphics queues (for now)
		QueueFamilyIndices indices = FindQueueFamilies(device);
		if (!indices.IsComplete()) {
			return 0;
		}

		bool extensionsSupported = CheckDeviceExtensionSupport(device);
		if (extensionsSupported) {
			bool swapChainAdequate = false;

			SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(device);

			// TODO (TF 3 FEB 2026): be more specific in swapchain requirements (min and max)
			swapChainAdequate = !swapChainSupport.formats.empty()
								&& !swapChainSupport.presentModes.empty();

			if (!swapChainAdequate) {
				return 0;
			}
		}
		else {
			return 0;
		}

		std::cout << "checking GPU [" << deviceProperties.deviceName << "][" << device << "]\n";
		return score;
	}

	void PickPhysicalDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		std::multimap<int, VkPhysicalDevice> candidates;

		for (const auto& device : devices) {
			int score = RateDeviceSuitability(device);
			candidates.insert(std::make_pair(score, device));
		}

		if (candidates.crbegin()->first > 0) {
			physicalDevice = candidates.crbegin()->second;
			msaaSamples = GetMaxUsableSampleCount();
			std::cout << "picking GPU [" << physicalDevice << "] with SAMPLES [" << msaaSamples <<"]" << std::endl;
		}
		else {
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	/// <summary>
	/// Iterates the physical device's queue family indices and caches which ones support which properties
	/// PERF: it is more efficient to use a single queue family index which supports both graphics and presentation
	/// instead of two separate queue familily indeces
	/// </summary>
	QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;
		uint32_t queueFamilyCount = 0;

		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		// will pick the first device family that supports graphics queues
		int queueFamilyIndex = 0;
		for (const auto& queueFamily : queueFamilies) {
			VkBool32 presentSupport = false;

			// TODO(? TF 2 FEB 2026): call may fail for unknown reasons, may be worth reacting to here
			/*VkResult callSuccess = */vkGetPhysicalDeviceSurfaceSupportKHR(device, queueFamilyIndex, surface, &presentSupport);

			if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
				&& (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
				indices.graphicsAndComputeFamily = queueFamilyIndex;
			}

			if (presentSupport) {
				indices.presentFamily = queueFamilyIndex;
			}

			if ((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0
				&& (queueFamily.queueFlags & VK_QUEUE_TRANSFER_BIT)) {
				indices.transferFamily = queueFamilyIndex;
			}

			if (indices.IsComplete()) {
				break;
			}
			queueFamilyIndex++;
		}

		return indices;
	}

	SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;
		uint32_t formatCount;
		uint32_t presentModeCount;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (formatCount > 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		if (presentModeCount > 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	// TODO (TF 3 FEB 2026): select different format and colorspace combination as needed (not hardcoded)
	VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB
				&& availableFormat.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}

		return availableFormats[0]; // TODO (TF 3 FEB 2026): don't default to the first surfaceFormat available
	}

	// TODO (TF 3 FEB 2026): explore different modes (minimize tearing)
	VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR; // FIFO guaranteed to be present by vulkan
	}

	VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		// max uint32 would indicate currentExtent can be customized
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent; 
		}

		int width;
		int height;
		glfwGetFramebufferSize(window, &width, &height); // extent in pixel units

		VkExtent2D actualExtent = {
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};

		actualExtent.width = std::clamp(actualExtent.width,
										capabilities.minImageExtent.width,
										capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height,
										capabilities.minImageExtent.height,
										capabilities.maxImageExtent.height);

		return actualExtent;
	}

	void CreateLogicalDevice() {
		QueueFamilyIndices indices = FindQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = {
			indices.graphicsAndComputeFamily.value(),
			indices.presentFamily.value(),
			indices.transferFamily.value()
		};

		// DEBUG: this may result in a single VkDeviceQueueCreateInfo entry if
		// a device queue family index was found which supports both graphics and presentation
		float queuePriority = 1.0f; // 0.0 to 1.0f (required, even for 1 queue)
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{
				VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, // sType
				nullptr,									// pNext
				0,											// flags
				queueFamily,								// queueFamilyIndex
				1,											// queueCount
				&queuePriority								// pQueuePriorities
			};
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures	{
			// TODO (TF 2 FEB 2026): left all VK_FALSE for now
		}; 
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		deviceFeatures.sampleRateShading = VK_TRUE; // costly, but improves internal aliasing

		// Logical device layers are deprecated in favor of instance layers
		// this is only here for legacy support
		uint32_t enabledLayerCount = 0;
		const char* const* ppEnabledLayerNames = nullptr;
		if (enableValidationLayers) {
			enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
			ppEnabledLayerNames = VALIDATION_LAYERS.data();
		}

		uint32_t enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
		const char* const* ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();

		VkDeviceCreateInfo createInfo {
			VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,			// sType
			nullptr,										// pNext
			0,												// flags
			static_cast<uint32_t>(queueCreateInfos.size()),	// queueCreateInfoCount
			queueCreateInfos.data(),						// pQueueCreateInfos
			enabledLayerCount,								// enabledLayerCount
			ppEnabledLayerNames,							// ppEnabledLayerNames
			enabledExtensionCount,							// enabledExtensionCount
			ppEnabledExtensionNames,						// ppEnabledExtensionNames
			&deviceFeatures									// pEnabledFeatures
		};

		VkResult result = vkCreateDevice(physicalDevice, &createInfo, nullptr, &logicalDevice);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create locial device!");
		}

		// get the handles for the queues to which work will be submitted
		// (default indices for future use will be 0)
		// DEBUG: if only one queue family index was used for creation, then both handles will be identical
		vkGetDeviceQueue(logicalDevice, indices.graphicsAndComputeFamily.value(), defaultQueueIndex, &graphicsAndComputeQueue);
		vkGetDeviceQueue(logicalDevice, indices.presentFamily.value(), defaultQueueIndex, &presentQueue);
		vkGetDeviceQueue(logicalDevice, indices.transferFamily.value(), defaultQueueIndex, &transferQueue);
	}

	void PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {
			VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,	// sType
			nullptr,													// pNext
			0,															// flags
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
			//VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,				// messageSeverity
			VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			//VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT | // DEBUG: requires the extensions VK_EXT_device_address_binding_report
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,			// messageType
			DebugCallback,												// pfnUserCallback
			nullptr														// pUserData
		};
	}

	void SetupDebugMessenger() {
		if (!enableValidationLayers) {
			return;
		}

		VkDebugUtilsMessengerCreateInfoEXT createInfo{};
		PopulateDebugMessengerCreateInfo(createInfo);

		VkResult result = CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	
	}

	void RecordComputeCommandBuffer(VkCommandBuffer commandBuffer) {
		VkCommandBufferBeginInfo beginInfo{
			VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			nullptr,
			0,
			nullptr
		};

		VkResult beginBufferResult = vkBeginCommandBuffer(commandBuffer, &beginInfo);
		if (beginBufferResult != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording compute command buffer!");
		}

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
		vkCmdBindDescriptorSets(commandBuffer, 
								VK_PIPELINE_BIND_POINT_COMPUTE,
								computePipelineLayout, 0, 1, &computeDescriptorSets[currentFrame], 0, nullptr);
		
		vkCmdDispatch(commandBuffer, WORKGROUP_SIZE_X, 1, 1);

		VkResult endBufferResult = vkEndCommandBuffer(commandBuffer);
		if (endBufferResult != VK_SUCCESS) {
			throw std::runtime_error("failed to record compute command buffer!");
		}
	}

	void RecordGraphicsCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
		// TODO (TF 4 FEB 2026): experiment with command buffer usage flags (eg: secondary buffer records, quick resubmit/rerecord)
		// TODO (TF 4 FEB 2026): experiment with secondary buffer inheritance
		VkCommandBufferBeginInfo beginInfo {
			VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,	// sType
			nullptr,										// pNext
			0,												// flags
			nullptr											// pInheritanceInfo
		};

		VkResult beginBufferResult = vkBeginCommandBuffer(commandBuffer, &beginInfo);
		if (beginBufferResult != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording graphics command buffer!");
		}

		VkRect2D renderArea = {
			{0, 0},
			swapChainExtent
		};

		// DEBUG: ensure order of clearValues matches order of renderPass attachments
		std::array<VkClearValue, 2> clearValues {};
		clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}}; // union size of 4x 32-bit values
		clearValues[1].depthStencil = {1.0f, 0}; // DEBUG: intialize to furthest possible depth (1.0f)

		VkRenderPassBeginInfo renderPassInfo {
			VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,	// sType
			nullptr,									// pNext
			renderPass,									// renderpass
			swapChainFramebuffers[imageIndex],			// framebuffer
			renderArea,									// renderArea
			static_cast<uint32_t>(clearValues.size()),	// clearValueCount
			clearValues.data()							// pClearValues
		};

		// TODO (TF 5 FEB 2026): experiment with secondary command buffer execution
		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		//VkBuffer vertexBuffers[] = { vertexBuffer };
		//VkDeviceSize offsets[] = { 0 };
		//vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		//vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &shaderStorageBuffers[currentFrame], offsets);

		// configure dynamic states (viewport and scissor)
		VkViewport viewport {
			0.0f,										// x
			0.0f,										// y
			static_cast<float>(swapChainExtent.width),	// width
			static_cast<float>(swapChainExtent.height),	// height
			0.0f,										// minDepth
			1.0f										// maxDepth
		};
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor {
			{0, 0},			// offset
			swapChainExtent	// extent
		};
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		// TODO (TF 13 FEB 2026): commented out model shader descriptor set to allow particle system to render
		//vkCmdBindDescriptorSets(graphicsAndComputeCommandBuffers[currentFrame], 
		//						VK_PIPELINE_BIND_POINT_GRAPHICS, 
		//						pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

		//vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
		vkCmdDraw(commandBuffer, PARTICLE_COUNT, 1, 0, 0);
		vkCmdEndRenderPass(commandBuffer);

		VkResult endBufferResult = vkEndCommandBuffer(commandBuffer);
		if (endBufferResult != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}
	}

	void MainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			DrawFrame();
		}

		// TODO (TF 5 FEB 2026): alternately wait with vkQueueWaitIdle on both queues
		vkDeviceWaitIdle(logicalDevice);
	}

	void UpdateTransformUniformBuffer(uint32_t currentImage) {
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		// TODO (TF 8 FEB 2026): remove hardcoded behavior
		glm::f32 rotationRate = time * glm::radians(90.0f);
		constexpr float fovy = glm::radians(45.0f);
		float aspect = swapChainExtent.width / (float)swapChainExtent.height;
		glm::vec3 upAxis = glm::vec3(0.0f, 0.0f, 1.0f);

		TransformUBO transformUBO{};
		transformUBO.model = glm::rotate(glm::mat4(1.0f), rotationRate , upAxis);
		transformUBO.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), upAxis);
		transformUBO.proj = glm::perspective(fovy, aspect , 0.1f, 10.0f);

		// DEBUG: this will cause verticies to be drawn in counter-clockwise order (ie: trigger backface culling)
		transformUBO.proj[1][1] *= -1; // flip GLM's openGL y-axis to match vulkan y-axis

		// TODO (TF 8 FEB 2026) instead of using a persistent memory mapped uniform buffer
		// to update this data every frame, experiment with the more performant push constants
		memcpy(transformUniformBuffersMapped[currentImage], &transformUBO, sizeof(transformUBO));
		// DEBUG: cache of this write need not be synced because it is using HOST_COHERENT_BIT
	}

	void UpdateTimeUniformBuffer(uint32_t currentImage) {
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		TimeUBO timeUBO{
			time
		};

		// TODO (TF 8 FEB 2026) instead of using a persistent memory mapped uniform buffer
		// to update this data every frame, experiment with the more performant push constants
		memcpy(timeUniformBuffersMapped[currentImage], &timeUBO, sizeof(timeUBO));
		// DEBUG: cache of this write need not be synced because it is using HOST_COHERENT_BIT
	}

	void DrawFrame() {
		// ========== BEGIN COMPUTE WORK ==========
		// TODO (TF 5 FEB 2026) Experiment with timeline semaphores
		vkWaitForFences(logicalDevice, 1, &computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		
		UpdateTimeUniformBuffer(currentFrame);
		// TODO (**STOPPED HERE** TF 12 FEB 2026): "Synchronizing graphics and compute"
		// at https://vulkan-tutorial.com/Compute_Shader#page_Compute-shaders
		// once that is complete, remember to change the rendering to points instead of triangles
		// and disable the model loading, etc in InitVulkan

		vkResetFences(logicalDevice, 1, &computeInFlightFences[currentFrame]);
		vkResetCommandBuffer(graphicsAndComputeCommandBuffers[currentFrame], 0);

		RecordComputeCommandBuffer(graphicsAndComputeCommandBuffers[currentFrame]);

		// FIXME (1/2) (TF 13 FEB 2026): if currentFrame throws validation errors,
		// then use imageIndex of the current swapChain image for the signal semaphore index
		VkSubmitInfo computeSubmitInfo{
			VK_STRUCTURE_TYPE_SUBMIT_INFO,
			nullptr,
			0, nullptr, nullptr,
			1, &graphicsAndComputeCommandBuffers[currentFrame],
			1, &computeFinishedSemaphores[currentFrame] 
		};

		VkResult computeSubmitResult = vkQueueSubmit(graphicsAndComputeQueue, 1, &computeSubmitInfo, computeInFlightFences[currentFrame]);
		if (computeSubmitResult != VK_SUCCESS) {
			throw std::runtime_error("failed to submit compute command buffer!");
		}
		// ========== END COMPUTE WORK ==========
		 
		// ========== BEGIN RENDER WORK ==========
		// TODO (TF 13 FEB 2026): instead of waiting for computeInFlightFences,
		// maybe just use different commandBuffers so they aren't prematurely reset
		VkFence waitFences[] = {
			computeInFlightFences[currentFrame],
			inFlightFences[currentFrame]
		};

		vkWaitForFences(logicalDevice, 2, waitFences, VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult acquireResult = vkAcquireNextImageKHR(logicalDevice, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
		
		if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
			RecreateSwapChain();
			return;
		}
		else if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swapchain image!");
		}
		
		// only reset the fence once work can be submitted on the swapchain image
		vkResetFences(logicalDevice, 1, &inFlightFences[currentFrame]);
		
		vkResetCommandBuffer(graphicsAndComputeCommandBuffers[currentFrame], 0); // TODO (TF 5 FEB 2026): experiment with releasing resources on reset

		//UpdateTransformUniformBuffer(currentFrame);

		RecordGraphicsCommandBuffer(graphicsAndComputeCommandBuffers[currentFrame], imageIndex);

		// FIXME (2/2) (TF 13 FEB 2026): if currentFrame throws validation errors,
		// then use imageIndex of the current swapChain image for the signal semaphore index
		std::array<VkSemaphore, 2> waitSemaphores = { 
			computeFinishedSemaphores[currentFrame],
			imageAvailableSemaphores[currentFrame]
		};

		// DEBUG: ensure the rendering semaphores are available to be signaled by indexing to swapChain images directly
		// ie: vkQueuePresentKHR doesn't have pSignalSemaphores like vkQueueSubmit does, so this DrawFrame function
		// relies on a Fence to assume the gpu semaphores are ready, which is not guaranted when only indexing by frames-in-flight
		VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[imageIndex]};

		VkPipelineStageFlags waitStages[] = {
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
		};

		VkSubmitInfo submitInfo {
			VK_STRUCTURE_TYPE_SUBMIT_INFO,						// sType
			nullptr,											// pNext
			static_cast<uint32_t>(waitSemaphores.size()),		// waitSemaphoreCount
			waitSemaphores.data(),								// pWaitSemaphores
			waitStages,											// pWaitDstStageMask	// TODO (TF 5 FEB 2026): experiment with other pipeline stage flags here
			1,													// commandBufferCount
			&graphicsAndComputeCommandBuffers[currentFrame],	// pCommandBuffers
			1,													// signalSemaphoreCount
			signalSemaphores									// pSignalSemaphores
		};

		VkResult submitResult = vkQueueSubmit(graphicsAndComputeQueue, 1, &submitInfo, inFlightFences[currentFrame]);
		if (submitResult != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer!");
		}

		VkSwapchainKHR swapChains[] = {swapChain};

		VkPresentInfoKHR presentInfo {
			VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,		// sType
			nullptr,								// pNext
			1,										// waitSemaphoreCount
			signalSemaphores,						// pWaitSemaphores
			1,										// swapchainCount
			swapChains,								// pSwapchains
			&imageIndex,							// pImageIndices
			nullptr									// pResults
		};

		VkResult presentResult = vkQueuePresentKHR(presentQueue, &presentInfo);

		// DEBUG: ensure the explicit framebufferResized check occurs after vkQueuePresentKHR call
		// so that waitSemaphores are actually waited on
		if (presentResult == VK_ERROR_OUT_OF_DATE_KHR 
			|| presentResult == VK_SUBOPTIMAL_KHR
			|| framebufferResized) {
			framebufferResized = false;
			RecreateSwapChain();
		}
		else if (presentResult != VK_SUCCESS) {
			throw std::runtime_error("failed to present swapchain image!");
		}

		// keep multiple frames in flight so CPU isn't waiting too long on the GPU
		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
		// ========== END RENDER WORK ==========
	}

	void Cleanup() {
		CleanupSwapChain();

		vkDestroySampler(logicalDevice, textureSampler, nullptr);
		vkDestroyImageView(logicalDevice, textureImageView, nullptr);

		vkDestroyImage(logicalDevice, textureImage, nullptr);
		vkFreeMemory(logicalDevice, textureImageMemory, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			//vkDestroyBuffer(logicalDevice, transformUniformBuffers[i], nullptr);
			//vkFreeMemory(logicalDevice, transformUniformBuffersMemory[i], nullptr);

			vkDestroyBuffer(logicalDevice, timeUniformBuffers[i], nullptr);
			vkFreeMemory(logicalDevice, timeUniformBuffersMemory[i], nullptr);

			vkDestroyBuffer(logicalDevice, shaderStorageBuffers[i], nullptr);
			vkFreeMemory(logicalDevice, shaderStorageBuffersMemory[i], nullptr);
		}

		// all descriptor sets are implicitly cleaned up when the descriptor pool is destroyed
		vkDestroyDescriptorPool(logicalDevice, descriptorPool, nullptr);
		vkDestroyDescriptorPool(logicalDevice, computeDescriptorPool, nullptr);

		vkDestroyDescriptorSetLayout(logicalDevice, descriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(logicalDevice, computeDescriptorSetLayout, nullptr);

		vkDestroyBuffer(logicalDevice, vertexBuffer, nullptr);
		vkFreeMemory(logicalDevice, vertexBufferMemory, nullptr);

		vkDestroyBuffer(logicalDevice, indexBuffer, nullptr);
		vkFreeMemory(logicalDevice, indexBufferMemory, nullptr);

		vkDestroyPipeline(logicalDevice, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);

		vkDestroyPipeline(logicalDevice, computePipeline, nullptr);
		vkDestroyPipelineLayout(logicalDevice, computePipelineLayout, nullptr);

		vkDestroyRenderPass(logicalDevice, renderPass, nullptr);
		
		// all command buffers are implicitly cleaned up when the command pool is destroyed
		vkDestroyCommandPool(logicalDevice, graphicsAndComputeCommandPool, nullptr);
		vkDestroyCommandPool(logicalDevice, transferCommandPool, nullptr);

		// device queues are implicitly cleaned up when the devices is destroyed
		vkDestroyDevice(logicalDevice, nullptr);

		if (enableValidationLayers) {
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		// destroy in reverse order of creation
		vkDestroySurfaceKHR(instance, surface, nullptr);

		// DEBUG: physicalDevice handles are implicitly cleaned up with the instance is destroyed
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);
		glfwTerminate();
	}

private:
	VkInstance instance;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice logicalDevice;
	VkQueue graphicsAndComputeQueue;
	VkQueue presentQueue;
	VkQueue transferQueue;
	uint32_t defaultQueueIndex = 0;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;
	GLFWwindow* window;

	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;

	VkRenderPass renderPass;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;

	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkCommandPool graphicsAndComputeCommandPool;
	VkCommandPool transferCommandPool; // only used to create transient command buffers
	std::vector<VkCommandBuffer> graphicsAndComputeCommandBuffers;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;

	std::vector<VkBuffer> transformUniformBuffers;
	std::vector<VkDeviceMemory> transformUniformBuffersMemory; // FIXME (TF 8 FEB 2026): use a single block of memory for all buffers
	std::vector<void*> transformUniformBuffersMapped;

	uint32_t mipLevels;
	VkImage textureImage;
	VkDeviceMemory textureImageMemory;
	VkImageView textureImageView;
	VkSampler textureSampler;

	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;

	VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
	VkImage colorImage;
	VkDeviceMemory colorImageMemory;
	VkImageView colorImageView;

	std::vector<VkBuffer> timeUniformBuffers;
	std::vector<VkDeviceMemory> timeUniformBuffersMemory; // FIXME (TF 8 FEB 2026): use a single block of memory for all buffers
	std::vector<void*> timeUniformBuffersMapped;

	std::vector<Particle> particles;
	std::vector<VkBuffer> shaderStorageBuffers;
	std::vector<VkDeviceMemory> shaderStorageBuffersMemory;
	VkDescriptorSetLayout computeDescriptorSetLayout;
	VkDescriptorPool computeDescriptorPool;
	std::vector<VkDescriptorSet> computeDescriptorSets;
	VkPipelineLayout computePipelineLayout;
	VkPipeline computePipeline;

	std::vector<VkFence> computeInFlightFences;
	std::vector<VkSemaphore> computeFinishedSemaphores;

	uint32_t currentFrame = 0;
	bool framebufferResized = false;
};

int main() {
	HelloTriangleApplication app;

	try {
		app.Run();
	}
	catch (const std::exception& e) {
		std::cout << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}