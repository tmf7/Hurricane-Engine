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

#include <glm/glm.hpp>
#include <array>

struct Vertex {
	glm::vec2 pos;
	glm::vec3 color;

	// TODO (TF 6 FEB 2026): pass in binding == 0 to indicate the singlar vertex buffer binding's index
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
	static std::array<VkVertexInputAttributeDescription, 2> GetAttributeDescriptions(uint32_t binding, uint32_t location) {
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

		attributeDescriptions[0].binding = binding;
		attributeDescriptions[0].location = location;
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = binding;
		attributeDescriptions[1].location = location + 1; // DEBUG: prior attribute only takes up one 32-bit slot
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);


		return attributeDescriptions;
	}

};

const std::vector<Vertex> vertices = {
	{{0.0f, -0.5f}, {1.0f, 0.0, 0.0f}},
	{{0.5f, 0.5f}, {0.0f, 1.0, 0.0f}},
	{{-0.5f, 0.5f}, {0.0f, 0.0, 1.0f}}
};

const uint32_t WINDOW_WIDTH = 800;
const uint32_t WINDOW_HEIGHT = 600;
const int MAX_FRAMES_IN_FLIGHT = 2;

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
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool IsComplete() const {
			return graphicsFamily.has_value() && presentFamily.has_value();
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
		CreateGraphicsPipeline();
		CreateFramebuffers();
		CreateCommandPool();

		CreateVertexBuffer();
		CreateCommandBuffers();
		
		CreateSyncObjects();
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

	void CreateVertexBuffer() {
		VkBufferCreateInfo bufferInfo {
			VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,	// sType
			nullptr,								// pnext
			0,										// flags  // DEBUG: not a sparce buffer for now
			sizeof(vertices[0]) * vertices.size(),	// size
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,		// usage
			VK_SHARING_MODE_EXCLUSIVE,				// sharingMode // DEBUG: only used by the graphicsQueue for now
			0,										// queueFamilyIndexCount
			nullptr									// pQueueFamilyIndices
		};

		VkResult bufferResult = vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &vertexBuffer);
		if (bufferResult != VK_SUCCESS) {
			throw std::runtime_error("failed to create vertex buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(logicalDevice, vertexBuffer, &memRequirements);

		uint32_t memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		VkMemoryAllocateInfo allocInfo {
			VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,		// sType
			nullptr,									// pNext
			memRequirements.size,						// allocationSize
			memoryTypeIndex								// memoryTypeIndex
		};

		VkResult memoryResult = vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &vertexBufferMemory);
		if (memoryResult != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate vertex buffer memory!");
		}

		// DEBUG: singlar vertex buffer and memory binding so offset is 0
		vkBindBufferMemory(logicalDevice, vertexBuffer, vertexBufferMemory, 0);
	}

	void CreateSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

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
			
			if (imageSempaphoreResult != VK_SUCCESS
				|| frameFenceResult != VK_SUCCESS) {
				throw std::runtime_error("failed to create synchronization objects for a frame!");
			}
		}
		for (size_t i = 0; i < swapChainImages.size(); ++i) {
			VkResult finishedSemaphoreResult = vkCreateSemaphore(logicalDevice, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]);
			if (finishedSemaphoreResult != VK_SUCCESS) {
				throw std::runtime_error("failed to create synchronization objects for a frame!");
			}
		}
	}

	void CreateCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		// TODO (TF 5 FEB 2026): expierment with seconday command buffers
		VkCommandBufferAllocateInfo allocInfo {
			VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,		// sType
			nullptr,											// pNext
			commandPool,										// commandPool
			VK_COMMAND_BUFFER_LEVEL_PRIMARY,					// level
			static_cast<uint32_t>(commandBuffers.size())		// commandBufferCount
		};

		VkResult result = vkAllocateCommandBuffers(logicalDevice, &allocInfo, commandBuffers.data());
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}
	}

	void CreateCommandPool() {
		QueueFamilyIndices queueFamilyIndices = FindQueueFamilies(physicalDevice);

		VkCommandPoolCreateInfo poolInfo {
			VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,			// sType
			nullptr,											// pNext
			VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,	// flags
			queueFamilyIndices.graphicsFamily.value()			// queueFamilyIndex
		};

		VkResult result = vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &commandPool);
		if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}
	}

	void CreateFramebuffers() {
		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (size_t i = 0; i < swapChainImageViews.size(); ++i) {
			VkImageView attachments[] = {
				swapChainImageViews[i]
			};

			VkFramebufferCreateInfo framebufferInfo {
				VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,	// sType
				nullptr,									// pNext
				0,											// flags
				renderPass,									// renderPass
				1,											// attachmentCount
				attachments,								// pAttachments
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
			0,									// flags
			swapChainImageFormat,				// format
			VK_SAMPLE_COUNT_1_BIT,				// samples
			VK_ATTACHMENT_LOAD_OP_CLEAR,		// loadOp // TODO (TF 4 FEB 2026): experiment with not clearning the image before rendering to it
			VK_ATTACHMENT_STORE_OP_STORE,		// storeOp
			VK_ATTACHMENT_LOAD_OP_DONT_CARE,	// stencilLoadOp
			VK_ATTACHMENT_STORE_OP_DONT_CARE,	// stencilStoreOp
			VK_IMAGE_LAYOUT_UNDEFINED,			// initialLayout
			VK_IMAGE_LAYOUT_PRESENT_SRC_KHR		// finalLayout // TODO (TF 4 FEB 2026): experiment with VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL for memory copy instead of presentation
		};

		VkAttachmentReference colorAttachmentRef {
			0,											// attachment DEBUG: aka the index referenced in "layout(location = 0) out vec4 outColor" of a shader
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
			nullptr,							// pResolveAttachments // matches the number in colorAttachmentCount (for mulitsampling)
			nullptr,							// pDepthStencilAttachment // only one
			0,									// preserveAttachmentCount
			nullptr								// pPreserveAttachments
		};

		// TODO (TF 5 FEB 2026): experiment with other VkDependencyFlagBits
		VkSubpassDependency dependency {
			VK_SUBPASS_EXTERNAL,								// srcSubpass
			0,													// dstSubpass
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,		// srcStageMask
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,		// dstStageMask
			0,													// srcAccessMask
			VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,				// dstAccessmask
			0													// dependencyFlags
		};


		VkRenderPassCreateInfo renderPassInfo {
			VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,	// sType
			nullptr,									// pNext
			0,											// flags
			1,											// attachmentCount
			&colorAttachment,							// pAttachments
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

	void CreateGraphicsPipeline() {
		// TODO (TF 4 FEB 2026): remove hardcoded shader loading dependency
		auto vertShaderCode = ReadFile("Shaders/vert.spv");
		auto fragShaderCode = ReadFile("Shaders/frag.spv");
		
		VkShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);


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

		// TODO (TF 6 FEB 2026): automate creating more than one binding
		auto bindingDescription = Vertex::GetBindingDescription(0); 
		auto attributeDescription = Vertex::GetAttributeDescriptions(0, 0);


		// TODO (TF 4 FEB 2026): use vertex and index buffers instead of hardcoding verticies in shaders
		VkPipelineVertexInputStateCreateInfo vertexInputInfo{
			VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,	// sType
			nullptr,													// pNext
			0,															// flags
			1,															// vertexBindingDescriptionCount
			&bindingDescription,										// pVertexBindingDescriptions
			1,															// vertexAttributeDescriptionCount
			attributeDescription.data()									// pVertexAttributeDescriptions
		};

		VkPipelineInputAssemblyStateCreateInfo inputAssembly {
			VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, // sType
			nullptr,													 // pNext
			0,															 // flags
			VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,						 // topology
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
			VK_POLYGON_MODE_FILL,										// polygonMode // TODO: (TF 4 FEB 2026): experiment with line and point modes
			VK_CULL_MODE_BACK_BIT,										// cullMode // TODO (TF 4 FEB 2026): experiment with cull off, etc
			VK_FRONT_FACE_CLOCKWISE,									// frontFace
			VK_FALSE,													// depthBiasEnable // TODO (TF 4 FEB 2026): experiment with depth bias for shadow maps / z-fighting resolution
			0.0f,														// depthBiasConstantFactor
			0.0f,														// depthBiasClamp
			0.0f,														// depthBiasSlopeFactor
			1.0f														// lineWidth // TODO (TF 4 FEB 2026): experiment with wideLines GPU feature
		};

		// TODO (TF 4 FEB 2026): experiment with multisampling
		VkPipelineMultisampleStateCreateInfo multisampling {
			VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, // sType
			nullptr,												  // pNext
			0,														  // flags
			VK_SAMPLE_COUNT_1_BIT,									  // rasterizationSamples
			VK_FALSE,												  // sampleShadingEnable
			1.0f,													  // minSampleShading
			nullptr,												  // pSampleMask
			VK_FALSE,												  // alphaToCoverageEnable
			VK_FALSE												  // alphaToOneEnable
		};

		// TODO (TF 4 FEB 2026): experiment with depth and stencil testing
		//VkPipelineDepthStencilStateCreateInfo depthStencil {
		//	VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,	// sType
		//	nullptr,													// pNext
		//	0,															// flags
		//	// depthTestEnable
		//	// depthWriteEnable
		//	// depthCompareOp;
		//	// depthBoundsTestEnable
		//	// stencilTestEnable
		//	// front
		//	// back
		//	// minDepthBounds
		//	// maxDepthBounds
		//};

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
			0,												// setLayoutCount
			nullptr,										// pSetLayouts
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
			nullptr,											// pDepthStencilState
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
		}

		for (size_t i = 0; i < swapChainImages.size(); ++i) {
			vkDestroySemaphore(logicalDevice, renderFinishedSemaphores[i], nullptr);
		}

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
		uint32_t queueFamilyIndices[] = {
			indices.graphicsFamily.value(),
			indices.presentFamily.value()
		};

		// DEBUG: best performance option for sharing, 
		// but **requires explicit sync/transfer of ownership**
		VkSharingMode imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; 
		uint32_t queueFamilyIndexCount = 0;
		uint32_t* pQueueFamilyIndices = nullptr;

		// check if swapchain images cn be used across queue families
		if (indices.graphicsFamily != indices.presentFamily) {
			imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			queueFamilyIndexCount = 2;
			pQueueFamilyIndices = queueFamilyIndices;
		}

		VkSwapchainCreateInfoKHR createInfo {
			VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,   // sType
			nullptr,									   // pNext
			0,											   // flags
			surface,									   // surface
			imageCount,									   // minImageCount
			surfaceFormat.format,						   // imageFormat
			surfaceFormat.colorSpace,					   // imageColorSpace
			extent,										   // imageExtent
			1,											   // imageArrayLayers // TODO (TF 3 FEB 2026): make more than 1 for stereoscopic 3D
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,		   // imageUsage // TODO (TF 3 FEB 2026): test VK_IMAGE_USAGE_TRANSFER_DST_BIT here for post-processing effects into the swapchain images
			imageSharingMode,							   // imageSharingMode
			queueFamilyIndexCount,						   // queueFamilyIndexCount
			pQueueFamilyIndices,						   // pQueueFamilyIndices
			swapChainSupport.capabilities.currentTransform,// preTransform // TODO (TF 3 FEB 2026): adapt this for rotated and/or flipped setups
			VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,			   // compositeAlpha // DEBUG: opaque to avoid using alpha to blen with other windows
			presentMode,								   // presentMode
			VK_TRUE,									   // clipped // TODO (TF 3 FEB 2026): disable clipping to ensure pixels are readable even if another window obscures the application
			VK_NULL_HANDLE								   // oldSwapChain // TODO (TF 3 FEB 2026): pass in the old/invalid swapchain if window is resized
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
			VkComponentMapping componentMapping {
				VK_COMPONENT_SWIZZLE_IDENTITY, // r
				VK_COMPONENT_SWIZZLE_IDENTITY, // g
				VK_COMPONENT_SWIZZLE_IDENTITY, // b
				VK_COMPONENT_SWIZZLE_IDENTITY  // a
			};

			VkImageSubresourceRange subresourceRange {
				VK_IMAGE_ASPECT_COLOR_BIT,	// aspectMask // TODO (TF 3 FEB 2026): experiment with Depth and Stencil aspects for views
				0,							// baseMipLevel
				1,							// levelCount // TODO (TF 3 FEB 2026): experiment with variable mip levels
				0,							// baseArraylayer
				1							// layerCount // TODO (TF 3 FEB 2026): experiment with multiple layers for stereoscopic 3D application
			};

			VkImageViewCreateInfo createInfo{
				VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, // sType
				nullptr,								  // pNext
				0,										  // flags
				swapChainImages[i],						  // image
				VK_IMAGE_VIEW_TYPE_2D,					  // viewType
				swapChainImageFormat,					  // format // TODO (TF 3 FEB 2026): experiment with different view formats
				componentMapping,						  // components // TODO (TF 3 FEB 2026): experiment with monochrome components, or const channels
				subresourceRange						  // subresourceRange
			};

			VkResult result = vkCreateImageView(logicalDevice, &createInfo, nullptr, &swapChainImageViews[i]);
			if (result != VK_SUCCESS) {
				throw std::runtime_error("failed to create swapchain image views");
			}
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
			std::cout << "picking GPU [" << physicalDevice << "]" << std::endl;
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

			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = queueFamilyIndex;
			}

			if (presentSupport) {
				indices.presentFamily = queueFamilyIndex;
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
			indices.graphicsFamily.value(),
			indices.presentFamily.value()
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
		vkGetDeviceQueue(logicalDevice, indices.graphicsFamily.value(), defaultQueueIndex, &graphicsQueue);
		vkGetDeviceQueue(logicalDevice, indices.presentFamily.value(), defaultQueueIndex, &presentQueue);
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

	void RecordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
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
			throw std::runtime_error("failed to begin recording command buffer!");
		}

		VkRect2D renderArea = {
			{0, 0},
			swapChainExtent
		};

		VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}}; // union size of 4x 32-bit values

		VkRenderPassBeginInfo renderPassInfo {
			VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,	// sType
			nullptr,									// pNext
			renderPass,									// renderpass
			swapChainFramebuffers[imageIndex],			// framebuffer
			renderArea,									// renderArea
			1,											// clearValueCount
			&clearColor									// pClearValues
		};

		// TODO (TF 5 FEB 2026): experiment with secondary command buffer execution
		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

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

		vkCmdDraw(commandBuffer, 3, 1, 0, 0);
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

	void DrawFrame() {
		// TODO (TF 5 FEB 2026) Experiment with timeline semaphores
		vkWaitForFences(logicalDevice, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

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
		
		vkResetCommandBuffer(commandBuffers[currentFrame], 0); // TODO (TF 5 FEB 2026): experiment with releasing resources on rest

		RecordCommandBuffer(commandBuffers[currentFrame], imageIndex);

		VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};

		// DEBUG: ensure the rendering semaphores are available to be signaled by indexing to swapChain images directly
		// ie: vkQueuePresentKHR doesn't have pSignalSemaphores like vkQueueSubmit does, so this DrawFrame function
		// relies on a Fence to assume the gpu semaphores are ready, which is not guaranted when only indexing by frames-in-flight
		VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[imageIndex]};

		VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

		VkSubmitInfo submitInfo {
			VK_STRUCTURE_TYPE_SUBMIT_INFO,		// sType
			nullptr,							// pNext
			1,									// waitSemaphoreCount
			waitSemaphores,						// pWaitSemaphores
			waitStages,							// pWaitDstStageMask	// TODO (TF 5 FEB 2026): experiment with other pipeline stage flags here
			1,									// commandBufferCount
			&commandBuffers[currentFrame],		// pCommandBuffers
			1,									// signalSemaphoreCount
			signalSemaphores					// pSignalSemaphores
		};

		VkResult submitResult = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]);
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
	}

	void Cleanup() {
		CleanupSwapChain();

		vkDestroyBuffer(logicalDevice, vertexBuffer, nullptr);
		vkFreeMemory(logicalDevice, vertexBufferMemory, nullptr);

		vkDestroyPipeline(logicalDevice, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(logicalDevice, pipelineLayout, nullptr);

		vkDestroyRenderPass(logicalDevice, renderPass, nullptr);
		
		// all command buffers are implicitly cleaned up when the command pool is destroyed
		vkDestroyCommandPool(logicalDevice, commandPool, nullptr);

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
	VkQueue graphicsQueue;
	VkQueue presentQueue;
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
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;

	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;

	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;

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