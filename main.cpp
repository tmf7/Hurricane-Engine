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


const uint32_t WINDOW_WIDTH = 800;
const uint32_t WINDOW_HEIGHT = 600;

const std::vector<const char*> VALIDATION_LAYERS = {
	"VK_LAYER_KHRONOS_validation"
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

private:
	void InitWindow(uint32_t WindowWidth, uint32_t WindowHeight) {
		// TODO (TF 30 JAN 2026): experiment with creating a window using 
		// the the Windows window vulkan extension directly		
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WindowWidth, WindowHeight, "Vulkan Hurricane", nullptr, nullptr);
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

	std::vector<const char*> GetRequiredExtensions() {
		// get which windowing extensions are required
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensionNames;

		glfwExtensionNames = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		// cross-check which extensions this computer supports
		uint32_t supportedExtensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &supportedExtensionCount, nullptr);
		std::vector<VkExtensionProperties> supportedExtensions(supportedExtensionCount);

		vkEnumerateInstanceExtensionProperties(nullptr, &supportedExtensionCount, supportedExtensions.data());

		std::cout << "glfw required extensions (windows):\n";
		for (int i = 0; i < glfwExtensionCount; ++i) {
			std::cout << "\t" << glfwExtensionNames[i] << (IsExtensionSupported(glfwExtensionNames[i], supportedExtensions) ? " SUPPORTED" : " NOT SUPPORTED") << "\n";
		}

		std::cout << "available extensions:\n";
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

		auto requiredExtensions = GetRequiredExtensions();

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

		uint32_t enabledExtensionCount = 0;
		uint32_t enabledLayerCount = 0;
		const char* const* ppEnabledLayerNames = nullptr;

		// Logical device layers are deprecated in favor of instance layers
		// this is only here for legacy support
		if (enableValidationLayers) {
			enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
			ppEnabledLayerNames = VALIDATION_LAYERS.data();
		}

		VkDeviceCreateInfo createInfo {
			VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,			// sType
			nullptr,										// pNext
			0,												// flags
			static_cast<uint32_t>(queueCreateInfos.size()),	// queueCreateInfoCount
			queueCreateInfos.data(),						// pQueueCreateInfos
			enabledLayerCount,								// enabledLayerCount
			ppEnabledLayerNames,							// ppEnabledLayerNames
			enabledExtensionCount,							// enabledExtensionCount
			nullptr,										// ppEnabledExtensionNames // TODO (TF 2 FEB 2025) use an extension (eg "VK_KHR_swapchain")
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

	void MainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void Cleanup() {
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