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


const uint32_t WINDOW_WIDTH = 800;
const uint32_t WINDOW_HEIGHT = 600;

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

		CreateGraphicsPipeline();
	}

	void CreateGraphicsPipeline() {
	
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

		VkSwapchainCreateInfoKHR createInfo{
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

	void MainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void Cleanup() {
		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(logicalDevice, imageView, nullptr);
		}

		// swapchain images are implicitly cleaned up when the swapchain is destroyed
		vkDestroySwapchainKHR(logicalDevice, swapChain, nullptr);

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