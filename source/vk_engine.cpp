//> includes
#include "vk_engine.h"

#include <SDL3\SDL.h>
#include <SDL3\SDL_vulkan.h>

#include "vk_initializers.h"
#include "vk_types.h"

#include <chrono>
#include <thread>

#include <VkBootstrap.h>
#include "vk_images.h"

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "vk_pipelines.h"

#define GLM_ENABLE_EXPERIMENTAL
//#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

// ======= BEGIN IMGUI UI ========
#include <imgui.h>
#include <imgui_backends\imgui_impl_sdl3.h>
#include <imgui_backends\imgui_impl_vulkan.h>
//#include <imstb/imstb_textedit.h>
//#include <imstb/imstb_rectpack.h>
//#include <imstb/imstb_truetype.h>
// ======= end IMGUI UI ========


#ifdef NDEBUG
constexpr bool bUseValidationLayers = false;
#else
constexpr bool bUseValidationLayers = true;
#endif

VulkanEngine* loadedEngine = nullptr;

void VulkanEngine::immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    VK_CHECK(vkResetFences(_device, 1, &_immFence));
    VK_CHECK(vkResetCommandBuffer(_immCommandBuffer, 0));

    VkCommandBuffer cmd = _immCommandBuffer;
    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));
    function(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkCommandBufferSubmitInfo cmdSubmitInfo = vkinit::command_buffer_submit_info(cmd);
    VkSubmitInfo2 submit = vkinit::submit_info(&cmdSubmitInfo, nullptr, nullptr);

    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, _immFence));
    VK_CHECK(vkWaitForFences(_device, 1, &_immFence, VK_TRUE, UINT64_MAX));
}

// ======= BEGIN IMGUI UI ========
void VulkanEngine::init_imgui()
{
    // FIXME (TF 27 FEB 2026): wasteful descriptor pool sizes (from imgui demo)
    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 },
    };

    VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 1000,
        .poolSizeCount = (uint32_t)std::size(pool_sizes),
        .pPoolSizes = pool_sizes
    };

    VkDescriptorPool imguiPool;
    VK_CHECK(vkCreateDescriptorPool(_device, &pool_info, nullptr, &imguiPool));

    ImGui::CreateContext();

    // SDL3 init
    ImGui_ImplSDL3_InitForVulkan(_window);
    
    // vulkan init
    VkPipelineRenderingCreateInfoKHR pipelineRenderingCreateInfo {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
        .pNext = nullptr,
        .viewMask = 0,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &_swapchainImageFormat
        // .depthAttachementFormat = VK_FORMAT_UNDEFINED
        // .stencilAttachmentFormat = VK_FORMAT_UNDEFINED
    };

    ImGui_ImplVulkan_PipelineInfo pipelineInfo {
        // .RenderPass = VK_NULL_HANDLE,
        // .Subpass = 0,
        .MSAASamples = VK_SAMPLE_COUNT_1_BIT,
        .PipelineRenderingCreateInfo = pipelineRenderingCreateInfo
    };

    ImGui_ImplVulkan_InitInfo vulkan_init_info {
        // .ApiVersion = [[Fill with API version of Instance]],
        .Instance = _instance,
        .PhysicalDevice = _chosenGPU,
        .Device = _device,
        // .QueueFamily = VK_QUEUE_FAMILY_IGNORED,
        .Queue = _graphicsQueue,
        .DescriptorPool = imguiPool,
        // .DescriptorPoolSize = 1000,
        .MinImageCount = 3,
        .ImageCount = 3,
        // .PipelineCache = VK_NULL_HANDLE,
        .PipelineInfoMain = pipelineInfo,
        .UseDynamicRendering = true,
        // .Allocator = nullptr,
        // .MinAllocationSize = 0,
        // .CustomShaderVertCreateInfo = NULL,
        // .CustomShaderFragCreateInfo = NULL
    };

    ImGui_ImplVulkan_Init(&vulkan_init_info);

    // DEBUG: ImGui_ImplVulkan_CreateFontsTexture() no longer necessary because ImGui_ImplVulkan_NewFrame() internally handles font lifetime

    _mainDeletionQueue.push_function([=]() {
        ImGui_ImplVulkan_Shutdown();
        vkDestroyDescriptorPool(_device, imguiPool, nullptr);
    });
}

void VulkanEngine::draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView)
{
    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(targetImageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingInfo renderInfo = vkinit::rendering_info(_swapchainExtent, &colorAttachment, nullptr);

    vkCmdBeginRendering(cmd, &renderInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);
}

// ======= END IMGUI UI ========

AllocatedImage VulkanEngine::create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    AllocatedImage newImage{
        .imageExtent = size,
        .imageFormat = format
    };

    // assign image, imageView, and allocation
    VkImageCreateInfo img_info = vkinit::image_create_info(format, usage, size);
    if (mipmapped) {
        img_info.mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(size.width, size.height)))) + 1;
    }

    VmaAllocationCreateInfo allocInfo{
        .usage = VMA_MEMORY_USAGE_GPU_ONLY,
        .requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    };

    VK_CHECK(vmaCreateImage(_allocator, &img_info, &allocInfo, &newImage.image, &newImage.allocation, nullptr));
    
    VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_D32_SFLOAT) {
        aspectFlags = VK_IMAGE_ASPECT_DEPTH_BIT; // depth-only
    }

    VkImageViewCreateInfo view_info = vkinit::imageview_create_info(format, newImage.image, aspectFlags);
    view_info.subresourceRange.levelCount = img_info.mipLevels;

    VK_CHECK(vkCreateImageView(_device, &view_info, nullptr, &newImage.imageView));
    return newImage;
}

// DEBUG: assumes RGBA8 bit format of data
AllocatedImage VulkanEngine::create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped)
{
    size_t data_size = size.depth * size.width * size.height * 4;
    AllocatedBuffer uploadBuffer = create_buffer(data_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    memcpy(uploadBuffer.info.pMappedData, data, data_size);
    AllocatedImage newImage = create_image(size, format, usage | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipmapped);

    immediate_submit([&](VkCommandBuffer cmd) {
        vkutil::transition_image(cmd, newImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        
        VkImageSubresourceLayers subresource{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1
        };

        VkBufferImageCopy copyRegion{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = subresource,
            .imageOffset = 0,
            .imageExtent = size
        };
    
        vkCmdCopyBufferToImage(cmd, uploadBuffer.buffer, newImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
        vkutil::transition_image(cmd, newImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });

    destroy_buffer(uploadBuffer);
    return newImage;
}
void VulkanEngine::destroy_image(const AllocatedImage& image)
{
    vkDestroyImageView(_device, image.imageView, nullptr);
    vmaDestroyImage(_allocator, image.image, image.allocation);
}

void VulkanEngine::update_scene()
{
    mainDrawContext.opaqueSurfaces.clear();

    // ================= BEGIN "ANIMATION" =========================================
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    glm::f32 rotationRate = time * glm::radians(90.0f);
    glm::vec3 upAxis = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::mat4 modelMatrix = glm::rotate(glm::mat4{ 1.0f }, rotationRate, upAxis);
    // =================== END "ANIMATION" =======================================

    for (int x = -3; x <= 3; ++x) 
    {
        glm::mat4 scale = glm::scale(glm::vec3{0.2f});
        glm::mat4 translation = glm::translate(glm::vec3{x, -1.0f, 0.0f});
        loadedNodes["Cube"]->Draw(translation * scale * modelMatrix, mainDrawContext);
    }

    loadedNodes["Suzanne"]->Draw(glm::mat4{ 1.0f } * modelMatrix, mainDrawContext);
    loadedScenes["structure"]->Draw(glm::mat4{ 1.0f }, mainDrawContext);

    mainCamera.Update();
    sceneData.viewMatrix = mainCamera.GetViewMatrix();
    // PERF: flipping the near and far plane values AND flipping the depth test increases the precision for distant objects
    sceneData.projectionMatrix = glm::perspective(glm::radians(70.0f), ((float)_windowExtent.width) / ((float)_windowExtent.height), 10000.0f , 0.1f);
    // DEBUG (TF 18 MAR 2026): was _drawExtent width/height but caused divide-by-zero error since update_scene called too early for _drawExtent to be defined

    // invert y-axis of openGL glTF file to match vulkan's downward y-axis
    sceneData.projectionMatrix[1][1] *= -1.0f;
    sceneData.viewProjectionMatrix = sceneData.projectionMatrix * sceneData.viewMatrix;

    sceneData.ambientColor = glm::vec4{0.1f};
    sceneData.sunlightColor = glm::vec4{ 1.0f };
    sceneData.sunlightDirection = glm::vec4{ 0.0f, 1.0f, 0.5f, 1.0f };

}

VulkanEngine& VulkanEngine::Get() { return *loadedEngine; }

void VulkanEngine::init()
{
    // only one engine initialization is allowed with the application.
    assert(loadedEngine == nullptr);
    loadedEngine = this;
    
    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    _window = SDL_CreateWindow(
        "Vulkan Engine",
        //SDL_WINDOWPOS_UNDEFINED,
        //SDL_WINDOWPOS_UNDEFINED,
        _windowExtent.width,
        _windowExtent.height,
        window_flags);

    init_vulkan();
    init_swapchain();
    init_commands();
    init_sync_structures();
    init_descriptors();
    init_pipelines();
    init_imgui();
    init_default_data();

    // ========= BEGIN TEST DATA ========
    mainCamera.velocity = glm::vec3{0.0f};
    mainCamera.position = glm::vec3{30.0f, 0.0f, -85.0f}; // glm::vec3(0.0f, 0.0f, 5.0f);
    mainCamera.pitch = 0.0f;
    mainCamera.yaw = 0.0f;

    std::string structurePath = { "Scenes/structure.glb" };
    auto structureFile = LoadGLTF(this, structurePath);
    assert(structureFile.has_value());
    loadedScenes["structure"] = *structureFile;
    // ========= END TEST DATA ========

    // everything went fine
    _isInitialized = true;
}

void VulkanEngine::cleanup()
{
    if (_isInitialized) {

        vkDeviceWaitIdle(_device);

        loadedScenes.clear(); // TODO (TF 19 MAR 2026): actually clean up the loaded scenes data

        for (int i = 0; i < FRAME_OVERLAP; ++i) {
            vkDestroyCommandPool(_device, _frames[i]._commandPool, nullptr);
            vkDestroyFence(_device, _frames[i]._renderFence, nullptr);
            vkDestroySemaphore(_device, _frames[i]._swapchainSemaphore, nullptr);

            _frames[i]._deletionQueue.flush();
        }

        for (int i = 0; i < _renderSemaphores.size(); ++i) {
            vkDestroySemaphore(_device, _renderSemaphores[i], nullptr);
        }

        for (auto& mesh : _testMeshes) {
            destroy_buffer(mesh->meshBuffers.indexBuffer);
            destroy_buffer(mesh->meshBuffers.vertexBuffer);
        }

        _mainDeletionQueue.flush();

        destroy_swapchain();
        vkDestroySurfaceKHR(_instance, _surface, nullptr);
        vkDestroyDevice(_device, nullptr);

        vkb::destroy_debug_utils_messenger(_instance, _debug_messenger, nullptr);
        vkDestroyInstance(_instance, nullptr);

        SDL_DestroyWindow(_window);
    }

    // clear engine pointer
    loadedEngine = nullptr;
}

void VulkanEngine::draw()
{
    update_scene();

    // TODO (TF 25 FEB 2026): set timeout to 1 second (1000000000 ns)
    VK_CHECK(vkWaitForFences(_device, 1, &get_current_frame()._renderFence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(_device, 1, &get_current_frame()._renderFence));

    get_current_frame()._deletionQueue.flush();
    get_current_frame()._frameDescriptors.clear_pools(_device);

    uint32_t swapchainImageIndex;
    VkResult nextImageResult = vkAcquireNextImageKHR(_device, _swapchain, UINT64_MAX, get_current_frame()._swapchainSemaphore, nullptr, &swapchainImageIndex);
    if (nextImageResult == VK_ERROR_OUT_OF_DATE_KHR) {
        _resizeRequested = true;
        return;
    }

    VkCommandBuffer cmd = get_current_frame()._mainCommandBuffer;
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
    
    _drawExtent.height = static_cast<uint32_t>(std::min(_swapchainExtent.height, _drawImage.imageExtent.height) * _renderScale);
    _drawExtent.width = static_cast<uint32_t>(std::min(_swapchainExtent.width, _drawImage.imageExtent.width) * _renderScale);

    VkCommandBufferBeginInfo cmdBeginInfo = vkinit::command_buffer_begin_info(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VK_CHECK(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    draw_background(cmd);

    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkutil::transition_image(cmd, _depthImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    draw_geometry(cmd);

    vkutil::transition_image(cmd, _drawImage.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    vkutil::copy_image_to_image(cmd, _drawImage.image, _swapchainImages[swapchainImageIndex], _drawExtent, _swapchainExtent);

    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    draw_imgui(cmd, _swapchainImageViews[swapchainImageIndex]);

    vkutil::transition_image(cmd, _swapchainImages[swapchainImageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(cmd));

    // wait on the _presentSemaphore, as it is signals when the swapchain is ready
    // signal the _renderSemaphore, to signals that rendering has finished
    VkCommandBufferSubmitInfo cmdInfo = vkinit::command_buffer_submit_info(cmd);
    VkSemaphoreSubmitInfo waitInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR, get_current_frame()._swapchainSemaphore);
    VkSemaphoreSubmitInfo signalInfo = vkinit::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, _renderSemaphores[swapchainImageIndex]);

    VkSubmitInfo2 submit = vkinit::submit_info(&cmdInfo, &signalInfo, &waitInfo);

    // block with _renderFence until graphic commands finish
    VK_CHECK(vkQueueSubmit2(_graphicsQueue, 1, &submit, get_current_frame()._renderFence));

    VkPresentInfoKHR presentInfo = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &_renderSemaphores[swapchainImageIndex],
        .swapchainCount = 1,
        .pSwapchains = &_swapchain,
        .pImageIndices = &swapchainImageIndex,
        .pResults = nullptr
    };

    VkResult presentResult = vkQueuePresentKHR(_graphicsQueue, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR 
        /*|| presentResult == VK_SUBOPTIMAL_KHR*/) {
        _resizeRequested = true;
    }
    _frameNumber++;
}

void VulkanEngine::run()
{
    SDL_Event event;
    bool bQuit = false;

    // main loop
    while (!bQuit) {
        // Handle events on queue
        while (SDL_PollEvent(&event) != 0) {
            // close the window when user alt-f4s or clicks the X button
            if (event.type == SDL_EVENT_QUIT) {
                bQuit = true;
            }

            if (event.window.type == SDL_EVENT_WINDOW_MINIMIZED) {
                stop_rendering = true;
            }
            if (event.window.type == SDL_EVENT_WINDOW_RESTORED) {
                stop_rendering = false;
            }

            mainCamera.ProcessSDLEvent(event);

            // ======= BEGIN IMGUI UI ========
            ImGui_ImplSDL3_ProcessEvent(&event);
            // ======= END IMGUI UI ========
        }

        // do not draw if we are minimized
        if (stop_rendering) {
            // throttle the speed to avoid the endless spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (_resizeRequested) {
            resize_swapchain();
        }

        // ======= BEGIN IMGUI UI ========
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        // test-only
        //ImGui::ShowDemoWindow();

        if (ImGui::Begin("background")) {
            ImGui::SliderFloat("Render Scale", &_renderScale, 0.3f, 1.0f);
            ComputeEffect& selected = _backgroundEffects[_currentBackgroundEffect];

            ImGui::Text("Selected effect: ", selected.name);
            ImGui::SliderInt("Effect Index", &_currentBackgroundEffect, 0, _backgroundEffects.size() - 1);
        
            ImGui::InputFloat4("data1", (float*)&selected.data.data1);
            ImGui::InputFloat4("data2", (float*)&selected.data.data2);
            ImGui::InputFloat4("data3", (float*)&selected.data.data3);
            ImGui::InputFloat4("data4", (float*)&selected.data.data4);
        }
        ImGui::End();

        ImGui::Render();
        // ======= END IMGUI UI ========

        draw();
    }
}

void VulkanEngine::init_vulkan()
{
    // abstract instance and validation setup using vk-bootstrap
    vkb::InstanceBuilder builder;

    vkb::Result<vkb::Instance> inst_ret = 
        builder.set_app_name("Hurricane")
               .request_validation_layers(bUseValidationLayers)
               .use_default_debug_messenger()
               .require_api_version(1, 3, 0)
               .build();

    vkb::Instance vkb_inst = inst_ret.value();
    _instance = vkb_inst.instance;
    _debug_messenger = vkb_inst.debug_messenger;

    SDL_Vulkan_CreateSurface(_window, _instance, nullptr, &_surface);

    // vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.dynamicRendering = true;
    features13.synchronization2 = true;

    // vulkan 1.2 features
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.bufferDeviceAddress = true;
    features12.descriptorIndexing = true;

    // abstract physical device selection and logical device creation using vk-bootstrap
    vkb::PhysicalDeviceSelector selector{ vkb_inst };
    vkb::PhysicalDevice physicalDevice = selector
        .set_minimum_version(1, 3)
        .set_required_features_13(features13)
        .set_required_features_12(features12)
        .set_surface(_surface)
        .select()
        .value();

    vkb::DeviceBuilder deviceBuilder{ physicalDevice };
    vkb::Device vkbDevice = deviceBuilder.build().value();
    _device = vkbDevice.device;
    _chosenGPU = physicalDevice.physical_device;

    // abstract graphics queue selection using vk-bootstrap
    _graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
    _graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

    // NOTE: optional elements of VmaAllocatorCreateInfo (default to 0 and nullptr)
    // VkDeviceSize                                 preferredLargeHeapBlockSize;
    // const VkAllocationCallbacks*                 pAllocationCallbacks;
    // const VmaDeviceMemoryCallbacks*              pDeviceMemoryCallbacks;
    // const VkDeviceSize*                          pHeapSizeLimit;
    // const VmaVulkanFunctions*                    pVulkanFunctions;
    // uint32_t                                     vulkanApiVersion;
    // const VkExternalMemoryHandleTypeFlagsKHR*    pTypeExternalMemoryHandleTypes;

    // abstract memory allocation, management, and resource binding with VulkanMemoryAllocator
    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.physicalDevice = _chosenGPU;
    allocatorInfo.device = _device;
    allocatorInfo.instance = _instance;
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    vmaCreateAllocator(&allocatorInfo, &_allocator);

    _mainDeletionQueue.push_function([&]() {
        vmaDestroyAllocator(_allocator);
    });
}

void VulkanEngine::init_swapchain()
{
    create_swapchain(_windowExtent.width, _windowExtent.height);

    VkExtent3D drawImageExtent = {
        _windowExtent.width,
        _windowExtent.height,
        1
    };

    // TODO (TF 26 FEB 2028): make these values dynamic, not hardcoded
    _drawImage.imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    _drawImage.imageExtent = drawImageExtent;

    _depthImage.imageFormat = VK_FORMAT_D32_SFLOAT;
    _depthImage.imageExtent = drawImageExtent;

    VkImageUsageFlags drawImageUsages{};
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_STORAGE_BIT;
    drawImageUsages |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    VkImageUsageFlags depthImageUsages{};
    depthImageUsages |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VkImageCreateInfo rimg_info = vkinit::image_create_info(_drawImage.imageFormat, drawImageUsages, drawImageExtent);
    VkImageCreateInfo dimg_info = vkinit::image_create_info(_depthImage.imageFormat, depthImageUsages, drawImageExtent);

    // abstract image allocation and binding with VulkanMemoryAllocator
    VmaAllocationCreateInfo img_allocInfo = {};
    img_allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    img_allocInfo.requiredFlags = VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    
    vmaCreateImage(_allocator, &rimg_info, &img_allocInfo, &_drawImage.image, &_drawImage.allocation, nullptr);
    VkImageViewCreateInfo rview_info = vkinit::imageview_create_info(_drawImage.imageFormat, _drawImage.image, VK_IMAGE_ASPECT_COLOR_BIT);

    VK_CHECK(vkCreateImageView(_device, &rview_info, nullptr, &_drawImage.imageView));

    vmaCreateImage(_allocator, &dimg_info, &img_allocInfo, &_depthImage.image, &_depthImage.allocation, nullptr);
    VkImageViewCreateInfo dview_info = vkinit::imageview_create_info(_depthImage.imageFormat, _depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);

    VK_CHECK(vkCreateImageView(_device, &dview_info, nullptr, &_depthImage.imageView));

    _mainDeletionQueue.push_function([=]() {
        vkDestroyImageView(_device, _drawImage.imageView, nullptr);
        vkDestroyImageView(_device, _depthImage.imageView, nullptr);
        vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);
        vmaDestroyImage(_allocator, _depthImage.image, _depthImage.allocation);
    });
}

void VulkanEngine::init_commands()
{
    VkCommandPoolCreateInfo commandPoolInfo = 
        vkinit::command_pool_create_info(_graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
   
    // ======= BEGIN IMGUI UI ========
    VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_immCommandPool));

    VkCommandBufferAllocateInfo immCmdAllocInfo = vkinit::command_buffer_allocate_info(_immCommandPool, 1);
    VK_CHECK(vkAllocateCommandBuffers(_device, &immCmdAllocInfo, &_immCommandBuffer));

    _mainDeletionQueue.push_function([=]() {
        vkDestroyCommandPool(_device, _immCommandPool, nullptr);
    });
    // ======= END IMGUI UI ========

    for (int i = 0; i < FRAME_OVERLAP; ++i) {
        VK_CHECK(vkCreateCommandPool(_device, &commandPoolInfo, nullptr, &_frames[i]._commandPool));
        
        VkCommandBufferAllocateInfo cmdAllocInfo = 
            vkinit::command_buffer_allocate_info(_frames[i]._commandPool, 1);

        VK_CHECK(vkAllocateCommandBuffers(_device, &cmdAllocInfo, &_frames[i]._mainCommandBuffer));
    }
}

void VulkanEngine::init_sync_structures()
{
    VkFenceCreateInfo fenceCreateInfo = vkinit::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);
    VkSemaphoreCreateInfo semaphoreCreateInfo = vkinit::semaphore_create_info();

    // ======= BEGIN IMGUI UI ========
    VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_immFence));
    _mainDeletionQueue.push_function([=]() {
        vkDestroyFence(_device, _immFence, nullptr);
    });
    // ======= END IMGUI UI ========

    for (int i = 0; i < FRAME_OVERLAP; ++i) {
        VK_CHECK(vkCreateFence(_device, &fenceCreateInfo, nullptr, &_frames[i]._renderFence));
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_frames[i]._swapchainSemaphore));
    }

    _renderSemaphores.resize(_swapchainImages.size());

    for (int i = 0; i < _renderSemaphores.size(); ++i) {
        VK_CHECK(vkCreateSemaphore(_device, &semaphoreCreateInfo, nullptr, &_renderSemaphores[i]));
    }
}

void VulkanEngine::create_swapchain(uint32_t width, uint32_t height)
{
    // abstract creation of the swapchain, its images and their imageViews using vk-bootstrap
    vkb::SwapchainBuilder swapchainBuilder{ _chosenGPU, _device, _surface };
    _swapchainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;

    vkb::Swapchain vkbSwapchain = swapchainBuilder
        .set_desired_format(VkSurfaceFormatKHR{ 
            .format = _swapchainImageFormat, 
            .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR })
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .set_desired_extent(width, height)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build()
        .value();

    _swapchainExtent = vkbSwapchain.extent;
    _swapchain = vkbSwapchain.swapchain;
    _swapchainImages = vkbSwapchain.get_images().value();
    _swapchainImageViews = vkbSwapchain.get_image_views().value();
}

void VulkanEngine::destroy_swapchain()
{
    vkDestroySwapchainKHR(_device, _swapchain, nullptr);

    for (int i = 0; i < _swapchainImageViews.size(); ++i) {
        vkDestroyImageView(_device, _swapchainImageViews[i], nullptr);
    }
}

void VulkanEngine::draw_background(VkCommandBuffer cmd)
{
    //VkClearColorValue clearValue;
    //float flash = std::abs(std::sin(_frameNumber / 120.0f));
    //clearValue = { {0.0f, 0.0f, flash, 1.0f} };

    //// TODO (REMOVE TF 26 FEB 2026): was _swapchainImages[swapchainImageIndex]
    //// force swapchain image clear each time its presented
    //VkImageSubresourceRange clearRange = vkinit::image_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT);
    //vkCmdClearColorImage(cmd, _drawImage.image, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);

    ////////////////////////
    ComputeEffect& effect = _backgroundEffects[_currentBackgroundEffect];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, effect.layout, 0, 1, &_drawImageDescriptors, 0, nullptr);


    vkCmdPushConstants(cmd, _backgroundPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePushConstants), &effect.data);

    vkCmdDispatch(cmd, static_cast<uint32_t>(std::ceil(_drawExtent.width / 16.0)), 
                       static_cast<uint32_t>(std::ceil(_drawExtent.height / 16.0)),
                       1);
}

void VulkanEngine::init_descriptors()
{
    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes =
    {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 }
    };

    // TODO (TF 26 FEB 2026): make hardcoded maxSets value dynamic
    globalDescriptorAllocator.init(_device, 10, sizes);

    // compute descriptor set layout at binding 0
    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        _drawImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_COMPUTE_BIT);
    }

    _drawImageDescriptors = globalDescriptorAllocator.allocate(_device, _drawImageDescriptorLayout);

    // ===== BEGIN DEPRECATED ========
    //VkDescriptorImageInfo imgInfo{
    //    .sampler = VK_NULL_HANDLE,
    //    .imageView = _drawImage.imageView,
    //    .imageLayout = VK_IMAGE_LAYOUT_GENERAL
    //};

    //VkWriteDescriptorSet drawImageWrite{
    //    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    //    .pNext = nullptr,
    //    .dstSet = _drawImageDescriptors,
    //    .dstBinding = 0,
    //    .dstArrayElement = 0,
    //    .descriptorCount = 1,
    //    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    //    .pImageInfo = &imgInfo
    //    //.pBufferInfo = nullptr,
    //    //.pTexelBufferView = nullptr
    //};

    //vkUpdateDescriptorSets(_device, 1, &drawImageWrite, 0, nullptr);
    // ===== END DEPRECATED ========

    {
        DescriptorWriter writer;
        writer.write_image(0, _drawImage.imageView, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        writer.update_set(_device, _drawImageDescriptors);
    }

    for (int i = 0; i < FRAME_OVERLAP; ++i) {
        std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> frame_sizes = {
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 }
        };

        _frames[i]._frameDescriptors = DescriptorAllocatorGrowable{};
        _frames[i]._frameDescriptors.init(_device, 1000, frame_sizes);

        _mainDeletionQueue.push_function([&, i]() {
            _frames[i]._frameDescriptors.destroy_pools(_device);
        });
    }

    {
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
        _gpuSceneDataDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    {
        // FIXME (TF 18 MAR 2026): GPU vendors say its more performant to use separate sampler & image
        DescriptorLayoutBuilder builder;
        builder.add_binding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        _singleImageDescriptorLayout = builder.build(_device, VK_SHADER_STAGE_FRAGMENT_BIT);
    }

    _mainDeletionQueue.push_function([&]() {
        globalDescriptorAllocator.destroy_pools(_device);
        vkDestroyDescriptorSetLayout(_device, _drawImageDescriptorLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _gpuSceneDataDescriptorLayout, nullptr);
        vkDestroyDescriptorSetLayout(_device, _singleImageDescriptorLayout, nullptr);
    });
}

void VulkanEngine::init_pipelines()
{
    init_background_pipelines();
    init_mesh_pipeline();
    _metalRoughMaterial.build_pipelines(this);
}

void VulkanEngine::init_background_pipelines()
{
    VkPushConstantRange pushConstant{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(ComputePushConstants)
    };

    VkPipelineLayoutCreateInfo computeLayout{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &_drawImageDescriptorLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstant
    };

    VK_CHECK(vkCreatePipelineLayout(_device, &computeLayout, nullptr, &_backgroundPipelineLayout));

    VkShaderModule gradientComputeShader;
    if (!vkutil::load_shader_module("Shaders/gradient_color.spv", _device, &gradientComputeShader)) {
        std::cout << "Error when building the gradient_color compute shader \n";
    }

    VkShaderModule skyComputeShader;
    if (!vkutil::load_shader_module("Shaders/sky.spv", _device, &skyComputeShader)) {
        std::cout << "Error when building the sky compute shader \n";
    }

    VkPipelineShaderStageCreateInfo stageInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = gradientComputeShader,
        .pName = "main"
        // pSpecializationInfo = nullptr
    };

    VkComputePipelineCreateInfo computePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = stageInfo,
        .layout = _backgroundPipelineLayout
         // basePipelineHandle = VK_NULL_HANDLE
         // basePipelineIndex = 0
    };

    ComputeEffect gradient;
    gradient.layout = _backgroundPipelineLayout;
    gradient.name = "gradient";
    gradient.data = {};

    // default colors
    gradient.data.data1 = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    gradient.data.data2 = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &gradient.pipeline));

    computePipelineCreateInfo.stage.module = skyComputeShader;

    ComputeEffect sky;
    sky.layout = _backgroundPipelineLayout;
    sky.name = "sky";
    sky.data = {};

    // default sky
    sky.data.data1 = glm::vec4(0.1f, 0.2f, 0.4f, 0.97f);

    VK_CHECK(vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, nullptr, &sky.pipeline));

    _backgroundEffects.push_back(gradient);
    _backgroundEffects.push_back(sky);

    vkDestroyShaderModule(_device, skyComputeShader, nullptr);
    vkDestroyShaderModule(_device, gradientComputeShader, nullptr);

    _mainDeletionQueue.push_function([=]() {
        vkDestroyPipelineLayout(_device, _backgroundPipelineLayout, nullptr);
        vkDestroyPipeline(_device, sky.pipeline, nullptr);
        vkDestroyPipeline(_device, gradient.pipeline, nullptr);
    });
}

void VulkanEngine::init_mesh_pipeline()
{
    VkShaderModule triangleFragShader;
    if (!vkutil::load_shader_module("Shaders/colored_triangle_frag.spv", _device, &triangleFragShader)) {
        std::cout << "Error when building the triangle fragment shader module!" << std::endl;
    }

    VkShaderModule triangleVertexShader;
    if (!vkutil::load_shader_module("Shaders/colored_triangle_vert.spv", _device, &triangleVertexShader)) {
        std::cout << "Error when building the triangle vertex shader module!" << std::endl;
    }

    VkPushConstantRange bufferRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(GPUDrawPushConstants)
    };

    VkPipelineLayoutCreateInfo pipeline_layout_info = vkinit::pipeline_layout_create_info();
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &bufferRange;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &_singleImageDescriptorLayout;

    VK_CHECK(vkCreatePipelineLayout(_device, &pipeline_layout_info, nullptr, &_meshPipelineLayout));

    PipelineBuilder pipelineBuilder;

    pipelineBuilder._pipelineLayout = _meshPipelineLayout;
    pipelineBuilder.set_shaders(triangleVertexShader, triangleFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    //pipelineBuilder.disable_blending();
    //pipelineBuilder.disable_depthtest();
    //pipelineBuilder.enable_blending_additive();
    pipelineBuilder.enable_blending_alphablend();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
    pipelineBuilder.set_color_attachment_format(_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(_depthImage.imageFormat);

    _meshPipeline = pipelineBuilder.build_pipeline(_device);

    vkDestroyShaderModule(_device, triangleVertexShader, nullptr);
    vkDestroyShaderModule(_device, triangleFragShader, nullptr);

    _mainDeletionQueue.push_function([&]() {
        vkDestroyPipelineLayout(_device, _meshPipelineLayout, nullptr);
        vkDestroyPipeline(_device, _meshPipeline, nullptr);
    });
}

void VulkanEngine::draw_geometry(VkCommandBuffer cmd)
{
    // ===== BEGIN DYNAMIC BUFFER ALLOCATION ===============================
    // TODO (TF 16 MAR 2026): It would be better to cache these buffers in the FrameData structure
    // ...currently only allocated each frame to demo temp/dynamic per-frame data.
    AllocatedBuffer gpuSceneDataBuffer = create_buffer(sizeof(GPUSceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    get_current_frame()._deletionQueue.push_function([=, this]() {
        destroy_buffer(gpuSceneDataBuffer);
    });

    GPUSceneData* sceneUniformData = static_cast<GPUSceneData*>(gpuSceneDataBuffer.allocation->GetMappedData());
    *sceneUniformData = sceneData;

    VkDescriptorSet globalDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _gpuSceneDataDescriptorLayout);
    DescriptorWriter writer;
    writer.write_buffer(0, gpuSceneDataBuffer.buffer, sizeof(GPUSceneData), 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.update_set(_device, globalDescriptor);
    // ===== END DYNAMIC BUFFER ALLOCATION ===============================

    VkRenderingAttachmentInfo colorAttachment = vkinit::attachment_info(_drawImage.imageView, nullptr, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment = vkinit::depth_attachment_info(_depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = vkinit::rendering_info(_drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);

    // =========== BEGIN DEPRECATED ===========
    /* 
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipeline);

    VkDescriptorSet imageSet = get_current_frame()._frameDescriptors.allocate(_device, _singleImageDescriptorLayout);
    {
        DescriptorWriter writer;
        writer.write_image(0, _errorCheckerboardImage.imageView, _defaultSamplerNearest, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
        writer.update_set(_device, imageSet);
    }
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, _meshPipelineLayout, 0, 1, &imageSet, 0, nullptr);

    GPUDrawPushConstants push_constants;

    // ================= BEGIN "ANIMATION" =========================================
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
    
    glm::f32 rotationRate = time * glm::radians(90.0f);
    glm::vec3 upAxis = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::mat4 modelMatrix = glm::rotate(glm::mat4{ 1.0f }, rotationRate, upAxis);
    // =================== END "ANIMATION" =======================================
    glm::mat4 viewMatrix = 
        //glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), upAxis);
        glm::translate(glm::vec3{0.0f, 0.0f, -5.0f});

    // PERF: flipping the near and far plane values AND flipping the depth test increases the precision for distant objects
    glm::mat4 projectionMatrix = glm::perspective(glm::radians(70.0f), ((float)_drawExtent.width) / ((float)_drawExtent.height), 10000.0f, 0.1f);

    // invert y-axis of openGL glTF file to match vulkan's downward y-axis
    projectionMatrix[1][1] *= -1.0f;

    push_constants.worldMatrix = projectionMatrix * viewMatrix * modelMatrix;
    push_constants.vertexBuffer = _testMeshes[2]->meshBuffers.vertexBufferAddress;

    vkCmdPushConstants(cmd, _meshPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &push_constants);
    vkCmdBindIndexBuffer(cmd, _testMeshes[2]->meshBuffers.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    VkViewport viewport{
        .x = 0,
        .y = 0,
        .width = static_cast<float>(_drawExtent.width),
        .height = static_cast<float>(_drawExtent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };

    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{
        .offset = {},
        .extent = _drawExtent
    };

    vkCmdSetScissor(cmd, 0, 1, &scissor);

    
    //vkCmdDrawIndexed(cmd, _testMeshes[2]->surfaces[0].count, 1, _testMeshes[2]->surfaces[0].startIndex, 0, 0);
    */
    // =========== END DEPRECATED ===========

    VkViewport viewport{
    .x = 0,
    .y = 0,
    .width = static_cast<float>(_drawExtent.width),
    .height = static_cast<float>(_drawExtent.height),
    .minDepth = 0.0f,
    .maxDepth = 1.0f
    };

    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{
        .offset = {},
        .extent = _drawExtent
    };

    vkCmdSetScissor(cmd, 0, 1, &scissor);

    for (const RenderObject& drawObject : mainDrawContext.opaqueSurfaces) 
    {
        // FIXME (TF 18 MAR 2026): (PERF) do not bind for the same objects every frame, keep it cached
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawObject.material->pipeline->pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawObject.material->pipeline->layout, 0, 1, &globalDescriptor, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, drawObject.material->pipeline->layout, 1, 1, &drawObject.material->materialSet, 0, nullptr);
    
        vkCmdBindIndexBuffer(cmd, drawObject.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        GPUDrawPushConstants pushConstants{
            .worldMatrix = drawObject.transform,
            .vertexBuffer = drawObject.vertexBufferAddress
        };
        vkCmdPushConstants(cmd, drawObject.material->pipeline->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &pushConstants);

        vkCmdDrawIndexed(cmd, drawObject.indexCount, 1, drawObject.firstIndex, 0, 0);
    }

    vkCmdEndRendering(cmd);
}

AllocatedBuffer VulkanEngine::create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage)
{
    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = allocSize,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr
    };


    VmaAllocationCreateInfo vmaallocInfo{
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = memoryUsage,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .memoryTypeBits = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0
    };

    AllocatedBuffer newBuffer;
    VK_CHECK(vmaCreateBuffer(_allocator, &bufferInfo, &vmaallocInfo, &newBuffer.buffer, &newBuffer.allocation, &newBuffer.info));
    
    return newBuffer;
}

void VulkanEngine::destroy_buffer(const AllocatedBuffer& buffer)
{
    vmaDestroyBuffer(_allocator, buffer.buffer, buffer.allocation);
}

void VulkanEngine::resize_swapchain()
{
    vkDeviceWaitIdle(_device);

    destroy_swapchain();

    int width;
    int height;
    SDL_GetWindowSize(_window, &width, &height);
    _windowExtent.width = width;
    _windowExtent.height = height;
    create_swapchain(_windowExtent.width, _windowExtent.height);
    _resizeRequested = false;
}

// TODO (TF 4 MAR 2026): perform this work on a separate CPU thread so it isn't blocking
GPUMeshBuffers VulkanEngine::uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices)
{
    const size_t vertexBufferSize = vertices.size() * sizeof(Vertex);
    const size_t indexBufferSize = indices.size() * sizeof(uint32_t);

    GPUMeshBuffers newSurface;
    newSurface.vertexBuffer = create_buffer(vertexBufferSize, 
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, 
                                            VMA_MEMORY_USAGE_GPU_ONLY);

    VkBufferDeviceAddressInfo deviceAddressInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .pNext = nullptr,
        .buffer = newSurface.vertexBuffer.buffer
    };
    newSurface.vertexBufferAddress = vkGetBufferDeviceAddress(_device, &deviceAddressInfo);

    newSurface.indexBuffer = create_buffer(indexBufferSize, 
                                           VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           VMA_MEMORY_USAGE_GPU_ONLY);

    // begin upload to GPU
    AllocatedBuffer staging = create_buffer(vertexBufferSize + indexBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    void* data = staging.allocation->GetMappedData();
    memcpy(data, vertices.data(), vertexBufferSize);
    memcpy((char*)data + vertexBufferSize, indices.data(), indexBufferSize);

    immediate_submit([&](VkCommandBuffer cmd) {
        VkBufferCopy vertexCopy{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = vertexBufferSize
        };

        VkBufferCopy indexCopy{
            .srcOffset = vertexBufferSize,
            .dstOffset = 0,
            .size = indexBufferSize
        };

        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.vertexBuffer.buffer, 1, &vertexCopy);
        vkCmdCopyBuffer(cmd, staging.buffer, newSurface.indexBuffer.buffer, 1, &indexCopy);
    });

    destroy_buffer(staging);

    return newSurface;
}

void VulkanEngine::init_default_data()
{
    _testMeshes = loadGltfMeshes(this, "Models/basicmesh.glb").value();

    uint32_t white = glm::packUnorm4x8(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f));
    _whiteImage = create_image((void*)&white, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t grey = glm::packUnorm4x8(glm::vec4(0.66f, 0.66f, 0.66f, 1.0f));
    _greyImage = create_image((void*)&grey, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t black = glm::packUnorm4x8(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    _blackImage = create_image((void*)&black, VkExtent3D{ 1, 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t magenta = glm::packUnorm4x8(glm::vec4(1.0f, 0.0f, 1.0f, 1.0f));
    std::array<uint32_t, 16 * 16> pixels; // 16 x 16 checkerboard
    for (int x = 0; x < 16; ++x) {
        for (int y = 0; y < 16; ++y) {
            pixels[y * 16 + x] = ((x % 2) ^ (y % 2)) ? magenta : black;
        }
    }
    _errorCheckerboardImage = create_image(pixels.data(), VkExtent3D{ 16, 16, 1 }, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT);

    VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST
        // .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        // .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        // .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        // .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        // .mipLodBias = 0.0f
        // .anisotropyEnable = VK_FALSE,
        // .maxAnisotropy = 0.0f,
        // .compareEnable = VK_FALSE
        // .compareOp = VK_COMPARE_OP_NEVER,
        // .minLod = 0.0f,
        // .maxLod = 0.0f,
        // .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        // .unnormalizedCoordinates = VK_FALSE
    };

    vkCreateSampler(_device, &samplerInfo, nullptr, &_defaultSamplerNearest);

    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    vkCreateSampler(_device, &samplerInfo, nullptr, &_defaultSamplerLinear);

    _mainDeletionQueue.push_function([&]() {
        vkDestroySampler(_device, _defaultSamplerNearest, nullptr);
        vkDestroySampler(_device, _defaultSamplerLinear, nullptr);

        destroy_image(_whiteImage);
        destroy_image(_greyImage);
        destroy_image(_blackImage);
        destroy_image(_errorCheckerboardImage);
    });

    // ============================
    GLTFMetallic_Roughness::MaterialResources materialResources{};
    materialResources.colorImage = _whiteImage;
    materialResources.colorSampler = _defaultSamplerLinear;
    materialResources.metalRoughImage = _whiteImage;
    materialResources.metalRoughSampler = _defaultSamplerLinear;

    AllocatedBuffer materialConstants = create_buffer(sizeof(GLTFMetallic_Roughness::MaterialConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    GLTFMetallic_Roughness::MaterialConstants* sceneUniformData = static_cast<GLTFMetallic_Roughness::MaterialConstants*>(materialConstants.allocation->GetMappedData());
    sceneUniformData->colorFactors = glm::vec4{ 1.0f, 1.0f, 1.0f, 1.0f };
    sceneUniformData->metal_rough_factors = glm::vec4{ 1.0f, 0.5f, 0.0f, 0.0f };

    _mainDeletionQueue.push_function([=, this]() {
        destroy_buffer(materialConstants);
    });

    materialResources.dataBuffer = materialConstants.buffer;
    materialResources.dataBufferOffset = 0;

    // FIXME (TF 18 MAR 2026): had to add VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, and VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER to globalDescriptorAllocator for it to work with _materialLayout's bindings
    _defaultMaterialData = _metalRoughMaterial.write_material(_device, MaterialPass::MainColor, materialResources, globalDescriptorAllocator);

    for (auto& meshAsset : _testMeshes)
    {
        std::shared_ptr<MeshNode> newNode = std::make_shared<MeshNode>();
        newNode->mesh = meshAsset;

        newNode->localTransform = glm::mat4{ 1.0f };
        newNode->worldTransform = glm::mat4{ 1.0f };

        for (auto& surface : newNode->mesh->surfaces) 
        {
            surface.material = std::make_shared<GLTFMaterial>(_defaultMaterialData);
        }

        loadedNodes[meshAsset->name] = std::move(newNode);
    }
}

void GLTFMetallic_Roughness::build_pipelines(VulkanEngine* engine)
{
    VkShaderModule meshFragmentShader;
    if (!vkutil::load_shader_module("Shaders/meshFragment.spv", engine->_device, &meshFragmentShader)) {
        std::cout << "Error when building the [meshFragment.spv] shader module!" << std::endl;
    }

    VkShaderModule meshVertexShader;
    if (!vkutil::load_shader_module("Shaders/meshVertex.spv", engine->_device, &meshVertexShader)) {
        std::cout << "Error when building the [meshVertex.spv] shader module!" << std::endl;
    }

    VkPushConstantRange matrixRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(GPUDrawPushConstants)
    };

    DescriptorLayoutBuilder layoutBuilder{};
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    _materialLayout = layoutBuilder.build(engine->_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    VkDescriptorSetLayout layouts[] = {
        engine->_gpuSceneDataDescriptorLayout,
        _materialLayout
    };

    VkPipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
    mesh_layout_info.setLayoutCount = 2;
    mesh_layout_info.pSetLayouts = layouts;
    mesh_layout_info.pushConstantRangeCount = 1;
    mesh_layout_info.pPushConstantRanges = &matrixRange;

    VkPipelineLayout newLayout;
    VK_CHECK(vkCreatePipelineLayout(engine->_device, &mesh_layout_info, nullptr, &newLayout));

    opaquePipeline.layout = newLayout;
    transparentPipeline.layout = newLayout;

    PipelineBuilder pipelineBuilder;
    pipelineBuilder.set_shaders(meshVertexShader, meshFragmentShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.disable_blending();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);
    pipelineBuilder.set_color_attachment_format(engine->_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(engine->_depthImage.imageFormat);
    pipelineBuilder._pipelineLayout = newLayout;

    opaquePipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    pipelineBuilder.enable_blending_additive();
    pipelineBuilder.enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL);

    transparentPipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    vkDestroyShaderModule(engine->_device, meshVertexShader, nullptr);
    vkDestroyShaderModule(engine->_device, meshFragmentShader, nullptr);
}

// FIXME (TF 18 MAR 2026): the material Pipeline, PipelineLayout, and DescriptorSetLayout are not destroryed during shutdown
MaterialInstance GLTFMetallic_Roughness::write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator)
{
    MaterialInstance materialData{};
    materialData.passType = pass;
    if (pass == MaterialPass::Transparent) {
        materialData.pipeline = &transparentPipeline;
    }
    else {
        materialData.pipeline = &opaquePipeline;
    }

    materialData.materialSet = descriptorAllocator.allocate(device, _materialLayout);

    writer.clear();
    writer.write_buffer(0, resources.dataBuffer, sizeof(MaterialConstants), resources.dataBufferOffset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.write_image(1, resources.colorImage.imageView, resources.colorSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(2, resources.metalRoughImage.imageView, resources.metalRoughSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.update_set(device, materialData.materialSet);

    return materialData;
}

void MeshNode::Draw(const glm::mat4& rootMatrix, DrawContext& ctx)
{
    glm::mat4 nodeMatrix = rootMatrix * worldTransform;

    for (auto& surface : mesh->surfaces)
    {
        RenderObject renderObject{
            .indexCount = surface.count,
            .firstIndex = surface.startIndex,
            .indexBuffer = mesh->meshBuffers.indexBuffer.buffer,
            .material = &surface.material->data,
            .transform = nodeMatrix,
            .vertexBufferAddress = mesh->meshBuffers.vertexBufferAddress
        };

        ctx.opaqueSurfaces.push_back(renderObject);
    }

    // draw any children
    Node::Draw(rootMatrix, ctx);
}
