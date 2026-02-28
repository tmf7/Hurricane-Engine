#include "vk_pipelines.h"
#include <fstream>
#include "vk_initializers.h"

bool vkutil::load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule)
{
	std::ifstream file(filePath, std::ios::ate | std::ios::binary);

	if (!file.is_open()) {
		return false;
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();

	// DEBUG: std::vector default allocator ensures data satisfies worst-case alignment requirements
	// so reinterpret_cast from char to uint32_t does not adversly affect byte alignment here
	VkShaderModuleCreateInfo createInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.codeSize = buffer.size(),
		.pCode = reinterpret_cast<const uint32_t*>(buffer.data())
	};

	VkShaderModule shaderModule;
	if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
		return false;
	}

	*outShaderModule = shaderModule;
	return true;
}

void PipelineBuilder::clear()
{
	_inputAssembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
	};

	_rasterizer = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO
		// DEBUG: the following are all left as default values
		// depthClampEnable;
		// rasterizerDiscardEnable;
		// depthBiasEnable;
		// depthBiasConstantFactor;
		// depthBiasClamp;
		// depthBiasSlopeFactor;
	};

	_colorBlendAttachment = {};

	_multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO
	};

	_pipelineLayout = {};

	_depthStencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO
	};

	_renderInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO
	};

	_shaderStages.clear();
}

VkPipeline PipelineBuilder::build_pipeline(VkDevice device)
{
	// dynamic viewport
	VkPipelineViewportStateCreateInfo viewportState{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.viewportCount = 1,
		.pViewports = nullptr, // dynamic
		.scissorCount = 1,
		.pScissors = nullptr // dynamic
	};

	// TODO (TF 27 FEB 2026): unused for now
	VkPipelineColorBlendStateCreateInfo colorBlending{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.logicOpEnable = VK_FALSE,
		.logicOp = VK_LOGIC_OP_COPY,
		.attachmentCount = 1,
		.pAttachments = &_colorBlendAttachment,
		// .blendConstants = {}
	};

	// TODO (TF 27 FEB 2026): unused for now
	VkPipelineVertexInputStateCreateInfo vertexInputInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
	};

	VkDynamicState dynamicStates[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR
	};

	VkPipelineDynamicStateCreateInfo dynamicStateInfo {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.dynamicStateCount = static_cast<uint32_t>(std::size(dynamicStates)),
		.pDynamicStates = dynamicStates
	};

	VkGraphicsPipelineCreateInfo pipelineInfo{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = nullptr,
		.flags = 0,
		.stageCount = static_cast<uint32_t>(_shaderStages.size()),
		.pStages = _shaderStages.data(),
		.pVertexInputState = &vertexInputInfo,
		.pInputAssemblyState = &_inputAssembly,
		//.pTessellationState = nullptr,
		.pViewportState = &viewportState,
		.pRasterizationState = &_rasterizer,
		.pMultisampleState = &_multisampling,
		.pDepthStencilState = &_depthStencil,
		.pColorBlendState = &colorBlending,
		.pDynamicState = &dynamicStateInfo,
		.layout = _pipelineLayout
		// .renderPass = VK_NULL_HANDLE // dynamic rendering is used
		// .subpass = 0, // dynamic rendering is used
		// .basePipelineHandle = VK_NULL_HANDLE,
		// .basePipelineIndex = 0
	};

	VkPipeline newPipeline;
	if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS) {
		std::cout << "Faild to create pipeline!" << std::endl;
		return VK_NULL_HANDLE;
	}

	return newPipeline;
}

void PipelineBuilder::set_shaders(VkShaderModule vertexShader, VkShaderModule fragmentShader)
{
	_shaderStages.clear();

	_shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_VERTEX_BIT, vertexShader));
	_shaderStages.push_back(vkinit::pipeline_shader_stage_create_info(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShader));

}

void PipelineBuilder::set_input_topology(VkPrimitiveTopology toplogy)
{
	_inputAssembly.topology = toplogy;
	_inputAssembly.primitiveRestartEnable = VK_FALSE;
}

void PipelineBuilder::set_polygon_mode(VkPolygonMode mode)
{
	_rasterizer.polygonMode = mode;
	_rasterizer.lineWidth = 1.0f;
}

void PipelineBuilder::set_cull_mode(VkCullModeFlags cullMode, VkFrontFace frontFace)
{
	_rasterizer.cullMode = cullMode;
	_rasterizer.frontFace = frontFace;
}

void PipelineBuilder::set_multisampling_none()
{
	_multisampling.sampleShadingEnable = VK_FALSE;
	_multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	_multisampling.minSampleShading = 1.0f;
	_multisampling.pSampleMask = nullptr;
	_multisampling.alphaToCoverageEnable = VK_FALSE;
	_multisampling.alphaToOneEnable = VK_FALSE;
}

void PipelineBuilder::disable_blending()
{
	// DEBUG: all left in current/default state
	// srcColorBlendFactor;
	// dstColorBlendFactor;
	// colorBlendOp;
	// srcAlphaBlendFactor;
	// dstAlphaBlendFactor;
	// alphaBlendOp;
	_colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT
										 | VK_COLOR_COMPONENT_G_BIT
										 | VK_COLOR_COMPONENT_B_BIT
										 | VK_COLOR_COMPONENT_A_BIT;
	_colorBlendAttachment.blendEnable = VK_FALSE;
}

void PipelineBuilder::set_color_attachment_format(VkFormat format)
{
}

void PipelineBuilder::set_depth_format(VkFormat format)
{
}

void PipelineBuilder::disable_depthtest()
{
}
