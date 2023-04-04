#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include <vector>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <cmath>
#include <fstream>
#include <array>

#include <random>
#include <ctime>
#include <chrono>

/*
device       = NVIDIA GeForce RTX 3090

fp32-scalar  = 20966.47 GFLOPS
fp32-vec4    = 26719.68 GFLOPS

fp16-scalar  = 20481.88 GFLOPS
fp16-vec4    = 40258.06 GFLOPS

fp64-scalar  = 643.81 GFLOPS
fp64-vec4    = 643.82 GFLOPS

int32-scalar = 20457.37 GIOPS
int32-vec4   = 20206.93 GIOPS

int16-scalar = 13301.41 GIOPS
int16-vec4   = 16060.26 GIOPS
*/

const int local_size_x = 32;
int loop = 1024;
int count = 256 * 1024; // 1MB
const int cmd_loop = 20;

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

#define VK_CHECK_RESULT(f) 																				\
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__);                  \
        assert(res == VK_SUCCESS);																		\
    }																									\
}

class ComputeApplication {
private:

    std::vector<const char *> enabledLayers;    
    VkDebugReportCallbackEXT debugReportCallback;

    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;

    glm::vec4* aData;
    glm::vec4* bData;
    int dataSize;
    VkBuffer aBuffer;
    VkBuffer bBuffer;
    VkBuffer cBuffer;
    VkDeviceMemory aBufferMemory;
    VkDeviceMemory bBufferMemory;
    VkDeviceMemory cBufferMemory;

    VkDescriptorSetLayout myDescriptorSetLayout;
    VkDescriptorPool myDescriptorPool;
    VkDescriptorSet myDescriptorSet;

    VkShaderModule myComputeShaderModule;
    VkPipelineLayout myPipelineLayout;
    VkPipeline myPipeline;

    VkCommandPool myCommandPool;
    VkCommandBuffer myCommandBuffer;

    VkQueue queue;
    uint32_t computeQueueFamilyIndex;

public:
    void run() {

        double max_gflops = -1;

        bool rerun = true;

        while (rerun) {
            rerun = false;

            max_gflops = -1;

            for (int i = 0; i < cmd_loop; i++) {

                aData = new glm::vec4[count];
                bData = new glm::vec4[count];
                dataSize = sizeof(glm::vec4) * count;

                createInstance();
                findPhysicalDevice();
                createDevice();
                createBuffer();
                createDescriptorSetLayout();
                createDescriptorSet();
                createComputePipeline();
                createCommandBuffer();

                auto start = std::chrono::high_resolution_clock::now();
                runCommandBuffer();
                auto end = std::chrono::high_resolution_clock::now();
                auto time_span = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                double time = time_span.count() / 1000.0;
                double mac = (double)count * (double)loop * 16 * 4 * 2;
                double gflops = mac / time / 1000000;
                max_gflops = std::max(max_gflops,gflops);
                
                cleanup();

                delete[] aData;
                delete[] bData;

                if (time < 500) {
                    loop *= 2;
                    count *= 2;
                    rerun = true;
                    break;
                }
            }
        }

        printf("fp32-scalar = %.2f GFLOPS\n", max_gflops);
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
        VkDebugReportFlagsEXT                       flags,
        VkDebugReportObjectTypeEXT                  objectType,
        uint64_t                                    object,
        size_t                                      location,
        int32_t                                     messageCode,
        const char*                                 pLayerPrefix,
        const char*                                 pMessage,
        void*                                       pUserData) {

        printf("Validation Debug Report: %s: %s\n", pLayerPrefix, pMessage);

        return VK_FALSE;
    }

    static uint32_t find_device_compute_queue(const std::vector<VkQueueFamilyProperties>& queueFamilyProperties)
    {
        for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
        {
            const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

            if ((queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
                    && !(queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT))
            {
                return i;
            }
        }

        for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
        {
            const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

            if ((queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
                    && (queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT))
            {
                return i;
            }
        }

        for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
        {
            const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

            if (queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                return i;
            }
        }

        return -1;
    }

    static std::vector<char> readFile(const std::string& filename) 
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Could not find or open spv file!");
        }

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }

    void createInstance() 
    {
        std::vector<const char *> enabledExtensions;

        if (enableValidationLayers) {
            uint32_t layerCount;
            vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
            std::vector<VkLayerProperties> layerProperties(layerCount);
            vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());
            bool foundLayer = false;
            for (VkLayerProperties prop : layerProperties) {
                if (strcmp("VK_LAYER_LUNARG_standard_validation", prop.layerName) == 0) {
                    foundLayer = true;
                    break;
                }
            }
            if (!foundLayer) {
                throw std::runtime_error("Layer VK_LAYER_LUNARG_standard_validation not supported\n");
            }
            enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation");

            uint32_t extensionCount;
            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
            std::vector<VkExtensionProperties> extensionProperties(extensionCount);
            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensionProperties.data());
            bool foundExtension = false;
            for (VkExtensionProperties prop : extensionProperties) {
                if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, prop.extensionName) == 0) {
                    foundExtension = true;
                    break;
                }
            }
            if (!foundExtension) {
                throw std::runtime_error("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
            }
            enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        }		


        VkApplicationInfo applicationInfo = {};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "vulkan compute app";
        applicationInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
        applicationInfo.pEngineName = "awesomeengine";
        applicationInfo.engineVersion = VK_MAKE_VERSION(1,0,0);
        applicationInfo.apiVersion = VK_API_VERSION_1_0;;
        
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.flags = 0;
        createInfo.pApplicationInfo = &applicationInfo;
        
        createInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
        createInfo.ppEnabledLayerNames = enabledLayers.data();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
        createInfo.ppEnabledExtensionNames = enabledExtensions.data();
    
        VK_CHECK_RESULT(vkCreateInstance(&createInfo, nullptr, &instance));


        if (enableValidationLayers) {
            VkDebugReportCallbackCreateInfoEXT createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
            createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
            createInfo.pfnCallback = &debugReportCallbackFn;

            auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
            if (vkCreateDebugReportCallbackEXT == nullptr) {
                throw std::runtime_error("Could not load vkCreateDebugReportCallbackEXT");
            }

            VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(instance, &createInfo, nullptr, &debugReportCallback));
        }
    }

    void findPhysicalDevice() 
    {
        uint32_t deviceCount;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("could not find a device with vulkan support");
        }
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        // show all device
        for (int i = 0; i < devices.size(); i++) {
            VkPhysicalDevice device = devices[i];
            VkPhysicalDeviceProperties physicalDeviceProperties;
            vkGetPhysicalDeviceProperties(device, &physicalDeviceProperties);

            uint32_t queueFamilyPropertiesCount;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyPropertiesCount, 0);
            std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertiesCount);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyPropertiesCount, queueFamilyProperties.data());
            computeQueueFamilyIndex = find_device_compute_queue(queueFamilyProperties);
        
            uint32_t compute_queue_count = queueFamilyProperties[computeQueueFamilyIndex].queueCount;

            printf("[%u %s]  queueC=%u[%u]\n",  i, physicalDeviceProperties.deviceName, computeQueueFamilyIndex, compute_queue_count);

            physicalDevice = device;
            break;
        }
    }

    void createDevice() {
        // specify compute queue
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        float queuePriorities = 1.0;
        queueCreateInfo.pQueuePriorities = &queuePriorities;

        // create logical device
        VkDeviceCreateInfo deviceCreateInfo = {};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(enabledLayers.size());
        deviceCreateInfo.ppEnabledLayerNames = enabledLayers.data();
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 1;
        VkPhysicalDeviceFeatures deviceFeatures = {};
        deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

        VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

        vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &queue);
    }

    uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memoryProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((memoryTypeBits & (1 << i)) && ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
                return i;
        }

        return -1;
    }

    void createBuffer() {
        {
            VkBufferCreateInfo bufferCreateInfo = {};
            bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferCreateInfo.size = dataSize;
            bufferCreateInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &aBuffer));

            VkMemoryRequirements memoryRequirements;
            vkGetBufferMemoryRequirements(device, aBuffer, &memoryRequirements);
            
            VkMemoryAllocateInfo allocateInfo = {};
            allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocateInfo.allocationSize = memoryRequirements.size;
            allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

            VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, nullptr, &aBufferMemory));
            VK_CHECK_RESULT(vkBindBufferMemory(device, aBuffer, aBufferMemory, 0));

            void* data;
            vkMapMemory(device, aBufferMemory, 0, VK_WHOLE_SIZE, 0, &data);
            memcpy(data, aData, dataSize);
            vkUnmapMemory(device, aBufferMemory);
        }
        {
            VkBufferCreateInfo bufferCreateInfo = {};
            bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferCreateInfo.size = dataSize;
            bufferCreateInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &bBuffer));

            VkMemoryRequirements memoryRequirements;
            vkGetBufferMemoryRequirements(device, bBuffer, &memoryRequirements);
            
            VkMemoryAllocateInfo allocateInfo = {};
            allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocateInfo.allocationSize = memoryRequirements.size;
            allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

            VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, nullptr, &bBufferMemory));
            VK_CHECK_RESULT(vkBindBufferMemory(device, bBuffer, bBufferMemory, 0));

            void* data;
            vkMapMemory(device, bBufferMemory, 0, VK_WHOLE_SIZE, 0, &data);
            memcpy(data, bData, dataSize);
            vkUnmapMemory(device, bBufferMemory);
        }
        {
            VkBufferCreateInfo bufferCreateInfo = {};
            bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferCreateInfo.size = dataSize;
            bufferCreateInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, &cBuffer));

            VkMemoryRequirements memoryRequirements;
            vkGetBufferMemoryRequirements(device, cBuffer, &memoryRequirements);
            
            VkMemoryAllocateInfo allocateInfo = {};
            allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocateInfo.allocationSize = memoryRequirements.size;
            allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

            VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, nullptr, &cBufferMemory));
            VK_CHECK_RESULT(vkBindBufferMemory(device, cBuffer, cBufferMemory, 0));
        }
    }

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding aBinding = {};
        aBinding.binding = 0;
        aBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        aBinding.descriptorCount = 1;
        aBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding bBinding = {};
        bBinding.binding = 1;
        bBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bBinding.descriptorCount = 1;
        bBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutBinding cBinding = {};
        cBinding.binding = 2;
        cBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        cBinding.descriptorCount = 1;
        cBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        std::vector<VkDescriptorSetLayoutBinding> bindings = { aBinding, bBinding, cBinding };

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo = {};
        descriptorSetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        descriptorSetLayoutInfo.pBindings = bindings.data();

        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutInfo, nullptr, &myDescriptorSetLayout));
    }

    void createDescriptorSet() {
        std::array<VkDescriptorPoolSize, 3> poolSizes = {};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = 1;
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[2].descriptorCount = 1;

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &poolInfo, nullptr, &myDescriptorPool));

        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = myDescriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &myDescriptorSetLayout;
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &myDescriptorSet));

        std::array<VkWriteDescriptorSet, 3> descriptorWrites = {};
        VkDescriptorBufferInfo aBufferInfo = {};
        aBufferInfo.buffer = aBuffer;
        aBufferInfo.offset = 0;
        aBufferInfo.range = dataSize;
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = myDescriptorSet;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &aBufferInfo;
        VkDescriptorBufferInfo bBufferInfo = {};
        bBufferInfo.buffer = bBuffer;
        bBufferInfo.offset = 0;
        bBufferInfo.range = dataSize;
        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = myDescriptorSet;
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &bBufferInfo;
        VkDescriptorBufferInfo cBufferInfo = {};
        cBufferInfo.buffer = cBuffer;
        cBufferInfo.offset = 0;
        cBufferInfo.range = dataSize;
        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = myDescriptorSet;
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pBufferInfo = &cBufferInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()),descriptorWrites.data(), 0, nullptr);
    }

    void createComputePipeline() {
        std::vector<char> code = readFile("shaders/test.spv");
        VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
        shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        shaderModuleCreateInfo.codeSize = code.size();
        VK_CHECK_RESULT(vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &myComputeShaderModule));

        VkSpecializationMapEntry constantEntry{};
        constantEntry.constantID = 0;
        constantEntry.offset = 0;
        constantEntry.size = sizeof(int);

        VkSpecializationInfo specializationInfo{};
        specializationInfo.mapEntryCount = 1;
        specializationInfo.pMapEntries = &constantEntry;
        specializationInfo.dataSize = sizeof(int);
        specializationInfo.pData = &loop;

        VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
        shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageCreateInfo.module = myComputeShaderModule;
        shaderStageCreateInfo.pName = "main";
        shaderStageCreateInfo.pSpecializationInfo = &specializationInfo;

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts = &myDescriptorSetLayout;
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &myPipelineLayout));

        VkComputePipelineCreateInfo pipelineCreateInfo = {};
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stage = shaderStageCreateInfo;
        pipelineCreateInfo.layout = myPipelineLayout;
        VK_CHECK_RESULT(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &myPipeline));
    }

    void createCommandBuffer() {
        VkCommandPoolCreateInfo commandPoolCreateInfo = {};
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.flags = 0;
        commandPoolCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
        VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &myCommandPool));

        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = myCommandPool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &myCommandBuffer));
    
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK_RESULT(vkBeginCommandBuffer(myCommandBuffer, &beginInfo));

        vkCmdBindPipeline(myCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, myPipeline);
        vkCmdBindDescriptorSets(myCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, myPipelineLayout, 0, 1, &myDescriptorSet, 0, nullptr);
        vkCmdDispatch(myCommandBuffer, count/local_size_x, 1, 1);
        VK_CHECK_RESULT(vkEndCommandBuffer(myCommandBuffer));
    }

    void runCommandBuffer() {
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &myCommandBuffer;

        VkFence fence;
        VkFenceCreateInfo fenceCreateInfo = {};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = 0;
        VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));

        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));

        VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000));

        vkDestroyFence(device, fence, nullptr);
    }

    void cleanup() {
        if (enableValidationLayers) {
            auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
            if (func == nullptr) {
                throw std::runtime_error("Could not load vkDestroyDebugReportCallbackEXT");
            }
            func(instance, debugReportCallback, nullptr);
        }

        vkFreeMemory(device, aBufferMemory, nullptr);
        vkFreeMemory(device, bBufferMemory, nullptr);
        vkFreeMemory(device, cBufferMemory, nullptr);
        vkDestroyBuffer(device, aBuffer, nullptr);
        vkDestroyBuffer(device, bBuffer, nullptr);
        vkDestroyBuffer(device, cBuffer, nullptr);

        vkDestroyShaderModule(device, myComputeShaderModule, nullptr);
        vkDestroyDescriptorPool(device, myDescriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, myDescriptorSetLayout, nullptr);
        vkDestroyPipelineLayout(device, myPipelineLayout, nullptr);
        vkDestroyPipeline(device, myPipeline, nullptr);
        vkDestroyCommandPool(device, myCommandPool, nullptr);	

        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);		
    }
};

int main() {
    ComputeApplication app;

    try {
        app.run();
    }
    catch (const std::runtime_error& e) {
        printf("%s\n", e.what());
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
