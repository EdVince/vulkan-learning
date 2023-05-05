# Vulkan-Learning

记录一下学习vulkan过程中的一些代码

CMakeLists.txt是用chatgpt生成的，非常好用

## Experiment
 * 1_check_environment: 按照[教程](https://vulkan-tutorial.com/Development_environment)配环境并写个简单的代码检查一下装到位没有
 * 2_check_instance: 基于上一个，尝试创建一个instance来进一步检查vulkan是否可用
 * 3_vulkan_tutorial: 按照[教程](https://vulkan-tutorial.com/Multisampling)走的简单渲染，代码非常多
 * 4_vulkan_minimal_compute: 基于[vulkan_minimal_compute](https://github.com/Erkaman/vulkan_minimal_compute)，用于配置一个跑vulkan compute的pipeline
 * 5_element_add: 基于4魔改，实现了output=input+1的功能
 * 6_element_add_vec4: 基于5魔改，实现了output1=input1+input2的功能，并使用了vec4数据类型和std140布局
 * 7_vkpeak: 跑了一下[vkpeak](https://github.com/nihui/vkpeak)这个项目，没动过代码
 * 8_add_constant: 在6的基础上，增加使用了一个常量，为后面做准备
 * 9_vkpeak_fp32: 在8的基础上，增加实现了vkpeak中计算fp32-scalar的GFLOPS的功能
 * 10_vkpeak_fp32vec4: 在9的基础上，实现了fp32的vec4的GFLOPS统计，并增加了windows的配置，后面都用windows开发了