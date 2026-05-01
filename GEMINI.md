# 项目概述 (Project Overview)

**GpyTorch: AI 进化史实战大师课** 是一个专门针对 Apple Silicon (M1 Mac) 用户的进阶深度学习实战课程。课程以 AI 进化史为时间线，通过各个历史阶段的架构突破和“卡脖子”瓶颈，带领学习者重走深度学习的发展历程。

项目目录按周（Week 01 到 Week 20）组织，从最基础的感知机和 XOR 危机出发，逐步扩展到 CNN、ResNet、RNN、LSTM 和 Transformer，最后涵盖 Pre-training、RLHF 甚至 Agent 框架等前沿技术。项目不提倡纯 API 调用，而是主张在 Jupyter Notebook 中进行“逻辑复现”，手写代码直击技术本质。

# 构建与运行 (Building and Running)

项目需要 Conda 环境，并专门针对 Apple Metal Performance Shaders (MPS) 版本的 PyTorch 进行了优化配置。

### 环境配置

1. 推荐 M1 Mac 用户安装 `Miniforge`。
2. 创建并激活专用的 Conda 环境：
   ```bash
   conda create -n pytorch_env python=3.10 -y
   conda activate pytorch_env
   ```
3. 安装为 M1 编译的 PyTorch：
   ```bash
   pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

### 运行方式

- 使用 Trae 或 VSCode 打开项目所在目录。
- 进入对应的 `WeekXX` 文件夹，打开 `.ipynb` 文件。
- 在右上角的内核 (Kernel) 选择器中，选择刚刚创建的 `pytorch_env` Python 虚拟环境，即可开始执行代码。

# 开发规范 (Development Conventions)

本项目具有严格的代码规范，目的是为了保证最高的教学价值和代码可读性：

- **代码格式化：** 所有代码都必须符合 **Flake8** 格式规范，保证代码的整洁与一致性。
- **完整的 Docstring：** 每一个类和方法都必须包含清晰的 Docstring 注释，明确说明其功能、输入参数以及返回结果。
- **逐行中文注释：** 在编写代码时，每一行必须添加详尽的中文注释（尤其在进行数学逻辑或张量运算时），不能遗漏任何一行推导。
- **禁用复杂语法：** 为了保障初学者和读者的理解，**严禁使用三元运算符（单行 if-else 条件表达式）和复杂的单行列表推导式（List Comprehension）**。所有条件判断都必须使用标准的多行缩进代码块展开。
