# Completed

- 已完成一轮 IR_tool 与 frontend 清晰化重构：重组 `DataDef`、`GraphLayer`、ONNX converter 主流程、op handler 公共 shape/ref/constant 工具、ONNX 预处理和属性读取逻辑，保持公开 API 与主要转换语义不变。
- 项目结构已经成型：主体位于 `CIMA_TC/Compiler`，按功能分为 IR (`IR_tool`)、前端转换 (`frontend`)、硬件描述 (`hw_def`)、映射 (`mapper`)、训练代码生成后端 (`backend`) 和示例/测试 (`test`)。
- IR 核心基础较完整：`IR_tool/core` 已实现 JSON/YAML 序列化、注册表/工厂机制、引用解析、`DataDef`、`BaseLayer`/`OpLayer`/`GraphLayer`/`BlockLayer`、图输入/输出层、拓扑排序、图校验、设备树、`BaseIR` 以及 DOT/PDF 可视化辅助函数。
- IR 算子目录较丰富：`IR_tool/ops` 已定义 conv/conv-transpose、matmul/linear/fc、activation、math、pooling、normalization、shape/transform、split/slice/reduce/resize、constant、identity 等算子，并有算子创建与元数据相关单元测试。
- ONNX 前端已经有较清晰的主流程：配置驱动转换、预处理、shape inference、可选节点名规范化、按指定输入/输出裁剪图、ONNX parser 状态、op handler registry、权重导出、运行期 `weight_store`/`bn_store` 附加，以及共享 IR rewrite pass。
- ONNX op handler 覆盖了常见 CNN/模型导出算子：Conv、ConvTranspose、MatMul、Gemm、Add、Constant、Relu、Sigmoid、Tanh、LeakyRelu、Softmax、LogSoftmax、Erf、Mul、Div、Transpose、Reshape、Flatten、Concat、MaxPool、AveragePool、GlobalAveragePool、Split、ReduceMean、Sqrt、Pow、BatchNormalization、LayerNormalization、Pad、Resize、Squeeze、Unsqueeze、Gather、Slice、Silu。
- PyTorch 前端已有两条路径：`frontend/pytorch` 通过内存中的 ONNX export 转 IR；`frontend/pytorch_fx` 直接用 FX 转 IR，但覆盖范围较小。两者都支持 dump IR 和导出权重。
- 前端共享 rewrite pass 已实现：可将 `Sigmoid + Mul` 融合为 `Silu`，并将 BN 层重命名到靠近上游 conv/linear 的命名形式，便于后续权重匹配。
- CIMA 硬件抽象已有第一版：`hw_def` 描述 4x9 chip mesh、compute/non-compute core 角色、每个 compute core 内的 PE/DMAC/MFOP、PE crossbar 几何参数 (`576x128`，16 个逻辑 XB)、线程数、bit width 和粗粒度 op 白名单。
- mapper 第一阶段已完成可执行实现：`mapper/xb_split.py` 可为 `conv2d`/`linear` 计算 XB row/column split plan；`mapper/split_pass.py` 可重写 IR、切分 conv/fc 权重、插入 split/add/concat 结构、可选切分相邻 BN 参数，并导出 split IR 与权重。
- 后端已有验证型原型：`backend/to_training_code` 可将受支持的 IR 子集重建为 PyTorch `nn.Module`，加载导出的权重，并生成显式 PyTorch model/training script。
- 示例流程存在：`Compiler/test/CustomizedNet` 和 `Compiler/test/ResNet50` 包含 ONNX 资产与脚本，可串起 ONNX -> IR -> XB split -> split weights；ResNet50 脚本还会生成 DOT/PDF 图和 PyTorch model/training script。
- 测试文件基础较多：已有针对 IR core、ref、data definition、layer、device、registry、serialization、visualization、ops 的 pytest 用例，也有硬件描述和模型 split 的脚本式集成检查。
- 本次验证结果：`python -m CIMA_TC.Compiler.hw_def.test_cima_chip` 可成功运行并构建/序列化 CIMA 4x9 chip 模型；该 smoke test 生成的临时检查文件已清理。

# In Progress

- 当前环境无法运行完整 pytest：`python -m pytest -q` 失败于 `No module named pytest`。测试代码已经存在，但还不能声称全量测试通过，需要先补齐本地依赖/测试环境。
- mapper 处于“可执行的第一阶段”：目前重点是 `conv2d`/`linear` 的 XB-aware 结构拆分，还没有完整 hardware placement、scheduling、routing、memory planning 或目标指令生成。
- 硬件模型已经表达 core/unit 层级和粗粒度能力，但 NoC 邻接关系、带宽/延迟、存储层级、资源竞争和精确执行调度尚未建模。
- backend 目前更像 PyTorch reconstruction/training-code backend，用于验证 split IR 行为；它还不是 CIMA runtime backend，也不是模拟存算一体架构的目标代码生成器。
- 直接 PyTorch FX 前端仍是窄覆盖/实验性路径：当前主要支持 Conv2d、Linear/addmm、ReLU、adaptive/global average pooling、Flatten、MaxPool2d、Sigmoid 等 CNN 常见子集。
- ONNX 前端覆盖较宽，但遇到未注册或 exporter-specific op 会显式报错；shape 处理较依赖 inferred value_info，一些 handler 仍带有 best-effort 假设。
- 量化相关配置字段已经存在，如 `weight_half_level`、`weight_scale`、`data_range_specify`、`data_clamp_std`，但 parser 文件明确说明当前模块尚未实现 quantization。
- `BaseIR` 的 `weight_store`/`bn_store` 是运行期附加字段，并不会随 IR 序列化保存。这对当前 mapper 很方便，但长期的持久化 IR/权重 manifest 契约还需要稳定下来。
- 项目过程文档仍在整理：`CODEX.md` 已有代码质量要求，`PLAN.md` 为空，现有 `PPOCESS.md` 似乎是拼写错误且为空。当前进度记录以本文件 `PROCESS.md` 为准。

# Remaining

- 补充项目环境信息：依赖清单、安装说明、Python 版本预期、pytest 设置、ONNX/Torch/Graphviz 等可选依赖，以及 smoke test/full test 的可复现命令。
- 安装依赖后运行并修复完整 pytest；最好加入 CI 或至少提供统一验证脚本，让后续进度可以客观衡量。
- 完成 splitting 之后的硬件感知 mapping：将 op 分配到 PE/DMAC/MFOP，分配 XB 和 buffer，处理首/末层或高精度层，建模 inter-core communication，生成 placement/schedule artifact。
- 扩展 mapper 支持范围：除 `conv2d`/`linear` 外，还需要覆盖 pooling、resize/up-sample、elementwise、normalization、matmul、residual add pattern，以及现代 CNN/Transformer 常见 shape ops。
- 实现量化/校准与模拟 CIMA 约束：bit width、scale、clamp、ADC/DAC 假设、signed/unsigned weights、activation range，以及带数值容差的正确性验证。
- 稳定 IR schema：明确 NCHW/channel metadata 语义、dynamic shape 行为、多输入/多输出 ref、持久化权重/BN 元数据、版本号，以及 serialized IR 与外部权重文件的兼容关系。
- 强化前端鲁棒性：覆盖更多 ONNX/PyTorch/FX 导出场景，包括 dynamic axes、constant handling、grouped/depthwise conv、ConvTranspose、LayerNorm、Pad/Resize/Slice 变体，以及非 CNN 图。
- 扩展或拆分 backend：明确区分 PyTorch validation backend 与真实 CIMA backend；真实 CIMA backend 仍需要指令/代码生成与 runtime 集成。
- 添加数值等价测试：原始 PyTorch/ONNX 模型、IR reconstruction、split IR reconstruction 之间应在代表性输入上比较输出，并覆盖权重加载、BN split、残差路径等情况。
- 清理并维护过程文档：补充 `PLAN.md`，决定是否删除或重命名 `PPOCESS.md`，记录完整 pipeline，并在每个阶段性 milestone 后更新 `PROCESS.md`。
