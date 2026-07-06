# Hardware architecture
## Core level
### PE
PE is analog compute-in-memory (CIM) unit responsible for executing the main convolution/fully connected static weight matrix multiplication tasks, with weights offline written to the memristor array. Each core has four directions of PE: East, South, West, and North. Each PE contains 8 crossbar arrays (XB), each with a size of 576x128. A PE can deploy up to 2 threads, with each thread able to use 1, 2, or 4 XBs for computation (even if the XB is not fully utilized, it still occupies a thread). The specific computation rules can be found in Mapping steps.adaptive split.
Operator config that can be supported in a PE thread:

| kernal_size     |   stride   |  dtype  |  channel_in  | channel_out | max_pading |
| ---- | ---- | ---- | ---- | ---- | ---- |
|   1   |  1/2   |  int4   |  64/128/256/512   |  64/128/256/512  |  0   |
|   1   |  1/2   |  int8   |  64/128/256/512   |  64/128/256   |  0   |
|   3   |  1/2   |  int4   |  64   |  64/128/256/512   |  1 for each side margin   |
|   3   |  1/2   |  int8   |  64   |  64/128/256/512   |  1 for each side margin   |

MatMul can be regarded as a Conv operator with kernel size = 1.

### DMAC
DMAC is digital compute-in-memory (DCIM) unit that can also perform a small amount of convolution/fully connected static weight matrix multiplication, with weights offline configured. Each core has only one array with 2 size configs: 256x64 or 512x32, which is often used to compute layers that require high precision, such as the first and last layers. Convolution/fully connected layers are generally deployed on PE by default, and manual specification is required to deploy them on DMAC. DMAC can only support 8bit input and ouput data.
Operator config that can be supported in a DMAC thread:

| kernal_size     |   stride   | channel_in  | channel_out | max_pading |
| ---- | ---- | ---- |---- |---- |
|   1   |  1  |   32/64/128/256/512   |  32  |  0   |
|   1   |  1   |  32/64/128/256   |  64 |  0   |
|   3   |  1/2   |   3/4/8/16/32   |  32   |  1 for each side   |
|   3   |  1/2   |   3/4/8/16   |  64   |  1 for each side   |
|   5   |  1/2   |   3/4/8/16  |  64   |  2 for each side   |
|   5   |  1/2   |   3/4/8  |  32   |  2 for each side   |
|   7   |  1/2   |   3/4/8   |  32   |  3 for each side   |
|   7   |  1/2   |   3/4   |  64   |  3 for each side   |
|   9   |  1/2   |   3/4   |  32   |  4 for each side   |

when kernel size > 1, at least 1 padding must be applied to all configurations.

### MFOP
MFOP is a digital compute unit that can perform operations including MaxPooling, AvgPooling and Upsample. Each core has only one MFOP unit, which can deploy up to 2 threads.
Operator config that can be supported in a MFOP thread:

| sample method    |   kernel size | stride   | dtype   | channel_in  |  max_pading |
| ---- | ---- | ---- | ---- | ---- | ---- |
| UPSAMPLE |   2/3/4   |  1  |   int4   | 64/128/256/512/1024   |  0   |
| UPSAMPLE |   2/3/4   |  1   |  int8   | 32/64/128/256/512  |  0   |
| UPSAMPLE |   5   |  1   |   int4   |  64/128/256/512   |  0   |
| UPSAMPLE |   5   |  1   |   int8   |  32/64/128/256   |  0   |
|   MaxPool/AvgPool   |  1/2   |   3/4/8/16  |  64   |  2 for each side   |
|   MaxPool/AvgPool   |  1/2   |   3/4/8  |  32   |  2 for each side   |
|   MaxPool/AvgPool   |  1/2   |   3/4/8   |  32   |  3 for each side   |
|   MaxPool/AvgPool   |  1/2   |   3/4   |  64   |  3 for each side   |
|   MaxPool/AvgPool   |  1/2   |   3/4   |  32   |  4 for each side   |

## SoC level
### NoC and routing
The NoC adopts mesh structure, including 4x9 cores. The routing is based on XY scheme, which means that when a thread from one core sends data to a thread in another core, it first aligns in the X direction and then moves in the Y direction. For example, the path from (3,4) to (2,5) would be [(3,4)->(3,5)->(2,5)]. The intermediate cores forward the data through virtual channels without needing to deploy actual threads. Only the destination core deploys the actual thread and allocates cache. It is important to note that the direction of the processing elements (PEs) is strictly bound to the transmission (TX) direction, meaning that the computing thread on a PE from a certain direction must be emitted from that direction. This may violate the XY routing principle. For example, if a PE thread from (3,4) in the North direction wants to send data to (2,5), it cannot follow the normal XY routing. In this case, an intermediate identity thread (which has no directional restrictions) needs to be inserted along the appropriate path. For instance, inserting an identity thread at (2,4) would change the path to [(3,4)->(2,4)->(2,5)].

### dataflow architecture
Dataflow architecture is adopted as a programming paradigm that allows for the asynchronous execution of operations based on the availability of data. In this architecture, computations are triggered by the arrival of data rather than by explicit control flow instructions. This is achieved through the use of a pull-based mechanism, where downstream threads can request data from upstream threads as soon as they have the necessary resources (e.g., dmem cache) to process it. Once the upstream thread receives a request, it can begin computing the required data and make it available for downstream consumption. After the downstream thread consumes the data, it can free up its cache and issue new requests as needed.

### the hardware operators
Hardware operators that can be supported includes PEConv, DMACConv, MaxPooling, AvgPooling, Upsample, Transfer[Transfer includes many operations, such as Relu (which has a direct and simple hardware implementation), special activation functions (other than Relu, which do not have direct hardware implementations and are implemented through lookup tables), identity (which does not perform any operations and only serves to route or balance the computation links), Add (reduce operations), Concat (assemble operations), and type_conversion (4bit/8bit data stream conversion)].

### Special cores
(0, 3), (0, 4), (0, 5), (3, 4), (3, 5) cores are special without PE, DMAC, MFOP modules, so they can only deploy Transfer operation. Besides, (0, 4) also acts as HOSTI, responsible for the input and output of activation values/feature map data, supporting multiple input/output.

### datawidth
4bit and 8bit dataflow is supported. Except for type_conversion operator, 4bit and 8bit dataflow must maintain consistent input and output data widths for all operations/threads. The type_conversion operator can expand 4bit to 8bit or truncate 8bit to 4bit. DMAC only supports 8bit input and output.

# Mapping steps
After the frontend of the compiler converts ONNX/PyTorch models to model_ir, several passes are needed to transform model_ir into ir with hardware mapping information.

## adaptive split
The adaptive split is designed to accommodate the specific requirements of the XB architecture for convolution and fully connected layers. If the unfolded convolution/fully connected layer cannot be deployed within a single PE thread, it needs to be split and reduced, broadcasted and concatenated, or a combination of both. The splitting is done using an averaging principle. Specifically, for example, if a convolution kernel has a size of [1024, 128, 3, 3], after flattening through the img2col method, the resulting matrix size is [1152, 1024]. Since the size of an XB is [576, 128], and a single thread can support a maximum of 4 XB columns in parallel, i.e., a maximum [576, 512] weight matrix deployment, the original convolution kernel needs to be split into 2 parts in the output channel (column direction) and 2 parts in the input channel (row direction), for a total of 2*2=4 parts. The parts split in the row direction need to be added together, while the parts split in the column direction need to be concatenated. This results in a split IR model called split_model_ir. After splitting, the whole split graph is renamed again in topological order. Each operator type uses its own counter, such as Conv_0, Conv_0_bn, Relu_0, Conv_1, Split_0, Add_0, and Concat_0; BN layers use the upstream layer name plus `_bn`, and split shards do not keep names derived from the original layer such as Conv_93_0_0. First, a summation tree is used to add the necessary parts together, and then a concatenation is performed. At the same time, BN also needs to be adaptively split to ensure it follows each convolution/fully connected layer closely, and its parameters also need to be modified accordingly. For example, if split by output channel, BN only needs to be split by channel, but if split by input channel, BN's bias and other parameters need to be divided by the number of splits. Support for exporting the mapped_ir's state_dict and weight matrix is also required.

For those Conv/MatMul layer that cannot be smoothly split into fixed configurations for PE, such as convolution with [64, 48, 1, 1], stride=1, not in the table configuration under Hardware architecture/Core Level/PE, please report an error in a timely manner. Such hardware-incompatible operators need to be adjusted by upstream algorithms, and the compiler is not responsible for adjustments.

## Deploy PE operators
First, the mapping of Conv/MatMul type operators on PE needs to be performed (at this point, they have already been split, ensuring that each shard can be placed on a thread of the PE).

There are 4*(4*9-5)=124 PEs globally, and 248 PE threads are available for allocation. The mapping adopts a priority strategy: 1. workload balance: If there are PEs without any deployed threads, prioritize deployment. After all PEs have at least one thread deployed, consider enabling a second thread. 2. distance: Minimize communication distance (routing hops/Manhattan distance), especially paying attention to PE direction binding issues to reduce intermediate thread insertion.
