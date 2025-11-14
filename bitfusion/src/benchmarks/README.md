benchmarks这一个子文件夹：
1. 对于不同的_bench.py:
ant_bench.py 定义了所有 13 个网络的层序列，每行包含 [输入张量, 权重, 输出, stride, padding, 精度, 操作类型]；以 VGG16 为例，只有首层使用 8bit，其余卷积/全连接多半是 4bit，体现 ANT 的混合精度策略 (bitfusion/src/benchmarks/ant_bench.py (lines 2-19))。
ant_weight_bench.py 和 ant_bench.py 在同一网络（如 ResNet18）的每一行都完全一致，连 stride、padding、层级数都相同，只是文件名字不同；因此这两个版本确实只有精度策略相同（都是 8/4bit 混合），没有额外结构差异 (bitfusion/src/benchmarks/ant_weight_bench.py (lines 2-42) 与 bitfusion/src/benchmarks/ant_bench.py (lines 2-42))。
adafloat_bench.py 将所有层（包括 VGG16 的三段卷积和全连接）都固定为 8bit，层描述与 ant_bench 一致，唯一差别就是精度字段统一为 8 (bitfusion/src/benchmarks/adafloat_bench.py (lines 2-19))。
olaccel_bench.py 复用了相同的层配置，但将除第一层外的 VGG16 卷积、ResNet 中的大部分层设为 4bit；结构、stride、padding 仍与 ant_bench 匹配 (bitfusion/src/benchmarks/olaccel_bench.py (lines 2-30))。
biscaled_bench.py 也沿用同一层列表，只是把所有层（卷积和全连接）都标为 6bit，因而和其他版本相比的确只在精度列不同 (bitfusion/src/benchmarks/biscaled_bench.py (lines 2-30))。
bitfusion_bench.py 对 CNN/GLUE 网络的条目同样匹配 ant_bench 的形状——例如 ResNet18 的前几层与 ant_bench 完全一致，只是精度字段多为 8bit (bitfusion/src/benchmarks/bitfusion_bench.py (lines 21-43) 对照 bitfusion/src/benchmarks/ant_bench.py (lines 21-43))。
唯一的例外是 ViT：ant_bench 把 ViT 看成 49 个全连接/投影层，而 bitfusion_bench 在列表最前面多出一个 16×16 patch embedding 卷积层，并且层数为 50，且精度模式几乎全 8bit；两者不只是精度不同，还包含额外的卷积算子与不同的顺序 (bitfusion/src/benchmarks/bitfusion_bench.py (lines 200-250) vs bitfusion/src/benchmarks/ant_bench.py (lines 207-257))。换言之，只有 ViT 的 BitFusion 版本在结构上与其他 *_bench 不同，其余网络确实只在精度字段上变化。