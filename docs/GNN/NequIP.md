# NEQUIP

E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials.

+ useful links：
  + NequIP [paper](https://www.nature.com/articles/s41467-022-29939-5)
  + NequIP [github](https://github.com/mir-group/nequip)
  + Allegro [paper](https://www.nature.com/articles/s41467-023-36329-y)
  + Allegro [github](https://github.com/mir-group/allegro)
  + 欧几里得神经网络库 [e3nn](https://e3nn.org/)

data pipline:

+ => data # AtomicDataDict(edge_index, pos, batch, ptr, pbc, edge_cell_shift, cell, r_max, atom_types)
+ => OneHotAtomEncoding  # one-hot
+ => SphericalHarmonicEdgeAttrs # spharm_edges
+ => RadialBasisEdgeEncoding # radial_basis
+ => AtomwiseLinear1 # chemical_embedding
+ => ConvNetLayer1 # convnet layer1
+ => ConvNetLayer2 # convnet layer2
+ => ConvNetLayer3 # convnet layer3
+ => AtomwiseLinear2 # conv_to_output_hidden
+ => AtomwiseLinear3 # output_hidden_to_scalar
+ => AtomwiseReduce # total_energy_sum

## data

+ 原始数据集[链接](http://quantum-machine.org/gdml/data/npz/toluene_ccsd_t.zip)
+ 默认测试数据集为toluene，也就是甲苯，其分子式为C7H8，每个分子由15个原子
+ 数据集一共有1000个不同的构型

data为AtomicDataDict对象，包含以下属性：

+ atom_types: [bs*natom, 1]
+ pos: [bs*natom, 3]
+ cell: [bs, 3, 3]
+ edge_index: [2, nedges]
+ edge_cell_shift: [nedges, 3]
+ batch: [bs*natom] # [0]*natom + [1]*natom ... + [bs-1]*natom
+ ptr: [0, 1*natom, 2*natom, ...]
+ r_max: 4.0
+ pbc: [bs, 3]

## OneHotAtomEnCoding

```python
...
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
    type_numbers = data['atom_types'].squeeze(-1)
    one_hot = torch.nn.functional.one_hot(
        type_numbers, num_classes=self.num_types
    ).to(device=type_numbers.device, dtype=data['pos'].dtype)
    data['node_attrs'] = one_hot
    if self.set_features:
        data['node_features'] = one_hot
    return data
```

从代码执行来看，`OneHotAtomEnCoding`增加了两项`node_attrs`和`node_features`，它们的值都是onehotencoding的结果。注意，这里是onehotencoding，而不是原子的embedding。

+ node_attrs: [bs*natom, 2]  # 这里的2表示该分子中只有两种原子类型。
+ node_features: [bs*natom, 2]

## SphericalHarmonicEdgeAttrs

```python
class SphericalHarmonicEdgeAttrs(...)
    def __init__(...)
        ...
        self.sh = o3.SphericalHarmonics(
                self.irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
            )
    ...
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=False)
        edge_vec = data['edge_vectors']
        edge_sh = self.sh(edge_vec)
        data[self.out_field] = edge_sh
        return data
```

首先调用类方法`AtomicDataDict.with_edge_vectors`计算了edge的向量`edge_vectors`，即两相邻原子的坐标相减。再通过球谐函数self.sh，计算edge_vector的特征`edge_attrs`

+ edge_vectors: [nedges, 3] # 每对edge的两个原子坐标相减
+ edge_attrs: [nedges, 4] # (x, y, z) 通过球谐函数得到(1, x', y', z')

### SphericalHarmonics函数的计算

```python
class SphericalHarmonics(torch.nn.Module):
    def __init__(self, irreps_out, normalize, normalization):
        ...
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # - PROFILER - with torch.autograd.profiler.record_function(self._prof_str):
        if self.normalize:
            x = torch.nn.functional.normalize(x, dim=-1)  # forward 0's instead of nan for zero-radius

        sh = _spherical_harmonics(self._lmax, x[..., 0], x[..., 1], x[..., 2])

        if not self._is_range_lmax:
            sh = torch.cat([sh[..., l * l : (l + 1) * (l + 1)] for l in self._ls_list], dim=-1)

        if self.normalization == "integral":
            sh.div_(math.sqrt(4 * math.pi))
        elif self.normalization == "norm":
            sh.div_(
                torch.cat(
                    [math.sqrt(2 * l + 1) * torch.ones(2 * l + 1, dtype=sh.dtype, device=sh.device) for l in self._ls_list]
                )
            )

        return sh
```

可以看到`SphericalHarmonics`的主要部分调用了`_spherical_harmonics`函数，输入为lmax和三维坐标信息(x,y,z)，输出的维度根据lmax而变化：

+ lmax=1, 输出为(1, x', y', z')，维度为1+3 = 4
+ lmax=2, 输出为(1, x', y', z', (xz), (xy), (y2,x2z2), (yz), (z2x2))， 维度为1+3+5 = 9
+ lmax=3, 输出为(1, x', y', z', ...), 维度为1+3+5+7=16
+ 依次类推 ...

## RadialBasisEdgeEncoding

```python
class RadialBasisEdgeEncoding(...):
    def __init__(self, basis=BesselBasis, cutoff=PolynomialCutoff, ...):
        ...

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        edge_length = data['edge_lengths']
        edge_length_embedded = (
            self.basis(edge_length) * self.cutoff(edge_length)[:, None]
        )
        data['edge_embedding'] = edge_length_embedded
        return data
```

这里显示计算了每条edge的长度edge_lengths，然后通过`BesselBasis`对长度进行展开，作为径向函数。注意，这里还有一个长度的截断项cutoff。

+ edge_lengths: [bs*natom]
+ edge_embeding: [bs*natom, 8]

## AtomwiseLinear

```python
class AtomwiseLinear(...):
    def __init__(...):
        ...
        self.linear = Linear(
            irreps_in=self.irreps_in[field], irreps_out=self.irreps_out[out_field]
        )
    def forward(self, data):
        data['node_features'] = self.linear('node_features')
        return data
```

对`node_feature`做一个线性变换。注意，这里的线性变换跟常规的不同，它是基于e3nn的线性变换。观察其构造函数：

```python
self.linear = Linear(irreps_in='2x0e', irreps_out='32x0e')
assert self.linear.weight_numel==64
```

可以看到这里它是将两个scalar（`2x0e`）线性变换到32个scalar（`32x0e`），因此一共有64个参数。

+ node_features: [bs*natom, 2] => [bs*natom, 32]

## ConvNetLayer1

```python
class ConvNetLayer(...):
    def __init__(self, ..., convolution=InteractionBlock,...):
        ...
        self.conv = convolution(
            irreps_in=self.irreps_in,
            irreps_out=conv_irreps_out,
            **convolution_kwargs,
        )
        self.equivariant_nonlin = e3nn.nn.Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    acts[nonlinearity_scalars[ir.p]] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[acts[nonlinearity_gates[ir.p]] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
            # save old features for resnet
            old_x = data['node_features']
            # run convolution
            data = self.conv(data)
            # do nonlinearity
            data['node_features'] = self.equivariant_nonlin(data['node_features'])
            # do resnet
            if self.resnet:
                data["node_features"] = old_x + data["node_features"]
            return data
```

这里主要调用了两个函数`self.conv`和`self.equivariant_nonlin`,前者做特征的融合，后者相当于对`node_features`做一个激活函数，引入非线性。
观察`self.equivarient_nonlin`：

```python
print(self.equivariant_nonlin.act_scalars)
# output
# Activation [x] (32x0e -> 32x0e)

print(self.equivariant_nonlin.act_gates)
# output
# Activation [x] (32x0e -> 32x0e)

print(self.equivariant_nonlin.mul)
# output
ElementwiseTensorProduct(32x1o x 32x0e -> 32x1o | 32 paths | 0 weights)
```

可以发现它其实是一个e3nn.nn.Gate对象，参考[这里](https://docs.e3nn.org/en/stable/api/nn/nn_gate.html)，三个输入分别为`32x0e`, `32x0e`, `32x1o`，由此可以计算出输入为一个长度为32+32+32*3=160的vector。输出为32+32*3=128长度的vector。

### InteractionBlock

这是nequip里面很重要的一个模块。

```python
class InteractionBlock(...):
    ...
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
            """
            Evaluate interaction Block with ResNet (self-connection).
            """
            weight = self.fc("edge_embedding") # [nedges, 8] => [nedges, 64]

            x = data["node_features"] # [bs*natom, 32]
            edge_src = data['edge_index'][1] # 768
            edge_dst = data['edge_index'][0] # 768

            if self.sc is not None:
                sc = self.sc(x, data['node_attrs']) # [bs*natom, 32], [bs*natom, 2] => [bs*natom, 160]

            x = self.linear_1(x) # [bs*natom, 32] => [bs*natom, 32]
            edge_features = self.tp(
                x[edge_src], data['edge_attrs'], weight
            )
            x = scatter(edge_features, edge_dst, dim=0, dim_size=len(x))

            # Necessary to get TorchScript to be able to type infer when its not None
            avg_num_neigh: Optional[float] = self.avg_num_neighbors
            if avg_num_neigh is not None:
                x = x.div(avg_num_neigh**0.5)

            x = self.linear_2(x)

            if self.sc is not None:
                x = x + sc

            data['node_features'] = x
            return data
```

其中，`self.fc`看起来就是一个全连接层，输入维度为8，输出维度为64。`weight`包含的是键长的信息。

`self.sc`是一个`FullyConnectedTensorProduct`对象:

```python
print(self.sc)
# output
# FullyConnectedTensorProduct(32x0e x 2x0e -> 64x0e+32x1o | 4096 paths | 4096 weights)
```

第一个输入为`32x0e`的data['node_features']，第二个输入为`2x0e`的data['node_attrs]，输出指定为`64x0e+32x1o`。输入1和输入2的`FullTensorProduct`结果为`64x0e`,这里指定的结果是`64x0e+32x1o`，由于`1o`并未出现在输入，因此无需考虑。考虑输出为`64x0e`，则权重参数为64\*64=4096。但是检查`self.sc`的输出为160维度，按照计算64+32\*3=160。可以猜测后面的32\*3对应指定输出的`32x1o`。经过测试发现，后面的32\*3元素的值均为0，看来是用的0作为填充。

总的来看，`self.sc`将节点信息与edge的信息（球谐表示）进行了混合。

+ sc: [bs*natom, 160]

`self.tp`是一个`o3.TensorProduct`对象：

```python
print(self.tp)
# output
# TensorProduct(32x0e x 1x0e+1x1o -> 32x0e+32x1o | 64 paths | 64 weights)

# input1: x[edge_src] [num_edges, 32]
# input2: data['edge_attrs'] [num_edges, 4]
# weight: [num_edges, 64]
```

可以看到，其输出为32+32*3=128维。~~由于计算规则太过于复杂，且文档写的不够清楚，目前还没理解self.tp计算规则。~~这里的作用时候将节点信息与edge信息（投影到球谐坐标后的参数）还有边长进行信息融合。

self.tp的计算方式可能是先计算input1和input2的tensor product，然后将weight作为权重作用到每一个tensor product的输出的path上。

最后，在融合了键长、球谐坐标和节点特征的信息后，更新了`node_features`。

+ node_features: [bs*natom, 160]

## 几何张量分析

这里记录所有网络层的几何张量的输入输出：

### OneHotAtomEncoding - IO

irreps_in: None
irreps_out:
    + **node_attrs: 2x0e**
    + **node_features: 2x0e**

### SphericalHarmonicEdgeAttrs - IO

irreps_in:
    + **pos: 1x1o**
    + **edge_index: None**
    + node_attrs: 2x0e
    + node_features: 2x0e
irreps_out:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 2x0e
    + node_features: 2x0e
    + **edge_attrs: 1x0e+1x1o**

### RadialBasisEdgeEncoding - IO

irreps_in:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 2x0e
    + node_features: 2x0e
    + **edge_attrs: 1x0e+1x1o**

irreps_out:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 2x0e
    + node_features: 2x0e
    + edge_attrs: 1x0e+1x1o
    + **edge_embedding: 8x0e**

### AtomwiseLinear1 - IO

irreps_in:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 2x0e
    + node_features: 2x0e
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

irreps_out:
    + pos: 1x1o
    + edge_index: None
    + **node_attrs: 32x0e**
    + node_features: 2x0e
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

### ConvNetLayer1 - IO

irreps_in:
    + pos: 1x1o
    + edge_index: None
    + **node_attrs: 32x0e**
    + node_features: 2x0e
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

irreps_out:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 32x0e+32x1o**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

#### InteractionBlock1 - IO

irreps_in:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + node_features: 32x0e
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

irreps_out:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 64x0e+32x1o**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

### ConvNetLayer2 - IO

irreps_in:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 32x0e+32x1o**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

irreps_out:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 32x0e+32x1e+32x1o**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

#### InteractionBlock2 - IO

irreps_in:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 32x0e+32x1o**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

irreps_out:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 96x0e+32x1o+32x1e**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

### ConvNetLayer3 - IO

irreps_in:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 32x0e+32x1e+32x1o**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

irreps_out:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 32x0e+32x0o+32x1e+32x1o**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

#### InteractionBlock3 - IO

irreps_in:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 32x0e+32x1e+32x1o**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

irreps_out:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 32x0o+96x0e+32x1o+32x1e**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

### ConvNetLayer4 - IO

irreps_in:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 32x0e+32x0o+32x1e+32x1o**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

irreps_out:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 32x0e+32x0o+32x1e+32x1o**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

#### InteractionBlock4 - IO

irreps_in:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 32x0e+32x0o+32x1e+32x1o**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

irreps_out:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 32x0o+96x0e+32x1o+32x1e**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

### AtomwiseLinear2 - IO

irreps_in:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 32x0o+96x0e+32x1o+32x1e**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

irreps_out:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 16x0e**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

### AtomwiseLinear3 - IO

irreps_in:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + **node_features: 16x0e**
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e

irreps_out:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + node_features: 16x0e
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e
    + **atomic_energy: 1x0e**

### AtomwiseReduce - IO

irreps_in:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + node_features: 16x0e
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e
    + **atomic_energy: 1x0e**

irreps_out:
    + pos: 1x1o
    + edge_index: None
    + node_attrs: 32x0e
    + node_features: 16x0e
    + edge_attrs: 1x0e+1x1o
    + edge_embedding: 8x0e
    + atomic_energy: 1x0e
    + **total_energy: 1x0e**

## ConvNetLayer中几何张量的变化

下面详细展示了几何张量从进入ConvNetLayer后到模型输出的变化，便于理解几何张量在神经网络中是如何变化的：

```bash
self.sc  # data['node_features'] x data['node_attrs']
> FullyConnectedTensorProduct(32x0e x 2x0e -> 64x0e+32x10 409 paths  4096 weights) 
self.linear_1 # data['node_features']
> Linear(32x0e -> 32x0e 1024 weights) 
self.tp #  self.sc x data['node_features'] x data['edge_attrs']
> TensorProduct(32x0e x 1x0e+1x10 -> 32x0e+32x10  64 paths  64 weights) 
self.linear_2 # data['node_features']
> Linear(32x0e+32x10 -> 64x0e+32x10 3072 weights) 
self.equivariant_nonlin # data['node_features']
> Gate (64x0e+32x10 -> 32xe+32x10) 

self.sc # data['node_features'] x data['node_attrs']
> FullyConnectedTensorProduct(32x0e+32x10 x 2x0e -> 96x0e+32x10+32x1e 8192 paths 8192 weights) 
self.linear_1 # data['node_features']
> Linear(32x0e+32x10 -> 32x0e+32x10 2048 weights) 
self.tp # self.sc x data['node_features'] x data['edge_attrs']
  TensorProduct(32x0e+32x10 x 1x0e+1x10 -> 64x0e+64x10+32x1e 160 paths 160 weights) 
self.linear_2 # data['node_features']
  Linear(64x0e+64x10+32xle -> 96x0e+32x10+32xle  9216 weights)
self.equivariant_nonlin # data['node_features']
> Gate (96x0e+32x10+32xle-> 32x0e+32x1e+32x1o)

self.sc # data['node_features'] x data['node_attrs']
> FullyConnectedTensorProduct(32x0e+32x1e+32x10 x 2x0e-> 32x00+96xe+32x10+32xle 10240 paths 1240 weights)> 
self.linear_1 # data['node_features']
  Linear(32x0e+32xle+32x10 -> 32x0e+32xle+32x10 3072 weights)
self.tp # self.sc x data['node_features'] x data['edge_attrs']
  TensorProduct(32x0e+32x1e+32x10 X 1x0e+1x10 -> 32x00+64x0e+96x10+64x1e 256 paths 256 weights)
self.linear_2 # data['node_features']
  Linear(32x00+64x0e+96x10+64x1e -> 32x00+96x0e+32x10+32x1e 12288 weights)
self.equivariant_nonlin # data['node_features']
>Gate (32x00+96x0e+32x10+32x1e -> 32x0e+32x00+32x1e+32x10)

self.sc # data['node_features'] x data['node_attrs']
> FullyConnectedTensorProduct(32x0e+32x0o+32x1e+32x10 x 2x0e-> 32x00+96x0e+32x10+32x1e 12288 paths 12288 weights)
self.linear_1 # data['node_features']
> Linear(32x0e+32x00+32xle+32x10 -> 320e+32x00+32xle+32x10 4096 weights)
self.tp # self.sc x data['node_features'] x data['edge_attrs']
  TensorProduct(32x0e+32x00+32x1e+32x10 X 1x0e+1x10 -> 64X00+4x0e+96x10+96x1e 320 paths 320 weights)
self.linear_2 # data['node_features']
> Linear(64x00+64x0e+96x10+96xle -> 32x00+96x0e+32x10+32xle  14336 weights)
self.equivariant_nonlin # data['node_features']
> Gate (32x00+96x0e+32x10+32x1e -> 32x0e+32x00+32x1e+32x10)
> 
self.linear # data['node_features']
> Linear(32x0e+32x00+32xle+32x10 -> 16x0e 512 weights)
self.linear # data['node_features']
> Linear(16xoe -> 1x0e 16 weights)
```
