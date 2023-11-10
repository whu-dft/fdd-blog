# FermiNet

Ref: [FermiNet paper](https://arxiv.org/abs/1909.02487)

FermiNet是采用DNN来求解多电子薛定谔方程的方法，下面记录下读完paper后的一些总结，主要关注其实现细节。

## 1. Background

简单回顾一下背景知识。不含时间的薛定谔方程：

$$
\hat{H}\psi (\boldsymbol{x}_1,\boldsymbol{x}_2,...,\boldsymbol{x}_n)=E\psi (\boldsymbol{x}_1,\boldsymbol{x}_2,...,\boldsymbol{x}_n) \tag{1}
$$

$$
\hat{H}=-\frac{1}{2}\sum_i^{}{\nabla _{i}^{2}}+\sum_{i>j}^{}{\frac{1}{\left| \boldsymbol{r}_i-\boldsymbol{r}_j \right|}}
\\
\,\,            -\sum_{iI}^{}{\frac{Z_I}{\left| \boldsymbol{r}_i-\boldsymbol{R}_I \right|}+\sum_{I>J}^{}{\frac{Z_IZ_J}{\left| \boldsymbol{R}_I-\boldsymbol{R}_J \right|}}} \tag{2}
$$

上式中，$\boldsymbol{x}_i=\left\{ \boldsymbol{r}_i,\sigma _i \right\}， $小写字母$i$为电子的下标，大小字母$I$表示原子的下标，$r_i$和$R_I$表示表示电子$i$和原子$I$的坐标。此外，$Z_I$表示原子$I$的电荷数，$\nabla _{i}^{2}$为Laplacian算子作用到电子$i$上。注意上述不含时薛定谔方程以采用了Born-Oppenheimer近似，即相对于电子的运动，可以认为原子的运动几乎是静止的。

对于一个给定的体系（分子），体系的哈密顿量就是固定的。因此求解上述静态薛定谔方程的关键在于找到一个合适的波函数$\psi(X)$，使得求解的系统的能量最低。一般来说，我们通过对波函数的各种近似，来逼近系统的真实波函数，这些波函数也称为尝试波函数(wave-function Ansatz)。

理论上来讲，对于波函数只需要满足一个条件-波函数的反对称性，即交换任意两个电子的位置，波函数会多出一个负号:

$$
\psi (...,\boldsymbol{x}_i,...,\boldsymbol{x}_j,...)=-\psi (...,\boldsymbol{x}_j,...,\boldsymbol{x}_i,...) \tag{3}
$$

这是由于电子为费米子（粒子的自旋为1/2的奇数倍）造成的。由于反对称性的要求，人们很自然想起矩阵的性质，交换任意两列，则行列式会多出一个负号。由此变得到所谓的Slater行列式波函数：

$$
\psi_k \left( \boldsymbol{x}_1,\boldsymbol{x}_2,...,\boldsymbol{x}_n \right) =\left[ \begin{matrix}
	\phi _{1}^{k}\left( \boldsymbol{x}_1 \right)&		\phi _{1}^{k}\left( \boldsymbol{x}_2 \right)&		\cdots&		\phi _{1}^{k}\left( \boldsymbol{x}_n \right)\\
	\phi _{2}^{k}\left( \boldsymbol{x}_1 \right)&		\phi _{2}^{k}\left( \boldsymbol{x}_2 \right)&		\cdots&		\phi _{2}^{k}\left( \boldsymbol{x}_n \right)\\
	\vdots&		\vdots&		\ddots&		\vdots\\
	\phi _{n}^{k}\left( \boldsymbol{x}_1 \right)&		\phi _{n}^{k}\left( \boldsymbol{x}_2 \right)&		\cdots&		\phi _{n}^{k}\left( \boldsymbol{x}_n \right)\\
\end{matrix} \right] \tag{4}
$$

其中，$\left\{\phi _{1}^{k}, ...,\phi _{n}^{k}\right\}$表示体系的$n$个轨道(实际轨道数大于等于n)。注意，这里的$\phi_1$理解为体系的第1个轨道，而不是第一个电子的轨道。

实际用来计算的波函数则由若干个行列式的线性组合构成：

$$
\psi \left( \boldsymbol{x}_1,\boldsymbol{x}_2,...,\boldsymbol{x}_n \right) =\sum_k^{}{\omega _k\tilde{\psi}_k\left( \boldsymbol{x}_1,\boldsymbol{x}_2,...,\boldsymbol{x}_n \right)} \tag{5}
$$

我们最终要求解的是多电子体系的基态能，其定义为：

$$
E\left( \theta \right) =\frac{\langle \psi _{\theta}|\left. \hat{H}|\psi _{\theta} \right>}{\langle \psi _{\theta}|\left. \psi _{\theta} \right>} \tag{6}
$$

其中，$\theta$为波函数中待定的参数。当波函数$\psi_\theta$越逼近真实的波函数，求解的基态就越接近真实的能量。

因此，对于FermiNet来说，其核心就是利用神经网络构造一个尝试波函数$\psi_\theta$，然后利用神经网络的反向传播优化$\theta$，使得目标函数$E(\theta)$的值达到最小值。

## 2. The network structure

作为一种第一性原理计算方法，FermiNet的输入只需要包含分子的基本信息，比如原子的坐标，原子的电荷，自旋向上和向下的电子数分别多少。我们假定体系的总电子数为$N$，总原子数为$M$。根据公式(2)，FermiNet充分利用了这些基本信息，其模型的输入分为两部分：

单电子部分主要是指电子与原子作用的部分，只依赖于电子和原子的坐标，第$i$个电子表示为$h_i$:

$$
h_i=\left\{ \boldsymbol{r}_i-\boldsymbol{R}_1,\left| \boldsymbol{r}_i-\boldsymbol{R}_1 \right|,\boldsymbol{r}_i-\boldsymbol{R}_2,\left| \boldsymbol{r}_i-\boldsymbol{R}_2 \right|,...,\boldsymbol{r}_i-\boldsymbol{R}_M,\left| \boldsymbol{r}_i-\boldsymbol{R}_M \right| \right\} \tag{7}
$$

这样，每个电子的表示的长度为$M\times(3+1)$。所有单电子表示的长度为$N\times M\times(3+1)$，这里的N包含了自旋向上和向下的电子。

双电子部分主要是指电子间的相互作用，只依赖两个电子的坐标，第$i$和$j$对电子的表示为$h_{ij}$:

$$
h_{ij}=\left\{ \boldsymbol{r}_i-\boldsymbol{r}_j,\boldsymbol{r}_{ij} \right\} \tag{8}
$$

$r_{ij}$表示电子$i$与电子$j$的距离。所有电子的相互作用元素个数为$(3+1)*(N-1)^2$。注意，这个计算扣除了电子与自己相互作用的项！

由此，可以得到FermiNet的input为$\left\{...,h^l_i,...\right\}$和$\left\{...,h^l_{ij},...\right\}$，这里的$k$表示第$l$个layer，见FermiNet paper中的FIG. 1。

### 2.1 Input and output of each layer

上面解释了模型输入的单电子项与双电子项中，忽视了电子自旋向上和向下的区分。考虑自旋的时候，第$l$层的单电子项可表达为$h^{l\alpha}_i$,双电子项表达为$h^{l\alpha\beta}_{ij}$,其中$\alpha ,\beta \in \left\{ \uparrow ,\downarrow \right\}$。下面分析FermiNet在每个layer中，对输入和输出的处理：

1. 对原始输入的单电子项($h^{l\uparrow}_i$, $h^{l\downarrow}_i$)做平均，区分自旋向上和向下，得到两个向量，其长度分别为$N_\uparrow\times(3+1)$和$N_\downarrow\times(3+1)$
	$$
	\begin{cases}
	g^{l\uparrow}=\frac{1}{n\uparrow}\sum_i^{n\uparrow}{h_{i}^{l\uparrow}}\\
	g^{l\downarrow}=\frac{1}{n\downarrow}\sum_i^{n\downarrow}{h_{i}^{l\downarrow}}\\
	\end{cases} \tag{9}
	$$

2. 对于电子$i$和自旋$\alpha$
    (1) 计算双电子项($h^{l\alpha\beta}_{ij}$)的平均，区分自旋向上和向下，得到两个向量，其长度都是$(3+1)$
	$$
	\begin{cases}
		g_{i}^{l\alpha \uparrow}=\frac{1}{n\uparrow}\sum_j^{n\uparrow}{h_{ij}^{l\alpha \uparrow}}\\
		g_{i}^{l\alpha \downarrow}=\frac{1}{n\downarrow}\sum_i^{n\downarrow}{h_{ij}^{l\alpha \downarrow}}\\
	\end{cases} \tag{10}
	$$
	(2) 将前面计算的各个物理量合并成一个向量，其长度为$M\times(3+1)+N\times(3+1)+2*(3+1)$
	 $$
	f_{i}^{l\alpha}=\left[ h_{i}^{l\alpha},g^{l\uparrow},g^{l\downarrow},g_{i}^{l\alpha \uparrow},g_{i}^{l\alpha \downarrow} \right] \tag{11}
	$$

3. 计算当前layer输出的单电子项和双电子项，激活函数为tanh，并且采用了类似resnet的结构
	$$
	\begin{cases}
		h_{i}^{\left( l+1 \right) \alpha}=\tanh \left( \boldsymbol{V}^lf_{i}^{l\alpha}+\boldsymbol{b}^l \right) +h_{i}^{l\alpha}\\
		h_{ij}^{\left( l+1 \right) \alpha}=\tanh \left( \boldsymbol{W}^lh_{ij}^{l\alpha \beta}+\boldsymbol{c}^l \right) +h_{ij}^{l\alpha \beta}\\
	\end{cases} \tag{12}
	$$

注意：在FermiNet中，后一层的单电子项是聚集了前一层的单电子层和双电子项的信息，而双电子双仅仅包含前一层的双电子项信息（见公式(12)）。

PS： 从公式(12)来看，好像激活函数的输出维度跟$h_{i}^{l\alpha}$不匹配？？？

FermiNet网络结构的宏观视图:

![Structure of FermiNet](./pics/Structure_of_FermiNet.png)

FermiNet的微观构造流程：

![Algorithms of FermiNet](./pics/Algorithms_of_FermiNet.png)

### 2.2 Construction of single electron orbital

通过2.1，我们知道整个FermiNet的结构。接下来分析FermiNet如何利用网络的输出来构造单电子轨道$\phi _{i}^{k\alpha}\left( \boldsymbol{r}_{j}^{\alpha};\left\{ \boldsymbol{r}_{/j}^{\alpha} \right\} \right)$。有了单电子轨道，就可以进一步构造Slater行列式了。

$$
\begin{cases}
	\phi _{i}^{k\alpha}\left( \boldsymbol{r}_{j}^{\alpha};\left\{ \boldsymbol{r}_{/j}^{\alpha} \right\} \right) =\left( M_{i}^{k\alpha}\cdot \boldsymbol{h}_{j}^{L\alpha}+g_{i}^{k\alpha} \right) \cdot \boldsymbol{e}_{ij}^{k\alpha}\\
	\boldsymbol{e}_{ij}^{k\alpha}=\sum_m^{}{\pi _{im}^{k\alpha}\exp \left( -\left| \varSigma _{im}^{k\alpha}\left( r_{j}^{\alpha}-R_m \right) \right| \right)}\\
\end{cases} \tag{13}
$$

可以看到，单电子波函数是有FermiNet网络输出层中的单电子项通过线性组合得到的。后面乘以了一个指数衰减因子$\boldsymbol{e}_{ij}^{k\alpha}$，是为了满足电子波函数的边界条件，即在无穷远处的值为0. 其中$\pi _{im}^{k\alpha}$和$\varSigma _{im}^{k\alpha}$分别是两个$3\times3$矩阵，控制单电子波函数在不同的方向衰减到0.

这里计算的$\phi _{i}^{k\alpha}\left( \boldsymbol{r}_{j}^{\alpha};\left\{ \boldsymbol{r}_{/j}^{\alpha} \right\} \right)$就是公式(4)中，构成slater行列式中的单电子波函数$\psi_i(\boldsymbol{x}_j)$。其中的$k$表示第$k$个行列式，$\alpha$表示不同的自旋。

### 2.3 Construction of the whole wavefunction

在2.2中，我们知道了如何构造单电子功函数，那么采用slater行列式，我们可以立即得到整个尝试波函数：

$$
\begin{cases}
	\tilde{\psi}^{k\alpha}\left( \boldsymbol{x}_1,\boldsymbol{x}_2,...,\boldsymbol{x}_n \right) =\left[ \begin{matrix}
	\phi _{1}^{k\alpha}\left( \boldsymbol{x}_1 \right)&		\phi _{1}^{k\alpha}\left( \boldsymbol{x}_2 \right)&		\cdots&		\phi _{1}^{k\alpha}\left( \boldsymbol{x}_n \right)\\
	\phi _{2}^{k\alpha}\left( \boldsymbol{x}_1 \right)&		\phi _{2}^{k\alpha}\left( \boldsymbol{x}_2 \right)&		\cdots&		\phi _{2}^{k\alpha}\left( \boldsymbol{x}_n \right)\\
	\vdots&		\vdots&		\ddots&		\vdots\\
	\phi _{n}^{k\alpha}\left( \boldsymbol{x}_1 \right)&		\phi _{n}^{k\alpha}\left( \boldsymbol{x}_2 \right)&		\cdots&		\phi _{n}^{k\alpha}\left( \boldsymbol{x}_n \right)\\
\end{matrix} \right]\\
	\psi \left( \boldsymbol{x}_1,\boldsymbol{x}_2,...,\boldsymbol{x}_n \right) =\sum_k^{}{\omega _k\tilde{\psi}^{k\uparrow}\left( \boldsymbol{x}_1,\boldsymbol{x}_2,...,\boldsymbol{x}_n \right) \tilde{\psi}^{k\downarrow}\left( \boldsymbol{x}_1,\boldsymbol{x}_2,...,\boldsymbol{x}_n \right)}\\
\end{cases}  \tag{14}
$$

这里需要注意几点：

1. 整个波函数是有若干个行列式的线性组合得到，行列式的个数是一个参数，可供调试；
2. 前面构造单电子波函数时候，区分了自旋向上和向下，在这里，自旋向上的行列式与自旋向下的行列式直接相乘得到整个的波函数；
3. 如何构造的波函数只有在交换相同自旋的电子时，波函数才会是反对称性的；
4. 给定个电子原子的坐标，对一个样本，波函数就是一个数值，其平方表示各电子在给定坐标位置的概率；

这里稍微有点奇怪的是，在整个FermiNet中，并没有出现虚数，而虚数在物理里面是十分常见的。在物理里，波函数就是一个虚数。

## 3. Wave-function optimization

我们最终的目标loss函数为系统的基态能量，按照如下物理公式计算：

$$
L\left( \theta \right) =\frac{\langle \psi _{\theta}|\hat{H}|\psi _{\theta}\rangle}{\langle \psi _{\theta}|\psi _{\theta}\rangle} \tag{15}
$$

其中，$\theta$为FermiNet中的待定参数，优化目标即使$L(\theta)$最小。

根据Paper中的公式，损失函数$L(\theta)$对参数$\theta$的导数为（没搞清楚咋来的。。。）:

$$
\frac{\partial L\left( \theta \right)}{\partial \theta}=E_{p\left( \boldsymbol{X} \right)}\left[ \left( E_L-E_{p\left( \boldsymbol{X} \right)}\left[ E_L \right] \right) \right] \nabla _{\theta}\log \left( \left| \psi \right| \right) \tag{16}
$$

注意，上式中的$p(\boldsymbol{X})$是波函数的平方，表示概率。$p\left( \boldsymbol{X} \right) \sim \psi _{\theta}^{2}\left( \boldsymbol{X} \right)$

其中公式(16)中的$E_L$称为local energy，我猜是一个样本对应的能量，表达式见公式(17)，能量的期望为电子坐标铺满整个空间后的local energy的期望。Paper中还说，网络的输出为波函数的对数(取log，虽然不知道为啥这样纸-_-!)，所以local energy $E_L$也需要转换到$log(\psi)$的形式。

$$\begin{aligned}
E_L\left( \boldsymbol{X} \right) &=\psi ^{-1}\left( \boldsymbol{X} \right) \hat{H}\psi \left( \boldsymbol{X} \right) 
\\
&=-\frac{1}{2}\sum_i^{}{\left[ \left. \frac{\partial ^2\log \left( \left| \psi \right| \right)}{\partial r_{i}^{2}} \right|_{\boldsymbol{X}}+\left( \left. \frac{\partial \log \left( \left| \psi \right| \right)}{\partial r_i} \right|_{\boldsymbol{X}} \right) ^2 \right]}+V\left( \boldsymbol{X} \right) 
\end{aligned} \tag{17}
$$

主要注意一下，由于在计算能量的过程中，对波函数取了对数，但是波函数也有可能是负数。因此，FermiNet的输出还需要记录波函数的正负号(代码中可见)。

## 4. Summary

FermiNet的核心是构造基于slater行列式的波函数，其目的是构成出满足物理反对称性的波函数。整个网络都在构造单电子轨道，但是在构造过程中利用了双电子的信息。此外，还利用到一个物理性质是，波函数的边界条件，即在无穷远处，波函数会衰减到0。神奇的是，其计算出来的分子的基态非常精确！
