

# 第二章 线性代数

## 矩阵

### 矩阵加法

按位相加 A+B, 要求A和B形状相同

 

### 矩阵乘法

A(m,n)和B(n, k)的乘法计算规则为 结果C(m, k)的每一项c_i,j是A的i行向量和B的j列向量的内积

表示方法为A·B (其中·需要显示表示)

 

### 矩阵按位乘法

上述的矩阵乘法并不是对矩阵的逐元素乘法, 逐元素乘法被称为Hadamard积

它的表示法为 A∘B

 

### 矩阵运算性质

- 满足结合律
- 满足分配律
- 与单位矩阵相乘等于本身
- 不满足交换律

 

### 矩阵的逆

对于一个方阵A, 如果存在一个方阵B, 使得AB = BA = I, 则B是A的逆矩阵. B可被记作A^-1

并不是所有矩阵都存在逆矩阵. 如果存在, 这个矩阵称为可逆矩阵/非奇异矩阵, 不存在, 这个矩阵可称为不可逆矩阵/奇异矩阵

如果一个矩阵存在逆, 则该逆矩阵必然唯一

 

#### 快速有逆判断(2维)

如果a_11 *a_22 - a_12 * a_21不为0, 则该2维矩阵有逆

 

### 矩阵的转置

对于矩阵A, 满足b_ij = a_ji的矩阵B被称作A的转置, 记作B = A^T

 

### 有关逆和转置的重要性质:

- AA^-1 = A^-1A = I
- (AB)^-1 = B^-1A^-1
- (A^T)^T = A
- (A + B)^T = A^T + B^T
- (AB)^T = B^T A^T
- (A+B)^-1 ≠ A^-1 + B^-1

 

### 对称矩阵

如果矩阵A满足 A = A^T, 则称其为对称矩阵

对称矩阵一定是方阵

 

### 矩阵的标量乘

很自由

满足各种结合律, 分配律和转置

 

## 解线性方程组

拆分为列向量的加权之和

Ax = B => [a_0, a_1 .... a_n] [x_0 ... x_n]^T = [b_0 ... b_n], 其中a_i为列向量, 第i列和第i个x的乘积之和是最后的b

 

可以使用高斯消元法, 将线性方程组简化为方便上述处理的形式

 

### 初等变换

> 其实无所谓, 一般AI应用不需要手解线性方程组, 不仔细看了

解线性方程组的关键是初等变换, 这些变换可以保持方程组的解集不变, 但可以将方程组转换为更简单的形式:

- 交换两个方程(矩阵中的行)
- 将一个方程(行)乘以一个非0常数
- 将两个方程(行)相加

 

### 计算逆矩阵

为了计算A^-1, 我们需要找到一个矩阵X, 使得AX = I, 则X = A^-1.

可以将此写成一组同时线性方程AX = I, 其中我们的解X = [x1]....[xn]

我们使用增广矩阵(就是把线性方程组中的所有数写成一个矩阵, 等号左边的系数矩阵放在左边, 等号右边)

例题算出来了

 

### 解线性方程组的算法

对于较弱的条件下(即A需要具有线性独立的列), 可以使用变换,将求A的逆转换为求(A^TA)^-1A^T, 使用Moore-Penrose.

高斯消元法比较直观, 在矩阵维度很大的情况下, 这么算不切实际, 在实践中一般采用间接方式计算, 比如Richardson方法, jacobi方法, Gauss-Seidel方法和逐次超松弛迭代法或Krylov子空间方法, 如共轭梯度法, 广义最小残差法或双共轭梯度法.

 

## 线性空间

 

### 群

群在计算机科学中扮演着重要角色. 除了为集合上的操作提供基本框架外, 它们还广泛应用于密码学, 编码理论和图形学.

 

#### 群的定义

考虑一个集合$$\mathcal{G}$$和一个在G上定义的二元运算$$\otimes$$ : $$\mathcal{G}xmathcal{G} \to \mathcal{G} $$  G := ($$\mathcal{G}$$, $$\otimes$$ )被称为一个群

如果满足以下条件:

1. 关于$$\otimes$$ 的封闭性: $$\forall$$x, y $$\in \mathcal{G}: x\otimes y \in \mathcal{G} $$
2. 结合律: $$ \forall x, y, z \in \mathcal{G}: (x\otimes y)\otimes z = x \otimes (y \otimes z)$$
3. 单位元: $$\exist e \in \mathcal{G}, \forall x \in \mathcal{G}: x\otimes e =2.718 x 且 e \otimes x = x $$
4. 逆元: $$\forall x \in \mathcal{G} \exist y \in \mathcal{G}: x\otimes y  = e 且 y\otimes x = e$$

如果此外 $$\forall x, y \in \mathcal{G}: x \otimes y = y \otimes x, 则G = (\mathcal {G}, \otimes)$$是一个Abel群(交换群).



#### 一般线性群

可逆矩阵A $$\in \mathbb{R}^{nxn} $$的集合关于矩阵乘法的定义称为一般线性群, 记作 $$GL(n, \mathbb{R})$$.

但由于矩阵乘法不满足交换律, 因此该群不是Abel群.



### 线性空间

在讨论群时, 我们研究了集合$$\mathcal{G}$$ 以及其上的内操作. 接下来, 我们将考虑包含内操作“+”和外操作“·”(标量乘法)的集合

> 内操作指集合内元素间的运算, 外操作指集合内元素和集合外元素的运算?

#### 线性空间的定义

一个实值线性空间V是这些资料$$(V, \mathbb{R}, +, ·)$$ V上具有两个运算:
$$
+: \mathcal{V} \times \mathcal {V } \to \mathcal{V} \\
\cdot : \mathbb{R} \times \mathcal{V} \to \mathcal{V}
$$


其中:

1. $$(\mathcal{V}, +) $$是Abel群
2. 满足分配律
3. 标量乘法满足结合律
4. 标量乘法的单位元是1

集合中的元素x称为向量, 其中的单位元称为零向量, 记作0. 



**接下来会用V表示线性空间, 当说明 $x \in V$时, 表示x是V中的向量**



### 向量子空间

直观上, 向量子空间是包含在原始线性空间内的集合, 具有这样的性质: 当我们对子空间内的元素进行线性空间操作时, 结果总是落在子空间上

**向量子空间时机器学习中的一个关键概念, 例如, 第10章将展示如何使用向量子空间进行降维**



#### 向量子空间的定义

$设V = (V, +, \cdot)$是一个线性空间, $且U\subseteq V , U \neq \Phi$, 那么 $U = (U, +, \cdot)$称为V的向量子空间(或线性子空间),

- 齐次线性方程组Ax = 0 的解集是一个子空间 
  - (因为解集没有引入特解的向量, 用参数向量可以表示所有解吧)
- 非齐次线性方程组Ax = b的解集不是一个子空间(但是升一维可以变成高维的齐次线性方程组, 也就可以是$$\mathbb{R}^{n+1}$$的子空间)
- 任意多个子空间的交集仍是一个子空间



## 线性无关

这样一组向量: 通过它们间的加法和缩放, 可以表示线性空间中的每一个向量, 这组向量称为基.



### 线性无关定义

设V是一个线性空间, 且x1....xk 属于V, 如果存在一个非平凡的线性组合使得
$$
0 = \sum_{i=1}^{k}\lambda_i x_i
$$
且其中至少有一个$$\lambda_i \neq 0$$, 则这组向量是线性相关的, 如果只有平凡解, 即每个$$\lambda$$都是0, 则这组向量是线性无关的.

一组线性无关的向量是由没有冗余的向量组成, 即如果我们从集合中移除任何一个向量, 都会失去某些信息.



#### 用于帮助判断向量是否线性无关的性质:

- k个向量要么线性相关, 要么线性无关, 没有第三种可能
- 如果向量中至少有一个零向量, 则它们是线性相关的
- 如果两个向量相同, 也是线性相关的
- 根据定义, 可以通过高斯消元法, 如果存在非主元列, 则一定是线性相关的

## 向量组的基和秩

在线性空间V中, 我们特别感兴趣的是哪些能够通过线性组合生成线性空间中每一个向量的向量集合.



### 生成集合基

#### 生成集合张成空间的定义

考虑一个线性空间$$V = (V, +, \cdot)$$以及一组向量$$A = {x_1, x_2...x_k} \subseteq V$$. 如果V中的每个向量v都可以表示为A中向量的线性组合, 那么A就是V的一个生成集. 所有A中向量的线性组合的集合称作A的张成空间, 记作span[A]或者span[x1...xk]. 如果A张成了线性空间V, 我们写作V = span[A]



生成集就是能够张成向量(子)空间的向量集合, 即每个向量都可以表示为生成集中向量的线性组合.



#### 基的定义

考虑一个线性空间$$V = (V, +, \cdot)$$和$$A \subseteq V$$ . 如果一个生成集A是最小的, 即不存在更小的集合$$\tilde{A} \subset A \subseteq V$$能够张成V, 则称A是V的一个基. 每个线性无关的生成集都是最小的, 因此称为V的一个基.

设$$V = (V, +, \cdot)$$是一个线性空间, $$B \subseteq V, B\neq \Phi$$, 那么以下陈述是等价的:

- B是V的一个基
- B是一个最小生成集
- B是V中的一个最大的线性无关向量集合, 即在B中添加任何其他向量都会使其线性相关
- V中的每个向量x都可以表示为B中向量的线性组合, 且每个线性组合都是唯一的

在只考虑有限维线性空间时, V的维度就是V的基向量的数量, 记作dim(V), 如果$$U \subseteq V$$是V的一个子空间, 那么dim(U)$$\le$$ dim(V), 且dim(U) = dim(V)当且仅当U=V, 置关上, 线性空间的维数可以看作是线性空间中独立方向的数量.

### 秩

矩阵$$A \in \mathbb{R}^{m \times n}$$的线性无关列的数量等于线性无关行的数量, 称为A的秩, 记作rank(A)

> 所以秩有什么用?
>
> 是否所有矩阵都有秩? 或者说是否所有矩阵的列秩都等于行秩?

#### 秩有一些重要的性质

-  Rank(A)=rank(A^T)即列秩等于行秩(线性无关行的数量)
- A的列张成一个子空间$$U \subseteq \mathbb{R}^m$$, 其维数是rank(A). 我们稍后将称这个子空间为🐘或者值域.可以通过对A应用高斯消元法来找到U的一个基, 确定主元列
- A的行张成一个子空间$$W \subseteq \mathbb{R}^n$$ , 其维数也是rank(A), 可以通过对A^T应用高斯消元法找到W的一个基
- 对于所有的$$A \in \mathbb{R}^{n\times n}$$, A是可逆的当且仅当rank(A) = n, (啊? 不线性无关就不可逆啊)
- 对所有的$$A \in \mathbb{R}^{m\times n}$$和所有的$$b \in \mathbb{R}^m$$, 线性方程组Ax = b有解当且仅当rank(A) = ran k([A|b]), (没看懂这条, A可逆不就一定有解吗)
- 对于$$A \in \mathbb{R}^{m\times n}$$, 齐次方程组Ax=0的解空间的维数是n - rank(A). 这个子空间被称为核或者零空间, 核的维数为n-rank(A) (从直觉角度挺好理解的)
- 一个矩阵$$A \in \mathbb{R}^{m \times n}$$具有满秩, 指它的秩等于对于相同维度的矩阵可能的最大秩. 这意味着满秩矩阵的秩是行数和列数中较小的那个, 如果一个矩阵的秩不等于满秩, 则称该矩阵是秩亏的.



## 线性映射

接下来, 我们将研究线性空间之间的映射, 这些映射保持线性空间的结构, 从而可以定义坐标的概念.

#### 线性映射的定义

对于线性空间V, W, 一个映射 $$\Phi: V \to W$$ 称为线性映射(或线性空间同态/线性变换), 当且仅当:
$$
\forall x, y \in V, \forall \lambda, \phi \in \mathbb{R}: \Phi(\lambda x + \phi y) = \lambda \Phi(x) + \phi \Phi(y)
$$
我们可以用矩阵来表示线性映射.

#### 单射, 满射, 双射的定义

考虑一个映射 $$\Phi : V \to W$$, 其中V, W可以是任意集合, 那么$$\Phi$$称为:

- 单射(injective): 如果$$\forall x, y\in V: \Phi(x) = \Phi(y) \Rightarrow x = y$$ 就是说两个向量不能映射到一个向量上
- 满射(surjective): 如果$$\Phi(V) = W$$ W的所有都可以被V映射
- 双射(Bijective): 如果它既是满射也是单射

如果是满射, 那么W中的每个元素都可以通过这个映射从V的某个元素“到达”.

如果是双射, 那么存在一个映射$$\Psi: W \to V$$, 使得$$\Psi \circ \Phi(x) = x$$. 这个映射\Psi 被称作\Phi的逆映射, 通常记作$$\Phi^{-1}$$. 根据这些定义, 我们引入以下线性映射之间的特殊情况:

- 同构(Isomorphism): $$\Phi: V \to W$$线性且双射
- 自同态(Endomorphism): $$\Phi: V \to V$$线性. 线性空间V的所有自同态形成一个集合, 记作End(V) (9命, 自同态不是映射嘛, 怎么形成集合了)
- 自同构(Automorphism): $$\Phi: V \to V$$线性且双射, 线性空间V的所有自同构形成一个集合, 记作Au t(V)
- 恒等映射(Identity Automorphism): $$id_V: V \to V, x \mapsto x$$是V中的恒等映射

#### Axl er 2015 定理2.17

有限维线性空间V和W是同构的, 当且仅当dim(V) = dim(W)

这个定义表明, 如果两个线性空间的维数相同, 则它们之间存在一个线性双射映射.

直观上, 这意味着维数相同的线性空间在某种意义上是相同的, 因为它们可以互相转换而不会有任何损失



#### 连续映射的一些性质

考虑线性空间V, W, X

- 对于线性映射$$\Phi V \to W 和 \Psi W \to X$$, 映射$$\Psi \circ \Phi: V \to X$$也是线性的
- 如果V到W的映射同构, 那么逆映射W到V也是同构
- 如果$$\Phi V \to W \Psi W \to X$$是线性的, 那么$$\Phi + \Psi 和\lambda \Phi$$也是线性的

### 线性映射的矩阵表示

根据定理2.17, 任何n维线性空间都与$$\mathbb{R}^n$$同构. 

我们考虑一个n维线性空间V的一个基{b1...bn}, 接下来有序基的顺序很重要. 因此, 我们写作

> 哦, {}表示无序列表, ()表示有序元组

$$
B = (b_1\dots b_n)
$$

####  坐标的定义(2.18)

考虑一个线性空间V和V的一个有序基B, 对于任意x属于V, 我们得到一个唯一表示(线性组合)
$$
x = \alpha_1b_1 + \dots + \alpha_nb_n
$$
那么$$ a_1, \dots a_n$$称为x关于B的坐标, 可以写作向量形式.这个向量称作x关于有序基B的坐标向量或坐标表示.

基有效定义了一个坐标系.

#### 变换矩阵的定义(2.19)

简单说就是V到W的映射, V的基为B, W的基为C, 把B写作C的坐标表示, 组成的矩阵就是变换矩阵.



### 基变换

接下来, 我们将更仔细的研究当我们在V和W中改变基时, 线性映射$$\Phi : V \to W$$的变换矩阵如何变化, 考虑V的两个有序基
$$
B = (b_1,\dots,b_n), \tilde{B} = (\tilde{b_1}, \dots, \tilde{b_n})
$$


以及W的两个有序基C和$$\tilde{C}$$

此外, $$A_{\phi} \in \mathbb{R}^{m\times n}$$ 是线性映射V到W关于基B和C的变换矩阵, 而$$\tilde{A_{\Phi}}$$是关于$$\tilde{B}和\tilde{C}$$的相应变换矩阵.

在接下来的内容中, 如果我们选择从B, C到tilde B, tilde C进行基变换, 我们将研究A和tilde A的关系



> Ps : 在例2.23中, 从规范基变为B是说V和W都从规范基变成B了, 不是只变一个

这个关系挺好理解的, A是B到C, tilde A是tilde B到tilde C, B到C也等于tilde B到B到C到tilde C, 所以计算B和tilde B以及C和tilde C之间的变换矩阵就行.

#### 基变换定理(2.20)

对于线性映射$$\Phi: V \to W$$, V的有序基 $$B, \tilde{B}$$, W的有序基$$C, \tilde{C}$$

并且关于B到C的变换矩阵是$$A_{\Phi}$$, 关于$\tilde{B}$到$\tilde{C}$的变换矩阵是$\tilde{A_{\Phi}}$ 

S是恒等映射$$id_V$$的变换矩阵, 将$\tilde{B}$映射为B,

T将$\tilde{C}$ 映射为C,

那么两个变换矩阵之间的关系为:
$$
\tilde{A_{\Phi}} = T^{-1}A_{\Phi}S
$$

#### 等价的定义(2.21)

如果存在可逆矩阵S和T, 使得
$$
\tilde{A} = T^{-1}AS
$$
那么$\tilde{A}$和A等价

#### 相似的定义(2.22)

如果存在一个可逆矩阵S, 使得
$$
\tilde{A} = S^{-1}AS
$$
那么$\tilde{A}$和A相似

> 相似矩阵总是等价的, 但等价矩阵不一定是相似的

### 像和核

线性映射的像和核是具有某些重要性质的向量子空间. 接下来, 我们将更仔细的描述它们.

#### 像和核的定义(2.23)

对于$\Phi: V \to W$, 我们定义
$$
Ker(\Phi):= \Phi^{-1}(0_W) = {v \in V: \Phi(v) = 0_W}
$$
为$\Phi$的核(kernel)或零空间(null space)

> 简单来说就是V中经过映射, 映射到W的单位元的向量的集合

$$
Im(\Phi):= \Phi(V) = {w \in W: \exist v \in V, \Phi(v) = w}
$$

为$\Phi$的像(image)或值域(ran ge, co-domain)

> 简单来说就是能被V中向量映射到的W中的向量的集合

另外, 我们称V是映射的定义域, W是映射的值域.



> 根据定义, 映射的核是V的子空间, 映射的像是W的子空间

#### 秩-零度定理

对于线性空间V, W和线性映射$\Phi: V \to W$, 我们有:
$$
dim(ker(\Phi)) + dim(Im(\Phi)) = dim(V)
$$
秩-零度定理也被称为线性映射的基本定理(axle r, 2015,定理3.22)

## 仿射空间

接下来, 我们将更加仔细的研究从原点偏移的空间, 即不再是向量子空间的空间.

此外, 我们将简要讨论这些仿射空间之间的映射的性质, 这些映射类似于线性映射



### 仿射子空间

#### 仿射子空间的定义(2.25)

设V是一个线性空间, $x_0 \in V$ 且 $U\subseteq V$是一个子空间, 那么子集
$$
L = x_0 + U := {x_0 + x, x \in U} \\
= {v \in V: \exist u \in U, x_0 + u = v} \subseteq V
$$
称为V的一个仿射子空间或线性流形. U称为方向或仿射子空间的方向空间, 仿射子空间不是V的一个(线性)子空间.

仿射子空间的例子是$\mathbb{R}^3$中的点, 线和平面中的点的集合, 它们不一定通过原点.

在$\mathbb{R}^n$ 中, (n-1)-维仿射子空间称为超平面.



#### 非齐次线性方程组和仿射子空间

对于$A \in \mathbb{R}^{m\times n}和x \in \mathbb{R}^m$, 线性方程组$A\lambda =x$的解要么是空集, 要么是$R^n$的一个n-rk(A)维的仿射子空间

> 其他的没太看懂

### 仿射映射

#### 仿射空间的定义(2.26)

对于两个线性空间V和W, 线性映射$\Phi: V \to W$, 以及$\alpha \in W$, 可以定义V到W的仿射映射
$$
\phi: V \to W
x \Rightarrow \alpha + \Phi(x)
$$
其中向量$\alpha$称为$\phi$的位移向量

- 每个仿射映射都可以唯一的写为一个线性映射和一个位移的复合, 也即$\phi = \tau\circ\Phi$
- 两个仿射映射的复合还是仿射映射
- 仿射映射保持几何结构(如维数和平行的性质)



## 多重线性代数和张量

> 好多内容, 看起来好高深, 跳了, 真的需要再看

# 第三章 解析几何

## 范数

当我们考虑几何意义的向量, 也就是从原点出发的有向线段时, 其长度显然是原点到有向线段终点之间的直线距离.

下面我们将用范数的概念讨论向量的长度.

#### 范数的定义(3.1)

一个范数是线性空间V上的一个函数:
$$
\parallel \cdot \parallel : V \to \mathbb{R}\\
x \mapsto \parallel x\parallel
$$
它给出每个线性空间中每个向量x的实值长度, 使得任意的$x, y\in V, \lambda \in \mathbb{R}$满足以下条件:

- 绝对一次齐次 $\parallel \lambda x\parallel =\parallel \lambda \parallel \parallel x\parallel$
- 三角不等式$\parallel x+y \parallel \le \parallel x\parallel + \parallel y\parallel$
- 半正定 $\parallel x \parallel \ge 0$, 当且仅当x = 0时取等

#### 曼哈顿范数

$\mathbb{R}^n$上的曼哈顿范数(又叫$l_1$范数), 的定义如下:
$$
\parallel x \parallel_1:= \sum_{i=1}^{n}\mid x_i\mid
$$

#### Euclid范数

欧几里得范数又叫$l_2$范数
$$
\parallel x\parallel_2 := \sqrt{\sum x_i^2} = \sqrt{x^Tx}
$$
在本书中, 若不指明, 范数一般指欧几里得范数



## 内积

### 点积

一些特殊形式的点积我们很熟悉了, 比如标量积或者$\mathbb{R}^{n}$中的点积, 由下面的式子给出:
$$
x^Ty = \sum_{i=1}^nx_iy_i
$$
在本书中, 我们称这样的内积形式为点积. 我们会介绍更一般的内积, 只需满足一些条件即可



### 一般的点积

我们可以根据线性映射的性质对加法和标量乘法进行重排.

一个V上的双线性映射$\Omega$接受两个参数, 并对其中的任意一个参数保持线性, 任取$x,y,z\in \Omega, \lambda, \phi\in \mathbb{R}$, 有:
$$
\Omega(\lambda x + \phi y, z) = \lambda\cdot\Omega(x,z) + \phi\cdot\Omega(y,z)\\
\Omega(x, \lambda y + \phi z) = \lambda\cdot\Omega(x,y) + \phi\cdot\Omega(x,z)
$$
根据线性映射的定义, 第一个式子表示函数对第一个变量线性; 第二个式子表示函数对第二个变量线性, 双线性可以同时满足两个变量的线性

#### 双线性映射对称和正定的定义(3.2)

设V为线性空间, 双线性映射$\Omega: V\times V \to \mathbb{R}$将两个V中的向量映射到一个实数, 则

- 若对于所有V中的x,y, 都有$\Omega(x, y) = \Omega{y, x}$, 也即两个变量可以调换顺序, 则称$\Omega$是对称的
- 如果对于所有的V中x, 都有 $\forall x \in V\setminus \{0\}: \Omega(x, x) > 0, \Omega(0, 0) = 0$, 则称$\Omega$是正定的

#### 内积的定义(3.3)

设V为线性空间, 双线性映射$\Omega: V \times V \to \mathbb{R}$将两个V中向量映射为一个实数, 则

- 对称且正定的双线性映射叫做V上的一个内积, 并简写$\Omega(x, y)为<x, y>$
- 二元组($V, <\cdot, \cdot>$)称为内积空间或装配有内积的(实)线性空间, 特别的, 如果内积采用(103)中定义的内积, 则称这个为Euclid线性空间, 简称欧式空间.



### 对称和正定矩阵

对称和正定矩阵在机器学习中十分重要, 它们由内积定义.

假设n维线性空间V装配有内积并取V中的一个有序基B, 则V中的任意x, y可以表示为B中向量的加权组合 

比如 $ x = \sum_{i=1}^n\phi_i b_i$ , 由于内积具有双线性的性质, 所以它俩的内积可以写作
$$
<x, y> = <\sum_{i=1}^n\phi_ib_i, \sum_{j=1}^n\lambda_jb_j> = \sum_{i=1}^n\sum_{j=1}^n\phi_i\lambda_j<b_i,b_j> = \hat{x}^TA\hat{y}\\
其中\hat{x}, \hat{y}为x,y在B下的坐标表示, A_{i,j}:=<b_i,b_j>
$$
这意味着内积被矩阵A唯一确定, 且由于内积具有对称性, 所以A是对称矩阵, 进一步地, 根据内积的正定性, 我们可以得出下面结论:
$$
\forall x \in V \setminus \{0\}: x^TAx >0
$$

#### 对称正定矩阵的定义(3.4)

一个n级对称矩阵$A \in \mathbb{R}^{n\times n}$ 如果满足(147), 则叫做对称正定矩阵(或仅称为正定矩阵), 如果只满足(147)中大于号变成大于等于号, 则称为对称半正定矩阵

如果一个A是对称正定矩阵, 则它可以定义一个在基B下的内积:
$$
<x, y> = \hat{x}^TA\hat{y}
$$

#### 内积的存在定理(3.5)

对于有限维实线性空间V和它的一个有序基B, 双线性函数是其上的一个内积当且仅当存在一个对称正定矩阵A与之对应, 即
$$
<x, y> = \hat{x}^TA\hat{y}
$$

#### 两个对称正定矩阵的性质

- 矩阵A的所有对角元都是正数
- 矩阵A的零空间只包含零向量

## 向量的长度和距离

在之前我们讨论过计算向量的长度需要使用范数, 内积和范数这两个概念紧密相连, 因为任意内积可以自然的诱导出一个范数
$$
\parallel x \parallel := \sqrt{<x, x>}
$$

> 不是所有范数都是由内积诱导出来的, 比如曼哈顿范数

#### 柯西-施瓦兹不等式

内积空间中由内积诱导的范数满足柯西-施瓦兹不等式:
$$
\mid <x,y> \mid \le \parallel x \parallel \cdot \parallel y \parallel
$$


> 不理解啊, 内积必然不为负, 那左边这个绝对值是干嘛的
>
> 哦, 只有诱导范数的内积<x, y>是保证非负的, x,y 之间的不一定

#### 向量距离的定义(3.6)

考虑一个内积空间$(V, <\cdot, \cdot>)$, 任取V上向量x, y, 称
$$
d(x, y):= \parallel x-y \parallel = \sqrt{<x-y, x-y>}
$$
为向量x和y之间的距离. 如果我们选用点积作为V上的内积, 则得出的距离称为欧式距离.

对于这样的映射
$$
d: V \times V \to \mathbb{R}\\
(x,y) \mapsto d(x, y)
$$
称为度量

一个度量d满足以下三条性质:

1. (正定性) 对于任意V上的x,y, d(x,y)大于等于0, 当且仅当x=y时取等
2. (对称性) 对任意V上的x,y, d(x,y) =d(y,x)
3. (三角不等式) 对V上的任意x,y,z, d(x, y) + d(y, z) $\geq$ d(x,z)



## 向量夹角和正交

内积除了可以对向量长度和向量间距离进行定义, 还可以通过定义两向量之间的夹角$\omega$ 来刻画线性空间中的几何特征.

我门使用Cauchy-Schwarz(柯西-施瓦兹)不等式定义内积空间中两个向量x, y之间的夹角$\omega$ , 假设两个向量均布为零(?), 有
$$
-1 \le \frac{<x,y>}{\parallel x \parallel \parallel y \parallel} \le 1
$$
因此在[0, pi]中, 有唯一的$\omega$满足
$$
cos \omega = \frac{<x, y>}{\parallel x \parallel \parallel y \parallel}
$$
而$\omega$就是x, y之间的夹角, 直观意义上, 夹角给出了其方向上的相似程度



内积的一个关键用途是判断向量之间是否正交

#### 向量的正交的定义(3.7)

两个向量x, y 正交(orthogonal)当且仅当它们的内积为0, 即$<x,y> = 0$ 记作$x\perp y$

进一步的, 如果x, y的范数都是1, 也即两个向量是单位向量, 则称它们单位正交(orthonormal)

> 根据使用的内积不同, 在一个内积下正交的两个向量在另一个内积下不一定正交

#### 正交矩阵的定义 (3.8)

方阵$A \in \mathbb{R}^{n\times n}$为正交矩阵当且仅当满足以下条件:
$$
AA^T = I = A^TA\\
进而有\\
A^{-1} = A^T
$$
也就是说正交矩阵的逆就是它的转置.

> 正交矩阵可以看作是向量正交的扩展, 向量正交描述两个向量之间是否正交, 而正交矩阵可以描述n个向量是否互相正交且都为单位向量

正交矩阵对应的变换在线性空间保持向量的长度和向量间的夹角, 正交矩阵对应的变换在$\mathbb{R}^2, \mathbb{R}^3$中属于刚体变换.

> 上面这句话的内容可以非常简单的得到证明, 使用(389)和(390).

## 正交基

基向量互相垂直且都为单位向量, 这样的基称为正交基

#### 正交基的定义(3.9)

考虑一个n维线性空间V和其上的一组有序基B, 如果
$$
<b_i, b_j> = 0, x \neq y\\
<b_i, b_i> = 1
$$


对所有的i,j = $1, \dots,n$都成立, 则称B为标准正交基(orthonormal basis ONB), 假如只满足正交不满足单位向量, 则称正交基(OB)



使用高斯消元法和增广矩阵将未标准化, 非正交的向量组变为标准正交基的方法叫做Gram-Schmidt正交化过程.

## 正交补

在定义了正交这一概念后可以看看互相正交的线性空间. 这样的线性空间在第十章讨论线性降维的几何视角时十分重要.

考虑一个D维的线性空间V和一个M维的子空间$U \subset V$, U的正交补(orthogonal complement)$U^{\perp}$是一个(D-M)维的子空间, 其中的任何向量都与U中的任何向量垂直, 进一步我们有$U \cap U^{\perp}=\{0\}$ , 于是V中的任何向量x都可以被唯一分解为下面的形式:
$$
x = \sum_{m=1}^{M}\lambda_mb_m + \sum_{j=1}^{D-M}\phi_jb_j^{\perp}, \lambda_m, \phi_j \in \mathbb{R}
$$


其中b是U的基, $b^{\perp}$是$U^{\perp}$的一个基.



一般地, 正交补空间可用来刻画n维线性空间和仿射空间中的超平面

> 搞一个向量当一维空间的基, 跟它正交的一切向量狗证一个超平面是吧



## 函数的内积

把一个函数的值排成一排, 怎么不能看作是一个向量呢

把两个函数的值排成两排, 每个值乘积再相加, 怎么不能看作是这俩函数的内积呢.

两个函数$u: \mathbb{R} \to \mathbb{R}, v: \mathbb{R} \to \mathbb{R}$之间的内积可以被定义为下面的定积分:
$$
<u, v> := \int_a^bu(x)v(x)dx
$$
其中积分限满足a,b < $\infty$



和通常的内积一样, 我们也可以通过内积定义函数的范数和正交关系.

如果(33)的结果为0, 则两个函数相互正交.

如果需要给出更加严格的睇你, 我们需要考虑测度和积分定义的方式, 这将引出Hibert空间

进一步地, 与有限维向量间的内积不同, 函数之间的内积可能发散(值为无穷大).

上述情形的讨论涉及实分析和泛函分析中的细节, 本书不做讨论.



#### 示例3.9

假设我们令$u=sin(x), v = cos(x)$, 则内积定义中的被积函数为$f = u(x)v(x)$, 它是一个奇函数, 也即-f(x) = f(-x), 在积分限为$a = -\pi, b = \pi$的定积分的值为零, 因此我们得出sin和cos互相正交的结论.

上述结论对于下面的函数族仍然成立:
$$
{1, cos(x), cos(2x), \dots}
$$
如果将积分限设置为$-\pi, \pi$. 换句话说, 这个函数族中的函数俩俩正交, 它们张成的巨大空间是以这个积分限为周期的所有连续函数. 将函数向这个子空间上投影是Fourier级数的核心思想.

> 太抽象了, 不能说完全懂了

在之后, 我们还会遇见第二种不常见的内积——随机变量之间的内积.



## 正交投影

投影是一类重要的线性变换(其他重要的线性变换还有旋转和反射), 在很多领域占有重要的地位.

在机器学习中, 我们经常需要和高维数据打交道, 它们往往难以进行分析和可视化. 然而, 高维数据往往具有大部分信息被包含在仅仅几个维度之中, 其他维度对于数据关键信息的刻画并不重要的特点.

当我们对高维数据进行压缩和可视化时, 我们将失去一些信息. 为了将压缩造成的信息损失最小化, 我们往往选择数据中最关键的几个维度.

之前提到, 数据可以被表示为向量, 在本章中, 我们将对基础的数据压缩方法进行讨论. 具体而言, 我们可以将原来的高维数据投影到低维特征空间(feature space). 然后在此空间中对数据进行处理和分析, 以更好的了解数据集并抽取相关的模式(pattern).

> 特征, 是数据表示中的一个常见说法

以主成分分析(PCA)为例的机器学习算法, 以及以自编码器为例的深度神经网络充分利用了降维的思想.

给定一个低维子空间, 来自高维空间中数据的正交投影会保留尽可能多的信息, 并最小化元数据和投影数据的区别和损失.



#### 投影的定义(3.10)

令V是一个线性空间, $U \subset V$是V的子空间, 如果一个线性映射$\pi: V \to U$ 满足 $\pi^2 = \pi \circ \pi =\pi$, 则称$\pi$是一个投影(projection).

> 我只看懂了 一个投影点经过投影应该映射在它本身, $\pi^2$这块没看懂

也即投影矩阵是一个特殊的映射矩阵, 满足$P_\pi =P_\pi^2$

### 向一维子空间(直线)投影



假设给定一条通过原点的直线(一维子空间), 和该空间的一个基$b \in \mathbb{R}^n$ . 这条直线是b张成的子空间$U \subset \mathbb{R}^n$ . 当我们将向量x投影到U中时, 我们需要在U中寻找距离x最近的向量$\pi_U(x) \in U$. 下面列举一些投影向量$\pi_U(x)$的性质

- 投影向量是(子空间中)距离x最近的向量, 最近是说它俩的距离最小.这表示它俩的差值向量和U是垂直的, 也和U的基b垂直.
- x到U的投影向量一定是U中的元素, 因此也和U的基b共线, 于是存在$\lambda \in \mathbb{R}$, 使得$\pi_U(x)= \lambda b$. 

> 想起来了, x到U的投影就是$<x, b>\cdot b$
>
> $\lambda$也是x到U投影在b下的坐标

下面我们将通过三个步骤确定坐标$\lambda$ , 投影向量, 以及将x投影到U的投影矩阵$P_\pi$

1. 计算坐标$\lambda$的值. 

由正交性条件可以得到
$$
<x-\pi_U(x), b> = 0 \Longleftrightarrow^{\pi_U(x) = \lambda b} <x-\lambda b, b> = 0
$$
由内积的双线性性可以得到
$$
<x, b> - \lambda <b, b> = 0 \Longleftrightarrow \lambda = \frac{<x, b>}{<b, b>} = \frac{<b, x>}{\parallel b \parallel ^ 2}
$$
假如使用点积作为内积, 那就是
$$
\frac{b^Tx}{b^Tb}
$$


2. 计算投影点$\pi_U(x) \in U$, 因为就等于$\lambda$b

3. 计算投影矩阵$P_\pi$ . 

$$
\pi_u(x) = \lambda b = b \frac{b^Tx}{\parallel b \parallel^2} = \frac{bb^T}{\parallel b \parallel^2}x
$$

因此 $P_\pi =\frac{bb^T}{\parallel b \parallel^2} $ (注意P是秩为1的对称矩阵)



> 在第四章, 我们将证明$\pi_U(x)$是矩阵$P_\pi$的一个特征向量, 对应的特征值为1,
>
> 夭寿啦, 一点看不懂, 啥是特征向量, 啥是特征值

### 向一般子空间投影

将向量$x \in \mathbb{R}^n$投影至较低维度的一般子空间$U \subset \mathbb{R}^n$, 其中U满足$dimU =m \ge 1$

假设$(b_1,\dots,b_m)$是U的一个有序基, U上的任意投影向量$\pi_U(x)$必定是U的元素, 因此必然是有基上的一个线性组合, 满足$\pi_U(x)= \sum_{i=1}^{m}b_i\lambda_i$ 

还是三步走

1. 确定$\lambda_1,\dots \lambda_m$

$$
<x - U_{\pi}(x), b_i> = 0\\
\Rightarrow b_i^T(x-B\lambda) = 0 \\
\Rightarrow B^T(x-B\lambda) = 0 \\
\Rightarrow B^TB\lambda = B^Tx
\Rightarrow \lambda = (B^TB)^{-1}B^Tx
$$

最后得到的方程叫做正规方程(normal equation), 由于b是基, 所以它们线性无关, 所以$B^TB\in \mathbb{R}^{m\times m}$是正规矩阵, 存在逆矩阵.

其中$(B^TB)^{-1}B^T$称为矩阵B的伪逆. 在实际操作中, 我们常常对$B^TB$添加一个摄动项(jitter term)$\epsilon I (\epsilon > 0)$来满足正定性和数值稳定性

2. 确定$\pi_U{x}$

$$
\pi_U(x) = B\lambda = B(B^TB)^{-1}B^T x
$$

3. 确定投影矩阵$P_{\pi}$

$$
P_{\pi}=B(B^TB)^{-1}B^T
$$



投影让我们对无解的线性系统$Ax=b$可以进行研究. 

> 这个Ax=b可以等价于寻找作用域中的x, 可以通过映射A, 映射为值域中的b
>
> 也可以理解成在A的列向量张成的空间中寻找b(的坐标)

那么如果b不在A的张成空间中, 我们现在可以找出一个近似解, 也就是A的张成空间中最接近b的向量.

我们可以计算b到A的列空间的投影, 这个投影就是要求的近似解. 其得到的结果叫做超定系统(over-determined system)的最小二乘估计(least-squares). 

如果再引入重构损失(reconstruction error), 就构成了推导主成分分析的一种方式.

另外, 如果B是一组标准正交基, 则$B^TB=I$, 则投影可以简化为$\pi_U(x)=BB^Tx,\lambda=B^Tx$



### Gram-Schmidt正交化

投影是Gram-Schmidt正交化的核心, 后者可以让我们从任意的n维线性空间V的一个基$(b_i,\dots,b_n)$构造出该空间的一个标准正交基$(u_1,\dots,u_n)$. 

所谓Gram-Schmidt正交化方法在给定V的任意基B的情况下迭代的构造出正交基U, 其过程如下:
$$
u_1:=b_1\\
u_k:=b_k-\pi_{span(u_1,\dots,u_{k-1})}(b_k), k=2,\dots,n
$$

### 向仿射子空间投影

给定一个仿射空间$L=x_0+U$, 其中$b_1, b_2$是U上的一个基.则x到L的投影为:
$$
\pi_L(x) = x_0 + \pi_U(x-x0)
$$
其中$\pi_U(x-x_0)$就是计算到子空间U的投影, 我们已经会算了.

## 旋转

保长和保角是正交矩阵所表示变换的特征.

一个旋转是指一个将某个平面关于原点旋转角度$\theta$的线性映射.根据通常的约定, 旋转角$\theta>0$表示逆时针旋转.

### $\mathbb{R}^3$中的旋转

确定一个一般的旋转最简单的方法就是找到它是如何旋转标准基$e_1, e_2, e_3$, 旋转后的标准基就是新的基, 拼起来就是旋转矩阵R.

### n维空间中的旋转

#### Givens旋转的定义

设V是n维欧氏空间, 其上的自同构$\Phi:V \to V$若可以表示为
$$
R_{i,j}(\theta):=\begin{bmatrix}I_{i-1}\\&cos\theta&&-sin\theta\\&&I_{j-i-1}\\&-sin\theta&&-cos\theta\\&&&&I_{n-j} \end{bmatrix} \in \mathbb{R}^{n\times n}
$$
则$R_{i,j}(\theta)$叫做Givens旋转



# 第四章 矩阵分解

在本章, 会介绍矩阵的三个方面: 如何对矩阵组合, 如何分解矩阵, 以及如何将这些分解用于矩阵近似

首先是能用几个数字来描述矩阵特征的方法, 这些数字表征了矩阵的整体性质.这些特征值具有重要的数学意义.使我们可以快速掌握矩阵具有哪些有用的性质.

在这里还会介绍矩阵分解的方法: 矩阵分解可以类比为数字的因式分解. 因此矩阵分解(matrix decomposition)也常被称为matrix factorization.

还有对称正定矩阵的平方根运算, 即Cholesky分解. 从这里, 会介绍将矩阵分解为规范形式的两种相关方法. 第一种称为矩阵对角化, 如果选择的基合适, 它允许我们使用对角变换矩阵来表示线性映射. 第二种方法是奇异值分解, 将这种因式分解扩展到非方阵, 它被认为是线性代数中的基本概念之一.

最后会以矩阵分类的形式系统的概述矩阵的类型和区分它们特征属性.

## 矩阵的行列式与迹

行列式是线性方程组分析和求解中的数学对象, 行列式仅在方阵$A\in \mathbb{R}^{n\times n}$上定义.

在本书中, 把行列式写作$det(A)$, 或$|A|$.

方阵A的行列式是将A映射为一个实数的函数.

#### 定理4.1

对于任意方阵$A\in \mathbb{R}^{n\times n}$, 当且仅当det(A)$\neq0$时A可逆.

对于2维方阵有:
$$
det(A) = \begin{vmatrix} a_{11} & a_{12}\\a_{21}& a_{22} \end{vmatrix} = a_{11}a_{22}-a_{12}a_{21}
$$


对于一个三角矩阵$T\in \mathbb{R}^{n\times n}$, 它的行列式就是对角线上元素的乘积
$$
det(T) = \Pi_{i=1}^nT_{ii}
$$


#### 定理4.2 拉普拉斯展开

考虑一个矩阵$A\in \mathbb{R}^{n\times n}$, 那么, 对于$j=1,\dots,n$, 有:

1. 按第j列展开:

$$
det(A) = \sum_{k=1}^n(-1)^{k+j}a_{kj}det(A_{k,j})
$$



2. 按第i行展开:

$$
det(A) = \sum_{k=1}^n(-1)^{k+i}a_{ik}det(A_{i,k})
$$

其中$A_{i,k}$表示矩阵A删除第i行和第k列得到的子矩阵.

对于$A\in \mathbb{R}^{n\times n}$, 行列式具有以下性质:

- 矩阵乘积的行列式等于行列式的乘积, 即$det(AB) = det(A)det(B)$

- 矩阵转置后求行列式和自身行列式相等, 即$det(A)=det(A^T)$

- 如果矩阵A是正规矩阵(可逆), 那么$det(A^{-1})=\frac{1}{det(A)}$

- 相似矩阵具有相同的行列式,

  - 因此, 对于自同态映射, 也即线性映射$\Phi: V \to V, \Phi$中的所有变换矩阵$A_{\Phi}$ 具有相同的行列式, 因此行列式对自同态映射基的选择是不变的

  > 对于V上的一组基B, 变换矩阵是A的话, 那么对于另一组基C, 假设B到C的变换矩阵是T, 那么变换矩阵就是$T^{-1}AT$, 显然$det(A)=det(T^{-1}AT)$

- 将行/列的倍数添加到另一行/列不会改变矩阵的行列式

- 将某一行/列放大n倍会使得行列式也放大n倍.特别地, $det(\lambda A) = \lambda^ndet(A)$

- 交换两行/两列会改变行列式的符号

由于最后三个性质, 我们可以使用高斯消元法来将A转换为行阶梯形式来计算det(A), 当A是三角矩阵的时候, de t(A)就是对角元素的乘积



#### 定理4.3 当且仅当A满秩时A可逆

一个方阵$A\in \mathbb{R}^{n\times n}$有$det(A)\neq 0$, 当且仅当ran k(A)=n

#### 定义4.4 方阵的迹

一个方阵$A\in \mathbb{R}^{n\times n}$的迹为:
$$
tr(A) = \sum_{i=1}^na_{ii}
$$
即, 一个矩阵的迹是对角线元素之和

迹满足以下性质:

- 对于$A,B\in \mathbb{R}^{n\times n}, tr(A+B)=tr(A)+tr(B)$
- 对于$A\in \mathbb{R}^{n\times n}, \alpha \in \mathbb{R}, tr(\alpha A)=\alpha tr(A)$
- $tr(I_n)=n$
- 对于$A\in \mathbb{R}^{n\times k},B\in \mathbb{R}^{k\times n}, tr(AB)=tr(BA)$, 这条有趣

只要一个函数同时满足以上四条性质, 就是矩阵的迹

对于相似矩阵, 迹也是相同的, 证明如下:
$$
假设B是A的相似矩阵, 则\\
B=S^{-1}AS \\
tr(B)=tr(S^{-1}AS)=tr(ASS^{-1})=tr(A)
$$


#### 定义4.5 方阵的特征多项式

使用多项式描述矩阵A的重要方程

对于$\lambda\in \mathbb{R}和A\in \mathbb{R}^{n\times n},$
$$
p_A(\lambda)=det(A - \lambda I)\\
=c_0 + c_1\lambda + c_2\lambda^2 + \dots +c_{n-1}\lambda^{n-1} + c_n\lambda^n
$$
其中$c_0,c_1\dots,c_n$被称作A的特征多项式, 特别地
$$
c_0 = det(A)
c_{n-1}=(-1)^{n-1}tr(A)
$$
特征多项式允许我们计算特征值和特征向量



## 特征值与特征向量

我们可以通过进行“特征”分析来解释线性映射及其相关的变换矩阵.

线性映射的特征值将告诉我们一组特殊向量(即特征向量)是如何被线性映射变换的.

#### 特征值, 特征向量, 特征值方程定义4.6

设$A \in \mathbb{R}^{n\times n}的方阵. 那么, 如果\lambda \in \mathbb{R}$满足
$$
Ax = \lambda x
$$


则称$\lambda$是A的特征值, 而$x \in \mathbb{R}^n\setminus\{0\} $ 为对应的特征向量. 我们称(53)为特征值方程



以下陈述是等价的:

- $\lambda$ 是$A \in \mathbb{R}^{n\times n}$的特征值
- 存在一个$x \in \mathbb{R}^n\setminus\{0\} $ , 使得$Ax = \lambda x$, 或等价地, $(A-\lambda I_n)x =0$存在非0解, 即x$\neq$0
- $rk(A-\lambda I_n) < n$
- $det(A-\lambda I_n) = 0$

#### 共线性和同向性(定义4.7)

两个指向相同方向的向量称为同向的. 如果两个向量指向相同或相反的方向, 则它们是共线的.

特征向量的非唯一性: 如果x是与特征值$\lambda$相关联的A的特征向量, 则对于任意的$x\in \mathbb{R}\setminus \{0\}$, cx也是与相同特征值相关联的A的特征向量, 因为:
$$
A(cx) = cAx = c\lambda x = \lambda (cx)
$$
因此所有与x共线的向量也都是A的特征向量.

#### 定理4.8 特征值与特征多项式的根

$\lambda$是A的特征值当且仅当$\lambda$是矩阵A的特征多项式$p_A(\lambda)$的根



#### 定义4.9 特征值的代数重数

设方阵A有一个特征值$\lambda_i$, 则$\lambda_i$的代数重数是指该根在特征多项式中出现的次数



#### 定义4.10 特征空间和特征值

对于$A \in \mathbb{R}^{n\times n}$ , 与特征值$\lambda$相关联的所有特征向量张成的$\mathbb{R}^n$的子空间称为A关于$\lambda$的特征空间, 记作$E_\lambda$, 矩阵A的所有特征值的集合称为A的特征谱或简称为谱.



如果$\lambda 是A$的特征值, 则对应的特征空间$E_\lambda$是齐次线性方程组$(A-\lambda I_n)x = 0$的解空间. 

关于特征值和特征向量的有用性质包括:

- 矩阵A及其转置矩阵$A^T$具有相同的特征值, 但不一定具有相同的特征向量

- 特征空间$E_\lambda$是$A-\lambda I$的零空间, 因为

  - $$
    Ax = \lambda x \Leftrightarrow (A - \lambda I)x = 0 \Leftrightarrow x \in ker(A-\lambda I)
    $$

- 相似矩阵具有相同相同的特征值, 因此线性映射的特征值和其变换矩阵的基的选择无关.

- 对称, 正定矩阵总是具有正实特征值



#### 定义4.11特征值的几何重数

 特征值的几何重数就是它关联的特征向量张成的特征空间的维度.

特征值的几何重数不能超过它的代数重数, 但可能更低; 一个特征值的几何重数至少是1



#### 定理4.12 

一个矩阵$A \in \mathbb{R}^{n\times n}$的具有n个不同特征值$\lambda_1,\dots,\lambda_n$的特征向量$x_1,\dots,x_n$是线性无关的.

这个定理表明, 具有n个特征值的矩阵的特征向量构成$\mathbb{R}^n$的一个基 (自同构?)



#### 定义4.13 缺陷矩阵

如果一个方阵$A \in \mathbb{R}^{n\times n}$拥有的线性无关特征向量少于n个, 则称该矩阵是缺陷的. 一个非缺陷的矩阵$A \in \mathbb{R}^{n\times n}$不一定需要有n个不同的特征值, 但它确实需要其特征向量构成$\mathbb{R}^n$的一个基. 观察缺陷矩阵的特征空间, 可以得出特征空间的维数之和小于n. 特别是, 缺陷矩阵至少有一个特征值$\lambda_i$, 其代数重数$m > 1$, 但几何重数小于m.



#### 定理4.14 方阵特征扩展到一般矩阵的前置定理?

给定矩阵$A \in \mathbb{R}^{m\times n}$, 我们总是可以通过定义
$$
S:= A^TA
$$
来获得一个对称且半正定的矩阵S$\in \mathbb{R}^{n\times n}$

特别地, 如果rk(A) = n, 那么S是对称且正定的

#### 定理4.15 谱定理

如果$A \in \mathbb{R}^{n\times n}$是对称的, 则存在由A的特征向量构成的对应线性空间V的一个正交规范基, 且每个特征值都是实数

> 是说, 如果A是对称方阵, 则它一定是非缺陷的?

谱定理的一个直接推论是, 对称矩阵A的特征分解存在(具有实数特征值), 并且我们可以找到一个由特征向量构成的正交规范基, 使得$A = PDP^T$, 其中D是对角矩阵, P的列包含特征向量.



#### 定理4.16 一个矩阵的行列式是其特征值的乘积, 即

$$
det(A) = \Pi_{i=1}^{n} \lambda_i
$$

其中$\lambda_i$是A的(可能重复的)特征值, 每个$\lambda_i$要乘它的代数重数次



#### 定理4.17 一个矩阵的迹是其特征值的和, 即

$$
tr(A) = \sum_{i=1}^n\lambda_i
$$

其中$\lambda_i$是A的(可能重复的)特征值, 每个$\lambda_i$要加它的代数重数次

## Cholesky分解

对于对称正定矩阵, 我们可以选择多种与平方根等效的操作. 其中, Cholesky分解提供了一种在对称正定矩阵上进行类似平方根操作的方法.



#### 定理4.18 Cholesky分解

一个对称正定矩阵A可以分解为两个矩阵的乘积, 即$A = LL^T$, 其中L是一个下三角矩阵, 且其对角线元素为正.

通过将矩阵分解为下三角矩阵的乘积, 对它的逆和行列式计算将变得简单, 且L也是好求的, 比如$a_{11} = l_{11}^2$



## 特征值分解与对角化

一个对角矩阵(Diagonal Matrix)是一个在所有非对角线上元素都为零的矩阵, 即它的形式为:
$$
D = \begin{bmatrix} c_1 &\dots &0\\\vdots &\ddots &\vdots \\
0&\dots&c_n\end{bmatrix}
$$
对角矩阵允许我们快速的计算行列式, 矩阵的幂以及逆矩阵. 

具体来说, 对角矩阵的行列式等于对角线上元素的乘积; 矩阵的幂$D^k$等价于对角线上元素求k次幂; 如果对角矩阵的所有对角元素不为0, 那么它的逆就是所有对角元素取倒数.

这一章节我们将讨论如何将矩阵化成对角形式. 其中对角矩阵的对角线上包含矩阵A的特征值



#### 定义4.19 可对角化

一个矩阵$A \in \mathbb{R}^{n\times n}$是可对角化的, 如果它与一个对角矩阵相似, 即如果存在一个可逆矩阵P, 使得$D = P^{-1}AP$

对角化一个矩阵A是表达相同线性映射但使用另一个基的一种方式. 这个基将被证明是用A的特征向量组成的.



证明D的对角线是特征值, P由特征向量组成:

令$A \in \mathbb{R}^{n\times n}, \lambda_1,\cdots, \lambda_n$为一系列的标量, $p_1,\cdots,p_n$是分布在$\mathbb{R}^n$空间的向量.我们定义矩阵$P:=[p_1,\cdots,p_n]$并令矩阵$D$为一个对角线为$\lambda_i$的对角矩阵. 

于是我们可以得到$AP=PD$当且仅当$\lambda$是A的特征值, 且$p_i$是A对应特征值的特征向量时, 以下等式成立:
$$
A = PDP^{-1}
$$
因为:
$$
AP = [Ap_1,\cdots, Ap_n] \\
PD = [\lambda_1 p_1, \cdots, \lambda_n p_n]\\
\rightarrow \\
Ap_i = \lambda_ip_i
$$


#### 定理4.20 特征分解

一个n x n的方阵A可以被分解为$A = PDP^{-1}$, 其中P是nxn的, D是一个对角矩阵, 且对角线上元素是A的特征值, 当且仅当A的特征向量构成$\mathbb{R}^n$的一个基.

#### 定理4.21 对称矩阵总是可以被对角化

根据谱定理, 对阵矩阵总是非缺陷的

另外矩阵的Jordan标准型提供了一种适用于缺陷矩阵的分解, 但这超出了本书的范围.



## 奇异值分解

矩阵的奇异值分解(SVD)是线性代数中的一种核心矩阵分解方法. 它被称作线性代数的基本定理.

因为它可以应用于所有矩阵, 而不仅仅是方阵, 并且它总是存在. 此外, 我们将论证, 矩阵A的SVD, 代表了一个线性映射$\Phi : V \to W$, 量化了这两个线性空间底层几何之间的变化.

#### 定理4.22 SVD定理

设$A \in \mathbb{R}^{m\times n}$是一个秩为$r \in [0, min(m,n)]$的矩阵, A的SVD是一种形式如下的分解:
$$
A = U\Epsilon V^T, 其中U\in \mathbb{R}^{m\times m}, \Epsilon \in \mathbb{R}^{m\times n}, V \in \mathbb{R}^{n\times n}
$$
其中U是一个正交矩阵, 其列向量记为$u_i$; V也是一个正交矩阵, 其列向量记为$v_i$; 此外$\Epsilon$的对角线元素$\Epsilon_{ii} = \alpha_i \geq 0$, 且非对角线元素一定为0

$\Epsilon$的对角线元素称为奇异值; $u_i$称为左奇异向量, $v_i$称为右奇异向量. 按照惯例, 奇异值是有序的, 即$\alpha_1 \ge \alpha_2\ge\cdots\ge \alpha_r \ge 0$

奇异值矩阵是唯一的. 对于任意矩阵A, 其SVD都是存在的.

### 奇异值分解的图形表示

矩阵的SVD可以被解释为相应线性映射$\Phi : \mathbb{R}^n \to \mathbb{R}^m$分解为三个操作:

- 先通过$V^T$进行基变换
- 然后通过$\Epsilon$ 进行尺度变换和维数增减
- 最后通过U再进行基变换

### 奇异值分解(SVD)的构建

对称正定矩阵的特征分解就是它的奇异值分解

接下来, 我们将证明SVD定理的成立以及SVD是如何构建的.

计算$A \in \mathbb{R}^{m\times n}$的SVD等价于找到陪域$\mathbb{R}^m$和定义域$\mathbb{R}^n$的两组正交归一基U和V. 从这些有序基中, 我们将构建矩阵U和V.

我们的计划是从构建右奇异向量的正交归一几何开始, 然后构建左奇异向量的正交归一几何, 之后, 我们将两者联系起来, 并要求在A的变换下保持v的正交性, 因为我们知道$Av_i$构成的集合是正交向量(是的, 把A拆成UEV可以看得更清楚). 然后我们将通过标量因子对这些图像进行归一化, 这些标量因子就是奇异值.

首先构建右奇异向量, 谱定理告诉我们, 对称矩阵一定可以被对角化, 此外, 我们可以从任何矩阵构造一个对称, 半正定的矩阵$A^TA$, 因此, 我们总可以对$A^TA$进行对角化, 得到
$$
A^TA = PDP^T = P\begin{bmatrix}\lambda_1&\cdots &0\\\vdots&\ddots & \vdots\\0& \cdots&\lambda_n \end{bmatrix}P^T
$$
其中P是一个正交矩阵, $\lambda_i\ge 0$是$A^TA$的特征值, 假设A的SVD存在, 并将(63)代入, 我们得到:
$$
A^TA = (UEV^T)^T(UEV^T) = VE^TU^TUEV^T, 因为U,V是正交矩阵\\
\Rightarrow A^TA = VE^TEV^T = V\begin{bmatrix}\alpha_1^2 &\cdots& 0\\\vdots &\ddots&\vdots\\0&\cdots&\alpha_n^2 \end{bmatrix}V^T
$$
比较(63)和(64), 可知
$$
P = V\\ \lambda_i = \alpha_i^2
$$
接下来要得到左奇异矩阵U, 我们计算$AA^T$的SVD, 得到
$$
AA^T = (UEV^T)(UEV^T)^T = UEV^TVE^TU^T = UEE^TU^T = U [..]U^T
$$
可以U就是$AA^T$特征分解得到的P

---



我们要求A下$v_i$的像必须正交, 也即$(Av_i)^T(Av_j) = 0, 当i\ne j \Rightarrow (Av_i)^T(Av_j)=v_i^TA^TAv_j =v_i\lambda_jv_j = \lambda_jv_i^Tv_j = 0$
$$
u_i := \frac{Av_i}{||Av_i||} = \frac{1}{\lambda_i}Av_i = \frac{1}{\alpha_i}Av_i
$$
我们可以得到奇异值方程:
$$
Av_i = \alpha_iu_i, i = 1,\cdots,r
$$
也可以写成
$$
AV = UE \Rightarrow A = UEV^T
$$
所以U可以通过V在A下的像来求



### 特征值分解与奇异值分解

- 所有矩阵都有奇异值分解, 只有方阵可能有特征值分解, 且不是所有方阵都有特征值分解
- 特征值分解矩阵P中的列向量不一定正交, 基基变换不仅仅是旋转和缩放;奇异值分解中的V和U中的列向量是正交归一的, 因此它们只表示旋转.
- 特征值分解和奇异值分解都是三个线性映射的组合.
- 在奇异值分解中, 对角元素都是实数且非负; 对于特征值分解这一点一般不成立
- 对于对称矩阵, 其特征值分解和奇异值分解是相同的

## 矩阵近似

现在, 我们不进行完整的SVD分解, 而是研究SVD如何允许我们将矩阵A表示为更简单(低秩)的矩阵$A_i$之和, 这种表示法构成了一种矩阵近似方案, 其计算成本低于完整的SVD

我们构造一个秩为1的矩阵$A_i\in \mathbb{R}^{m\times n}$ ,形式为:
$$
A_i:=u_iv_i^T
$$
一个秩为r的矩阵A可以表示为秩为1的矩阵之和, 即
$$
A = \sum_{i=1}^r\alpha_iu_iv_i^T = \sum_{i=1}^r\alpha_iA_i
$$
如果求和不是遍历所有r个矩阵, 而是仅到中间值$k < r$, 则我们得到一个秩为k的近似.
$$
\hat{A(k)}:= \sum_{i=1}^k\alpha_iu_iv_i^T = \sum_{i=1}^k\alpha_iA_i
$$
其中$rk(\hat{A}(k)) = k$.

为了测量矩阵A和其秩为k的近似矩阵之间的差异, 我们需要范数的概念.

#### 定义4.23 矩阵的谱范数

对于$x\in \mathbb{R}^n\setminus\{0\} $ ,矩阵$hat{A}\in \mathbb{R}^{m\times n}$的谱范数定义为:
$$
||A||_2:= max_x\frac{||Ax||_2}{||x||_2}
$$
谱范数决定了任何向量x在乘以A之后可能达到的最大长度



#### 定理4.24 矩阵A的谱范数是其最大的奇异值$\alpha_1$



#### Eckart-Young定理 4.25

考虑一个秩为r的矩阵A, 以及一个秩为k的矩阵B, 对于任意的$k\le r$, 且$hat{A}(k) = \sum_{i=1}^k\alpha_iu_iv_i^T$, 则
$$
\hat{A}(k) = argmin_{rk(B)=k}||A-B||_2 \\
||A-\hat{A}(k)||_2 = \alpha_{k+1}
$$
Eckart-Young定理明确指出了我们使用秩为k的近似引入的误差量. 我们可以将使用SVD获得的秩k近似解释为全秩矩阵A在秩至多为k的矩阵构成的低维空间的投影. 在所有可能的投影中, SVD使A与任何秩k近似之间的误差(就谱范数而言)最小化.

# 第五章 向量微积分

许多机器学习算法都在优化一个目标函数, 也即优化一组模型参数, 这些参数控制着模型解释数据的好坏.

如何寻找好的参数可被表述为一个优化问题.

本章的核心概念是函数, 一个函数f是一个数学对象, 它将两个数学对象进行联系. 本书中涉及的数学对象即为模型输入$x\in \mathbb{R}^D$以及拟合目标(函数值)f(x), 如无额外说明, 默认拟合目标都是实数. 这里$\mathbb{R}^D$称为f的定义域(domain), 相对应的函数值f(x)所在的集合称为f的像集(image)或陪域(codomain).

## 一元函数的微分

#### 定义5.1 一元函数的差商

$$
\frac{\delta y}{\delta x} := \frac{f(x + \delta x) - f(x)}{\delta x}
$$

计算连接函数f之图像上的两点的割线的斜率.

如果f是线性函数, 差商也可以看作函数f上从点x到$x+\delta x$之间的平均斜率.若对$\delta x$去极限$\delta x \to 0$, 我们得到f在x处的斜率;如果f可微, 这个切线斜率就是f在x处的导数.



#### 定义5.2 导数

对正实数h>0, 函数f在x处的导数由下面的极限定义:
$$
\frac{df}{dx} := lim_{h\to 0}\frac{f(x+h)-f(x)}{h}
$$
f的导数时刻指向f提升最快的方向.



### Taylor 级数

所谓泰勒级数是将函数f表示成的那个无限项求和式.其中所有的项都和f在点$x_0$处的导数相关.

#### 定义5.3 泰勒多项式

函数$f: \mathbb{R} \to \mathbb{R}$在点$x_0$的n阶泰勒多项式是
$$
T_n(x) := \sum_{k=0}^n\frac{f^{(k)}(x_0)}{k!}(x-x_0)^k
$$
其中$f^{(k)}(x_0)$是f在$x_0$处的k阶导数(假设其存在), 而$\frac{f^{(k)}(x_0)}{k!}$是多项式各项的系数.

对于所有的$t \in \mathbb{R}$, 我们约定$t^0 = 1$



#### 定义5.4 泰勒级数

对于光滑函数$f \in \mathcal{C}^\infty, f: \mathbb{R} \to \mathbb{R}$, 它在点$x_0$处的泰勒级数定义为:
$$
T_{\infty}(x) = \sum_{k=0}^\infty\frac{f^{(k)}(x_0)}{k!}(x - x_0)^k
$$
若$x_0=0$, 我们得到了一个泰勒级数的特殊情况 ——Maclaurin级数. 如果$f(x) = T_\infty(x)$, 则我们称f是解析函数.



> 三角函数的幂级数表示:
> $$
> cos(x) = \sum_{k=0}^\infty\frac{(-1)^k}{(2k)!}x^{2k}\\
> sin(x) = \sum_{k=0}^\infty\frac{(-1)^k}{(2k+1)!}x^{2k+1}
> $$
> 

### 微分法则

下面介绍基本的微分法则, 其中我们使用$f'$表示f的导数

- 乘法法则: $[f(x)g(x)]' = f'(x)g(x) + f(x)g'(x)$
- 除法法则: $[\frac{f(x)}{g(x)}]' = \frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}$
- 加法法则: $[f(x) + g(x)]' = f'(x) + g'(x)$
- 链式法则: $(g[f(x)])' = (g\circ f)'(x) = g'[f(x)]f'(x)$



## 偏导数和梯度

一元函数的导数推广到多元情形就变成了梯度.

#### 定义5.5 偏导数

给定n元函数$f: \mathbb{R}^n \to \mathbb{R}, x \mapsto f(x), x\in \mathbb{R}^n$, 它的各偏导数为
$$
\frac{\delta f}{\delta x_i} = \lim_{h\to 0}\frac{f(x_i+h, x1,\dots,x_n) - f(x_1,\dots, x_n)}{h}
$$


然后将各偏导数组合为向量, 就得到了梯度向量:
$$
\nabla_xf = grad f = \frac{df}{dx} = [\frac{\delta f}{\delta x_1}, \dots, \frac{\delta f}{\delta x_n}] \in \mathbb{R}^{1\times n}
$$
其中n是变元数, 1是f象集(陪域)的维数. 我们在此定义列向量$x=[x_1,\dots,x_n]^T\in \mathbb{R}^n$. 行向量称为f的梯度或者jacobi矩阵

> jacobi矩阵是不是在函数是$f: \mathbb{R}^n\to \mathbb{R}^m$的情况.

> 注意梯度向量是个行向量, 特地写明了$\mathbb{R}^{1\times n}, 而非\mathbb{R}^n$, 行向量可以作用于对应的列向量(比如$\mathbb{R}^n$上的列向量x)得到一个标量, 这样的行向量称为余向量(covector)



### 偏导数的基本法则

$$
Product \ rule: \frac{\delta}{\delta x}[f(x)g(x)] = \frac{\delta f}{\delta x}g(x) + \frac{\delta g}{\delta x}f(x)\\
Sum \ rule: \frac{\delta}{\delta x}[f(x) + g(x)] = \frac{\delta f}{\delta x} + \frac{\delta g}{\delta x}\\
Chain\ rule: \frac{\partial}{\partial x}(g\circ f)(x) = \frac{\partial g}{\partial f}\frac{\partial f}{\partial x}
$$

### 链式法则

考虑变元为$x_1, x_2$的函数$f: \mathbb{R}^2 \to \mathbb{R}$, 而$x_1(t), x_2(t)$又是变元t的函数. 为了计算f对t的梯度, 需要用到链式法则
$$
\frac{\mathrm{d} f}{\mathrm{d} t} = \begin{bmatrix}\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2} \end{bmatrix}\begin{bmatrix}\frac{\partial x_1(t)}{\partial t}\\ \frac{\partial x_2(t)}{\partial t}\end{bmatrix} = \frac{\partial f}{\partial x_1}\frac{\partial x_1}{\partial t} + \frac{\partial f}{\partial x_2}\frac{\partial x_2}{\partial t}
$$
其中$\mathrm{d}$表示梯度, $\partial$表示偏导数.



## 向量值函数的梯度

之前讨论的是$f: \mathbb{R}^n \to \mathbb{R}$的偏导数和梯度, 接下来要将这个概念扩展至向量值函数(向量场)$f: \mathbb{R}^n\to\mathbb{R}^m$的情形, 其中$n\ge1, m\ge 1$.

给定向量值函数$f: \mathbb{R}^n \to\mathbb{R}^m, 和向量x = [x_1,\dots,x_n]^T\in \mathbb{R}^n$, 该函数的函数值可以写为
$$
f(x) = \begin{bmatrix}f_1(x)\\\vdots\\f_m(x)\end{bmatrix} \in mathbb{R}^m
$$
这样, 向量值函数f对变元$x_i$的偏导数为
$$
\frac{\partial f}{\partial x_i} = \begin{bmatrix} \frac{\partial f_1}{\partial x_i} \\\vdots \\\frac{\partial f_m}{\partial x_i}\end{bmatrix}
$$
f对x的梯度就是这些偏导数作为列排起来的矩阵



#### 定义5.6 Jacobi矩阵

向量值函数$f: \mathbb{R}^n \to \mathbb{R}^m$的各一阶偏微分的合集称为Jacobi矩阵, 它的形状是$m\times n$, 定义如下:
$$
J = \nabla_xf = \frac{\mathrm{d}f}{\mathrm{d}x} = \begin{bmatrix}\frac{\partial f}{\partial x_1},\dots,\frac{\partial f}{\partial x_n} \end{bmatrix} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1}&\cdots&\frac{\partial f_1}{\partial x_n} \\\vdots & \cdots&\vdots\\
\frac{\partial f_m}{\partial x_1}&\cdots&\frac{\partial f_m}{\partial x_n}\end{bmatrix} \\
J(i, j) = \frac{\partial f_i}{\partial x_j}
$$

> 本书的微分使用分子布局(numerator layout), 即f决定矩阵有几行, x决定矩阵有几列. 如果是分母布局(denominator layout), 就是分子布局的转置

行列式可以用来计算面积, 变换矩阵的行列式是整体的缩放比例, J acobi矩阵的行列式$|J|$称为Jacobi行列式, 它的值就是面积或体积变换前后的缩放比例.

容易证明, 变换矩阵其实就是变换映射的Jacobi矩阵
$$
给定 f(x) = Ax, f(x) \in \mathbb{R}^M, A\in \mathbb{R}^{M\times N} \\
为了计算梯度\frac{df}{dx}, 首先确定它的维度: 由于f: \mathbb{R}^N \to \mathbb{R}^M, 所以\frac{df}{dx} \in \mathbb{R}^{M\times N}, 为了计算梯度, 接下来计算每个偏导数\\
f_i(x) = \sum_{j=1}^NA_{i,j}x \Longrightarrow \frac{\partial f_i}{\partial x_j} = A_{i,j}\\
故 \frac{df}{dx} = A \in \mathbb{R}^{M\times N}
$$


Jacobi行列式在机器学习和深度学习的重参数技巧(Reparametrization Trick)中十分重要, 也被称为无穷摄动分析(Infinite Perturbation Analysis)

## 矩阵的梯度

当求矩阵对向量(或其他矩阵)的梯度时, 结果是一个多维度的张量(tensor). 

如果我们求mxn的矩阵A对pxq的矩阵B的梯度, 结果Jacobi矩阵的形状将是(m x n)x(p x q), 即一个四维张量J, 它的每个分量可以写作 $j_{i,j,k,l} = \frac{\partial A_{i,j}}{\partial B_{k,l}}$

>这个例子是从pxq的矩阵映射成mxn的函数



## 常用梯度恒等式

下面给出一些机器学习中常用的梯度恒等式(Petersen and Pedersen, 2012

其中$f(x)^{-1}$表示$f(x)$的逆(假设其存在).
$$
\frac{\partial}{\partial x}f(x)^T = (\frac{\partial f(x)}{\partial x})^T  \ 转置的偏微分等于偏微分的转置\\
\frac{\partial}{\partial x}tr[f(x)] = tr[\frac{\partial f(x)}{\partial x}] \ 迹的偏微分等于偏微分的迹 \\
\frac{\partial}{\partial x}det[f(x)] = det[f(x)]tr[f(x)^{-1}\frac{\partial f(x)}{\partial x}] \ 行列式的偏微分 \\
\frac{\partial}{\partial x}f(x)^{-1} = -f(x)^{-1}[\frac{\partial f(x)}{\partial x}]f(x)^{-1} \ 逆的偏微分 \\
\frac{\partial a^TX^{-1}b}{\partial X} = -(X^{-1})^Tab^T(X^{-1})^T\\
\frac{\partial a^TXb}{\partial X} = ab^T \\
\frac{\partial x^TBx}{\partial } = x^T(B + B^T) \\
\frac{\partial x^Ta}{\partial x} = a^T \\
\frac{\partial a^Tx}{\partial x} = a^T \\
\frac{\partial}{\partial s}(x-As)^TW(x - As) = -2(x-As)^TWA, for\ symmetric W
$$


对于高维张量, 它的转置和迹没有定义, 在这样的情况下, 形状为$D\times D\times E\times F$的张量的迹将是一个$E\times F$形状的矩阵, 这是张量缩并(tensor contraction)的一种特殊情况, 类似的, 当我们转置一个张量时, 我们说的是交换前两个维度.

> 就是对于高维张量, 把它看作2维矩阵, 只不过每一个元素都是一个矩阵, 然后执行迹计算的相加或转置的交换



## 反向传播与自动微分

在许多机器学习的应用中, 通过计算学习目标关于模型参数的梯度, 然后执行梯度下降找更优的模型参数.

对于给定的目标函数, 可以利用微积分的链式法则得到其对于模型参数的梯度.

考虑下面的函数:
$$
f(x) = \sqrt{x^2 + exp(x^2)} + cos[x^2+exp(x^2)]
$$


由链式法则 +微分的线性性, 我们可以得到

> 不展开写了, 反正一长串的链式

这样的显式求解麻烦得狠.

对于神经网络模型, 反向传播算法是一个种计算误差对模型参数梯度的有效方法.



### 深度神经网络中的梯度

深度学习领域将链式法则的功用发挥到了极致, 输入x经过多层复合的函数得到函数值y:
$$
y = (f_k\circ f_{k-1} \circ \dots \circ f_1)(x)
$$
其中x是输入(如图像), y是观测值(如类标签), 每个函数f_i各有参数.

在一般的多层神经网络中, 第i层中有函数
$$
f_i(x_{i-1}) = \sigma(A_{i-1}x_{i-1} + b_{i-1})
$$
训练这样的模型,需要一个损失函数L, 对它的值球关于所有模型参数$A_j, b_j$的梯度, 这同时要求我们求其对模型中各层的输入的梯度.

例如, 有输入x和观测值y和一个网络结构:
$$
f_0:=x
f_i:= \sigma_i(A_{i-1}f_{i-1} + b_{i-1}), i = 1,\dots,k
$$


我们关心找到使下面的平方损失最小的A, b:
$$
L(\theta) = ||y-f_k(\theta, x)||^2, 其中\theta = \{A_0, b_0, \dots,A_{k-1}, b_{k-1}\}
$$
根据链式法则, 可以得到:
$$
\frac{\partial L}{\partial \theta_{k-1}} = \frac{\partial L}{\partial f_k}\frac{\partial f_k}{\partial \theta_{k-1}}\\
\frac{\partial L}{\partial \theta_{k-2}} = \frac{\partial L}{\partial f_k}\frac{\partial f_k}{\partial f_{k-1}}\frac{\partial f_{k-1}}{\partial \theta_{k-2}}\\
\dots \\
\frac{\partial L}{\partial \theta_i} = \frac{\partial L}{\partial f_k}\dots \frac{\partial f_{i+2}}{\partial f_{i+1}} \frac{\partial f_{i+1}}{\partial \theta_i}
$$
其中A $\frac{\partial f_m}{\partial f_{m-1}}$是某层的输出相对于输入的偏导数, 而B$\frac{\partial f_m}{\partial \theta_{m-1}}$是某层的输出相对于参数的偏导数. 

其中A在计算的时候是可复用的.

### 自动微分

事实上, 反向传播是数值分析中常采用的自动微分(automatic differentiation)的一种特殊情况. 可以看作是一组通过中间变量和链式法则, 计算一个函数的精确数值梯度.

自动微分始于一系列初等算术运算(如加法, 乘法)和初等函数(如sin, exp, log), 通过将链式法则应用于这些操作, 可以自动计算出相当复杂的函数的梯度. 自动微分适用于一般的程序, 具有正向和反向两种模式

> 正向和反向?
>
> 从结果到输入(从最外层的函数到最内层的函数)就是反向
>
> 从输入到结果(从最内层的函数到最外层的函数)就是正向



下面重点关注反向自动微分, 即反向传播. 在神经网络中, 输入的维度通常比标签的维度高得多, 反向自动微分在计算上比正向的计算消耗低得多.



自动微分(反向)简单来说就是将一个大函数表示成初等函数为边的图结构, 然后从图的最终输出节点, 反向通过链式法则求微分.



## 高阶导数

梯度是一阶导数. 有时我们也关注更高阶的导数, 例如当使用牛顿法进行优化时, 需要二阶导数.

在之前讨论了泰勒级数, 即使用多项式近似函数, 在多变量的情况下也可以做同样的事.

考虑一个函数$f: \mathbb{R}^2\to\mathbb{R}$它有两个输入变量x,y, 我们使用以下符号表示高阶偏导数(和梯度):
$$
\frac{\partial ^2f}{\partial x^2}是f关于x的二阶偏导数\\
\frac{\partial^nf}{\partial x^n}是f关于x的n阶偏导数\\
\frac{\partial^2f}{\partial y \partial x} = \frac{\partial}{\partial y}(\frac{\partial f}{\partial x})是先对x求偏导, 然后对y求偏导得到的偏导数
$$
Hessian矩阵是所有二阶偏导数的集合.



如果f(x, y)是二阶(连续)可微函数, 那么$\frac{\partial^2f}{\partial x \partial y} = \frac{\partial^2f}{\partial y \partial x}$, 二阶偏导与求导顺序无关, 相应的Hessian矩阵为:
$$
H = \begin{bmatrix} \frac{\partial^2f}{\partial x^2} & \frac{\partial^2f}{\partial x \partial y} \\\frac{\partial^2f}{\partial x \partial y} & \frac{\partial^2f}{\partial y^2}\end{bmatrix}
$$
且是对称的. Hessian矩阵还可以表示为$\nabla^2_{x,y}f(x,y)$, 一般地, 函数$f: \mathbb{R}^n\to\mathbb{R}$的Hessian矩阵是一个nxn矩阵.

Hessian矩阵衡量了函数在(x,y)附近的局部曲率.



向量场的Hessian矩阵: 如果$f: \mathbb{R}^n \to \mathbb{R}^m$是一个向量场, Hessian矩阵是一个(m x n x n)的张量



## 线性近似和多元Taylor级数

函数f的梯度$\nabla f$通常被用作f在$x_0$附近的局部线性近似
$$
f(x)\approx f(x_0) + (\nabla_x)f(x_0) (x - x_0)
$$


对于定义在$\mathbb{R}^D$上的光滑函数 $f: \mathbb{R}^D \to \mathbb{R}$, 设差值向量$\delta := x - x_0$, 其泰勒级数展开为:
$$
f(x) = \sum_{k=0}^\infty\frac{D_x^kf(x_0)}{k!}\delta^k, 其中D_x^kf(x_0)表示f在x_0处的第k阶全导数.
$$
f在x0处的n阶泰勒多项式由级数的前n+1项构成:
$$
T_n(x) = \sum_{k=0}^n\frac{D_x^kf(x_0)}{k!}\delta^k
$$
其中$D_x^kf和\delta^k$都是k阶张量, 即k维数组

> 向量的幂是外积啊



# 第六章 概率与统计

## 概率空间的构建

在机器学习和统计学中, 概率有两种主要解释: 贝叶斯解释和频率解释

贝叶斯解释: 使用概率来指定用户对某个事件发生的不确定性成都, 它有时候被称为“主观概率”或“信念程度” (基于先验信息)

频率解释: 考虑感兴趣事件相对于发生事件总数的相对频率. (没有先验信息)



### 概率与随机变量

三个概念的区分:

- 概率空间
- 随机变量
- 与随机变量相关的分布或定律



现代概率论基于Kolmogorov提出的一组公理, 这些公理引入了样本空间, 事件空间和概率测度三个概念.

概率空间模型用于模拟具有随机结果的现实世界过程.

- 样本空间$\Omega$: 样本空间是实验所有可能结果的集合. 通常表示为$\Omega$, 例如, 连续两次抛硬币的样本空间为$\{hh, tt, th, ht\}$
- 事件空间$\mathcal{A}$: 事件空间是实验潜在结果的集合. 事件空间是通过考虑$\Omega$的子集获得的, 对于离散概率分布, $\mathcal{A}$通常是$\Omega$的幂集

> 幂集: 是一个集合所有子集构成的集合

- 概率$\mathcal{P}$: 对于每个事件$A \in \mathcal{A}$, 关联一个数P(A), 它衡量了事件发生的概率或信念程度, P(A)被称为A的概率.

## 加法规则, 乘法规则与贝叶斯公式

p(x, y)是两个随机变量x, y的联合分布, 分布p(x), p(y)是相应的边缘分布, 而p(y|x)是在给定x的条件下y的条件分布

则加法规则为:
$$
p(x) = \begin{cases}
\sum_{y\in \gamma}p(x,y) & \text{如果y是离散的}\\
\int_\gamma p(x, y)dy & \text{如果y是连续的}
\end{cases}
$$


乘法规则为:
$$
p(x, y) = p(y|x)p(x)
$$
乘法规则可以理解为, 任意两个随机变量的联合分布可以分解为另外两个分布: 一个随机变量的边缘分布p(x), 和另一个随机变量在给定第一个随机变量的条件分布p(y|x).



在机器学习和贝叶斯统计中, 经常在观察到其他随机变量的情况下, 对未观察到的(潜在的)随机变量进行推断. 假设我们对一个未观察到的随机变量x有一些先验经验p(x), 以及x与我们可以观察到的随机变量y之间的某种关系p(y|x).如果我们观察到了y, 可以使用贝叶斯定理根据观察到的y的值来得处关于x的一些结论. 贝叶斯定理
$$
\underbrace{p(x|y)}_{后验} = \frac{\overbrace{p(y|x)}^{似然度}\overbrace{p(x)}^{先验}}{\underbrace{p(y)}_{证据}}
$$
其中p(y)是边缘似然/证据
$$
p(y):= \int p(y|x)p(x)dx = \mathbb{E}_x[p(y|x)]
$$
贝叶斯定理允许我们反转由似然给出的x和y的关系, 因此, 贝叶斯定理也被称为概率逆定理.



## 汇总统计量与独立性

广为人知的汇总统计量: 均值和方差.

比较一对随机变量的两种方法: 如何判断两个随机变量是独立的 ; 以及如何计算它们之间的内积.



### 均值与协方差

均值与(协)方差通常用于描述概率分布的性质(期望值与离散程度).



#### 定义6.3 期望值

对于单变量连续随机变量$X \sim p(x)$的函数$g: \mathbb{R} \to \mathbb{R}$, 其期望值定义为:
$$
E_X[g(x)] = \int_x g(x)p(x)dx
$$
相应的, 对于离散随机变量$X\sim p(x)$的函数g, 其期望值定义为:
$$
E_X[g(x)] = \sum_{x\in X}g(x)p(x)
$$
其中X是随机变量所有可能结果的集合.



多元随机变量X被视为单变量随机变量$[X_1, \dots, X_D]^T$的有限向量. 对于多元随机变量, 我们逐元素的定义期望值.
$$
\mathbb{E}_X[g(\bold{x})] = \begin{bmatrix}\mathbb{E}_x[g(x_1)]\\\vdots\\\mathbb{E}_x[g(x_D)] \end{bmatrix} \in \mathbb{R}^D
$$

#### 定义6.4 均值

> 均值就是g取恒等函数时候的期望值

随机变量X, 其状态$x\in \mathbb{R}^D$, 的均值是一个平均值, 定义为:
$$
E_X[\bold{x}] = \begin{bmatrix} E_{x_1}[x_1]\\\vdots\\E_{x_D}[x_D] \end{bmatrix} \in \mathbb{R}^D
$$


#### 定义6.5 协方差(单变量)

> 对于两个随机变量, 我们可能希望描述它们之间的对应关系. 协方差直观地表示了随机变量之间依赖性的概念

两个单变量随机变量$X,Y\in \mathbb{R}$之间的协方差由它们各自偏离各自均值的乘积的期望值给出, 即
$$
\text{Cov}_{x,y}[x,y]:= E_{x,y}[(x - E_X[x])(y - E_Y[y])]
$$
根据期望的线性性质, 可以转换为:
$$
\text{Cov}_{x,y} [x,y] := E[xy] - E[x]E[y]
$$
变量与自身的协方差C o v[x,x]称为方差, 记作$\mathcal{V}_X[x]$

方差的平方根称为标准差, 记作$\sigma(x)$



#### 定义6.6 协方差(多变量)

如果我们考虑两个多变量随机变量X和Y, 其状态分别为$x\in \mathbb{R}^D, y\in \mathbb{R}^E$, 则X和Y之间的协方差定义为
$$
\text{Cov}_{[x,y]} = E[xy^T] - E[x]E[y]^T = \text{Cov}_{[y,x]}^T
$$


#### 定义6.7 方差

随机变量X的方差, 其状态为$x\in \mathbb{R}^D$, 均值为$\mu \in \mathbb{R}^D$, 定义为:
$$
\mathbb{V}_X[x] = \text{Cov}_{[x,x]} = 饿,D\times D 一个矩阵
$$
这个协方差矩阵是对称且半正定的, 在其对角线上, 协方差矩阵包含了边缘分布的方差.



每个随机变量的方差都会影响协方差的值, 而协方差的归一化版本被称为相关系数



#### 定义6.8 相关系数

两个随机变量X, Y之间的相关系数由
$$
corr[x,y] = \frac{Cov[x,y]}{\sqrt{V[x]V[y]}} \in [-1, 1]
$$
相关系数矩阵时标准化随机变量$x/\sigma(x)$的协方差矩阵. 换句话说, 在相关系数矩阵中, 每个随机变量都被其标准差(方差的平方根)除.



协方差(和相关系数)表明了两个随机变量之间的关系: 正相关corr[x,y]意味着当x增长时, y预期也会增长. 负相关意味着当x增加时, y会减小.



### 经验均值和协方差

对于均值, 给定一个特定的数据集, 可以获得均值的估计值, 这被称为经验均值或样本均值, 经验协方差也是如此.



#### 定义6.9 经验均值和经验协方差

经验均值向量是每个变量观测值的算术平均值, 定义为:
$$
\bar{x} := \frac{1}{N}\sum_{n=1}^Nx_n
$$
经验协方差矩阵是一个$D\times D$矩阵
$$
\Sigma := \frac{1}{N}\sum_{n=1}^N(x_n-\bar{x})(x_n-\bar{x})^T
$$


### 方差的三种表达式

