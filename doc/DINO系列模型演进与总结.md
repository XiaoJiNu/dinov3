

# **从DINO到DINOv3：自监督视觉学习的演进与新范式**

## **序言：DINOv3——迈向通用视觉基础模型的里程碑**

近年来，计算机视觉领域正在经历一场深刻的范式转移，其核心驱动力源于“基础模型”（Foundation Models）概念的兴起。这一转变标志着研究重点从为特定任务训练专门化模型，转向开发能够生成通用、可迁移视觉特征的单一、大规模模型。自监督学习（Self-Supervised Learning, SSL）在这一进程中扮演了关键角色，它使得模型能够直接从海量、无标注的原始图像数据中学习，从而摆脱了对昂贵且耗时的人工标注的依赖 1。

DINOv3的问世，不仅是其前代产品的增量式更新，更是一个重要的里程碑。它直面并解决了其前代模型在成功规模化后所暴露出的一个关键瓶颈——密集特征退化（dense feature degradation）。这一问题曾严重制约了通过延长训练周期和扩大模型规模来进一步提升性能的潜力。本报告旨在系统性地追溯DINO系列的演进脉络，深入剖析DINOv3的核心技术创新，并全面分析其在各类视觉任务中达到的SOTA（state-of-the-art）性能。报告将论证，DINOv3通过其独特的解决方案，为无需微调（fine-tuning-free）的通用视觉编码器设立了新的行业标杆 1。

---

## **第一部分：DINO系列的演进之路：从奠基到新高峰**

### **1.1 奠基之作：DINOv1与自蒸馏范式**

DINO（self-**DI**stillation with **NO** labels）的首次亮相，为自监督学习领域引入了一种新颖的视角。它将SSL问题巧妙地重构为一种无需标签的知识蒸馏过程：训练一个“学生”网络，使其输出与一个“教师”网络的输出相匹配 2。这一框架的提出，本身就代表了一种重要的理念转变。此前的自监督方法，如MoCo和SimCLR，大多依赖于显式的对比学习，即通过一个损失函数来拉近“正样本对”（同一图像的不同增广视图）的表示，同时推远“负样本对”（不同图像的视图）的表示。这种方法通常需要巨大的批次大小或复杂的内存队列（memory bank）来提供足够丰富的负样本 2。

疑问：DINO与MoCo和SimCLR的具体区别是什么？

DINOv1则另辟蹊径，其框架内不包含负样本或对比损失 2。它通过一个简单的交叉熵损失函数，让学生网络预测教师网络对同一图像不同视图的输出概率分布 6。这从根本上将SSL的目标从“区分此图像与所有其他图像”转变为“在不同数据增广下，预测该图像的一个稳定表示”。

疑问：它通过一个简单的交叉熵损失函数，让学生网络预测教师网络对同一图像不同视图的输出概率分布。具体怎么做？

为了实现这一目标，DINOv1整合了几个关键组件：

* **学生-教师架构与动量编码器**：学生和教师网络拥有相同的架构（例如Vision Transformer, ViT），但参数不同。教师网络的权重并非通过梯度下降直接更新，而是作为学生网络权重的指数移动平均（Exponential Moving Average, EMA）进行更新。这一机制被称为“动量编码器”（momentum encoder），它确保了教师网络能够提供一个比学生网络更稳定、更一致的学习目标，从而有效防止训练崩溃 3。其更新规则可表示为：  
  θt​←λθt​+(1−λ)θs​  
  其中，θt​ 和 θs​ 分别是教师和学生的参数，λ 是一个动量系数 6。  
  
* **多尺度裁剪策略（Multi-Crop Strategy）**：DINOv1采用了一种非对称的多尺度裁剪增广策略。对于每一张输入图像，它会生成两个高分辨率的“全局视图”（覆盖图像超过50%的区域）和多个低分辨率的“局部视图”（覆盖图像小于50%的区域）。关键在于，教师网络只处理全局视图，而学生网络则需要处理所有的全局和局部视图。这种设计迫使学生网络学习“从局部到全局”的对应关系，即理解一个小的局部图像块是如何构成一个完整的全局场景的，从而极大地丰富了所学特征的粒度和鲁棒性 2。  

* **模型崩溃的预防机制**：为了防止所有输出都收敛到一个平凡解（即模型崩溃），DINOv1引入了\*\*中心化（centering）**和**锐化（sharpening）\*\*两种关键技术。中心化通过给教师网络的输出添加一个偏置项（该偏置项是基于批次内特征的EMA进行更新的）来确保特征在各个维度上是零中心的，从而防止某个维度过度主导。锐化则是通过在教师网络的softmax函数中使用一个较低的温度系数，使得其输出的概率分布更加尖锐，从而防止模型输出一个无信息的均匀分布。这两者之间的精妙平衡是维持训练稳定性的核心 4。

  疑问：温度系数是什么？怎么用？

DINOv1最重要的发现是，通过这种自监督学习方式训练的ViT模型，其特征中自然地“涌现”出了关于图像语义分割的明确信息。这一点在模型最后一层自注意力图的可视化中表现得尤为明显，不同的注意力头会自动关注到图像中不同的语义对象边界，而这种特性在监督学习训练的模型中并不突出 3。

### **1.2 规模化探索：DINOv2的数据策展与双重目标**

DINOv1在ImageNet等标准数据集上的成功，自然引出了一个问题：如何将其扩展到更大规模、更多样化、未经整理的网络数据上？早期的尝试表明，简单地将模型在随机的网络图像上进行训练，会导致所学特征的质量显著下降 12。这促使DINOv2团队将研究重点转向了两个核心方向：大规模数据的自动化策展和学习目标的增强。

DINOv2最显著的贡献之一是其强大的通用性，它需要在全局语义理解和细粒度局部细节捕捉两个方面都表现出色。为了实现这一目标，DINOv2将单一的学习目标扩展为一个双重目标框架，这成为其方法论的核心。

* **自动化的数据策展流程**：DINOv2不再依赖于现成的学术数据集，而是构建了一套自动化流程来创建一个名为LVD-142M的大规模、高质量数据集。该流程包括：  
  1. **数据池与种子集**：从一个包含数十亿张图片的未整理网络数据池开始，并选用一系列高质量的已整理数据集（如ImageNet-22k）作为“种子集”。  
  2. **相似性检索**：利用图像特征，从数据池中检索出与种子集中的图像在视觉上相似的图片。  
  3. **去重与过滤**：对检索到的图像进行严格的去重和内容过滤（例如，NSFW过滤），以确保数据的多样性和质量 12。  
* **双重学习目标**：DINOv2对损失函数进行了关键的扩展，将全局和局部学习信号明确地结合起来。  
  1. **全局图像级损失（DINO Loss）**：沿用DINOv1的核心思想，通过比较学生和教师网络对全局视图的\`\` token输出来学习整体的、高层次的图像表示。  
  2. **局部掩码-补丁级损失（iBOT Loss）**：引入了iBOT 14 的思想，这是一种在自蒸馏框架下的掩码图像建模（Masked Image Modeling）。具体而言，在送入学生网络之前，会随机遮盖（mask）掉一部分图像补丁（patch）。然后，训练学生网络利用可见的补丁信息，来预测被遮盖区域对应于教师网络输出的特征。这个过程强迫模型学习到更加精细的局部纹理和结构信息 14。

通过融合这两种损失，DINOv2构建了一个既能理解图像整体语义（DINO Loss），又能捕捉像素级细节（iBOT Loss）的强大模型。这种设计使得DINOv2的特征无需微调即可在分类等全局任务和分割等密集任务上都表现出色，奠定了其作为通用视觉特征提取器的地位。此外，DINOv2还引入了其他技术优化，如使用源自SwAV的Sinkhorn-Kpp算法进行中心化，以及引入KoLeo正则化器来促进特征空间分布的均匀性，进一步提升了训练的稳定性和效率 12。

### **1.3 演进脉络对比：DINO v1, v2, v3 核心异同分析**

下表系统地总结了从DINOv1到DINOv3的演进脉络，直观地展示了其在核心理念、技术实现和目标问题上的逐步深化与变革。这一演进过程清晰地反映了自监督视觉学习领域从可行性验证到规模化应用，再到解决规模化所带来新挑战的完整研发闭环。

**表1: DINO系列演进对比**

| 特征维度 | DINOv1 | DINOv2 | DINOv3 |
| :---- | :---- | :---- | :---- |
| **核心学习目标** | 自蒸馏 (Self-Distillation) | 全局+局部双重目标 (Dual Global+Local) | 双重目标 \+ Gram锚定正则化 (Dual \+ Gram Anchoring) |
| **具体损失函数** | DINO Loss (基于\`\` token的交叉熵) | DINO Loss \+ iBOT Loss (掩码补丁预测) | DINO Loss \+ iBOT Loss \+ **Gram Loss** |
| **数据策略** | ImageNet (1.3M) | LVD-142M (142M, 自动筛选策划) | LVD-1689M (1.69B, 层次化聚类筛选) |
| **核心创新** | 证明SSL+ViT的协同效应，发现涌现的分割属性 | 大规模数据策展流程，融合全局与局部学习目标 | **Gram锚定 (Gram Anchoring)** 解决密集特征退化问题 |
| **主要解决的问题** | 提升ViT在SSL下的性能，探索其新特性 | 创建无需微调的通用视觉特征 | 解决超大规模、长周期训练下的密集特征质量下降问题 |
| **特征质量与特点** | 优异的全局特征，初步展现良好的局部语义 | 强大的通用特征，在全局和密集任务上均表现出色 | **SOTA级别的密集特征**，在像素级任务上表现卓越，同时保持强大的全局性能 |

---

## **第二部分：深度解析DINOv3：核心技术贡献**

DINOv3的研发是建立在DINOv2成功规模化的基础之上，旨在突破更大规模训练所带来的新瓶颈。其核心技术贡献围绕着“如何更有效地进行超大规模训练”以及“如何解决由此产生的密集特征退化问题”展开。

### **2.1 规模化的艺术：数据、模型与训练策略**

DINOv3将“规模化”推向了新的高度，这不仅体现在数据量和模型参数的增长，更体现在一系列为适应这种规模而精心设计的策略上。

* **数据扩展与精细策展**：DINOv3的训练数据集LVD-1689M达到了16.89亿张图像的规模。与DINOv2基于检索的策展方式不同，DINOv3采用了一种更为精细的**分层k-means聚类**方法。这种方法能够确保对网络上出现的各种视觉概念进行更均衡的覆盖，避免数据集中常见概念的过度饱和。同时，该方法结合了检索式策展和高质量公共数据集（如ImageNet），并设计了一种混合采样策略：在训练过程中，有10%的批次是完全由高质量的ImageNet数据组成的同质批次，其余则是混合批次。这种策略旨在同时提升模型的泛化能力和在标准基准上的性能 1。  

* **模型架构的现代化升级**：DINOv3的主力模型是一个拥有近70亿参数的ViT-7B。其架构上的关键升级包括：  
  * **更大的嵌入维度**：将特征维度提升至4096，以增强模型的表示能力。  
  
  * **旋转位置编码（RoPE）**：放弃了传统的可学习位置编码，转而采用RoPE。RoPE通过在注意力计算中旋转键（key）和查询（query）向量来编码相对位置信息，这使得模型能够更好地泛化到训练中未见过的图像分辨率和长宽比。  
  
    疑问：RoPE具体是如何实现的？
  
  * **RoPE-box Jittering**：为了进一步增强鲁棒性，DINOv3在训练中对RoPE的坐标系进行随机缩放（抖动），这使得模型对输入尺寸的变化更加不敏感 1。  
  
* **适应超长周期的训练策略**：在动辄百万次迭代的超大规模训练中，传统的余弦退火等学习率调度策略变得难以应用，因为它们要求预先知道总的训练步数。为了解决这个问题，DINOv3采取了一种极为务实的做法：**使用恒定的超参数**。即在初始的预热阶段之后，学习率、权重衰减等关键超参数在整个训练过程中保持不变。这不仅简化了超参数调优，也使得训练可以根据下游任务性能的反馈而无限期地进行下去，直到性能饱和为止 1。

### **2.2 Gram锚定：稳定密集特征的关键创新**

DINOv3最核心、最具突破性的技术贡献是**Gram锚定（Gram Anchoring）**，它直接解决了大规模SSL训练中的密集特征退化问题。

* **问题的根源：密集特征退化**：DINOv3团队观察到一个棘手的现象：在对大型模型（如ViT-Large及以上）进行长时间训练时，虽然模型的全局任务性能（如ImageNet分类准确率）会持续提升，但其密集任务性能（如语义分割的mIoU）在达到一个峰值后便开始显著下降。通过可视化分析发现，随着训练的进行，模型输出的特征图变得越来越“嘈杂”，补丁（patch）级别的特征一致性丧失。具体表现为，一个补丁与图像中许多语义不相关的遥远补丁产生了过高的相似度。其深层原因在于，随着模型对全局语义的理解加深，\`\` token的影响力逐渐增强，导致各个局部补丁的特征趋同，失去了其独有的局部性和区分度 1。  
* **解决方案：Gram锚定**：Gram锚定的设计思想极为精妙，它体现了一种“解耦正则化”的哲学。它认识到，学习高级语义（优化全局特征）和维持低级空间一致性（优化密集特征）这两个目标在极端规模下可能存在冲突。与其寻找一个能完美平衡两者的单一损失函数，DINOv3引入了一个专门针对密集特征结构的独立正则化项。  
  1. **核心理念**：该方法不直接对特征向量本身进行约束，而是约束特征向量之间的**相似性结构**。这种结构由**Gram矩阵**（XXT）来捕捉，该矩阵包含了图像中所有补丁特征之间的两两内积（即相似度）1。  
  2. **“Gram教师”**：DINOv3引入了一个“Gram教师”的概念，它实际上是模型在训练早期（例如第20万次迭代时）的一个检查点，此时模型的密集特征质量正处于巅峰状态。训练目标是让当前“学生”模型的Gram矩阵，去逼近这个高质量“Gram教师”的Gram矩阵 1。  
  3. **损失函数**：通过最小化学生和Gram教师的Gram矩阵之间的弗罗贝尼乌斯范数来实现：  
     LGram​=∣∣XS​XST​−XG​XGT​∣∣F2​  
     这个损失函数允许学生模型的特征空间继续演化和提升（以适应DINO和iBOT损失所驱动的语义学习），只要补丁之间的相对相似性关系能够与高质量的Gram教师保持一致即可。它像一个“结构脚手架”，在不冻结特征本身的前提下，维持了特征图的几何一致性，从而解决了特征退化问题，同时不影响全局任务性能的持续提升 1。  
* **利用高分辨率特征进行增强**：为了进一步提升效果，DINOv3还采用了一个技巧：将更高分辨率的图像输入给Gram教师，然后将其输出的特征图下采样到与学生匹配的尺寸。由于高分辨率输入能产生更精细、更连贯的特征图，下采样后的特征图所计算出的Gram矩阵便成了一个质量更高、更平滑的学习目标，从而进一步改善了学生模型的密集特征质量 1。

### **2.3 通用性赋能：高分辨率适应、模型蒸馏与文本对齐**

在核心训练之后，DINOv3通过一系列后处理步骤，极大地增强了模型的实用性和通用性。

* **高分辨率适应**：为了让模型能够处理各种尺寸的输入图像，DINOv3在主训练后增加了一个短暂的高分辨率适应阶段。在此阶段，模型会在混合的、更高分辨率的图像上进行少量迭代的训练。值得注意的是，**Gram锚定**在这一阶段同样至关重要，它能有效防止模型在高分辨率数据上训练时出现密集特征的退化。经过这一步骤，最终的模型能够稳健地处理高达4K分辨率的图像，并输出高质量的特征图 1。  
* **高效的多学生蒸馏**：强大的ViT-7B模型虽然性能卓越，但其巨大的计算开销限制了其在实际场景中的应用。为此，DINOv3设计了一套创新且高效的**多学生并行蒸馏流水线**。其核心思想是，在一次迭代中，昂贵的7B教师模型只需进行一次前向传播，其计算结果（即输出的特征）会被广播并共享给多个正在并行训练的、不同规模的学生模型（包括ViT和ConvNeXt系列）。这种方法极大地摊销了教师模型的计算成本，使得能够以较低的代价将大模型的知识高效地迁移到一个完整的、覆盖不同计算预算的模型家族中 1。  
* **文本对齐（dino.txt）**：为了赋予DINOv3多模态理解的能力，团队采用了dino.txt方案。该方案在保持DINOv3视觉骨干网络**冻结**的情况下，训练一个文本编码器来与之对齐。其关键技术在于，文本编码器的学习目标不仅要对齐图像的全局\`\` token，还要对齐所有局部补丁特征的平均池化表示。这种“全局+局部”的双重对齐方式，使得文本能够与图像的整体语义和局部细节都建立联系，从而让模型在开放词汇的分割等密集视觉-语言任务上取得了优异的性能 1。

---

## **第三部分：性能验证：DINOv3在多元视觉任务中的卓越表现**

DINOv3的性能评估覆盖了从底层的特征质量分析到上层的系统级应用，并拓展到了非通用的专业领域，全面验证了其作为新一代视觉基础模型的强大实力。

### **3.1 无与伦比的密集特征：分割、深度与3D理解**

DINOv3最引人注目的成就在于其卓越的密集特征质量，这在多项像素级和几何理解任务中得到了验证。

* **定性分析**：首先，从特征图的PCA可视化（图13）中可以直观地看到DINOv3的优势。相较于DINOv2、SigLIP 2等强有力的竞争对手，DINOv3的特征图在视觉上更加清晰、锐利，噪声更少，并且展现出卓越的语义一致性，能够清晰地勾勒出物体的轮廓和不同区域 1。  
* **定量优势**：在定量的线性探测（linear probing）评估中，即在冻结的骨干网络之上只训练一个线性分类器，DINOv3展现了压倒性的优势：  
  * **语义分割**：在ADE20k数据集上，DINOv3取得了55.9的mIoU，比之前的自监督SOTA模型高出超过6个点，甚至比那些利用了全监督分割模型SAM进行蒸馏的聚合模型（如AM-RADIOv2.5，mIoU为53.0）还要高 1。  
  * **深度估计**：在NYUv2和KITTI数据集上，DINOv3同样取得了最佳性能，大幅超越所有基线模型，为冻结骨干网络下的单目深度估计设立了新标准 1。  
  * **3D对应关系**：在评估多视角一致性的NAVI（几何对应）和SPair（语义对应）数据集上，DINOv3的召回率均排名第一，显示出其特征蕴含了强大的3D几何感知能力 1。  
  * **视频分割跟踪**：在DAVIS 2017等视频数据集上，DINOv3取得了惊人的83.3 J\&F得分，比DINOv2高出6.7个点，证明了其特征具有出色的时间连续性 1。  
* **在完整系统中的表现**：当DINOv3作为冻结骨干网络被集成到更复杂的下游系统（例如，结合Mask2Former进行语义分割，或结合DPT进行深度估计）时，它同样能够驱动系统达到SOTA性能，并且通常只需要训练比竞争方案更少的参数 1。

### **3.2 稳健通用的全局描述子：分类与实例识别**

尽管DINOv3的开发重点在于提升密集特征，但它在传统的全局图像任务上也表现出了极强的竞争力，成功挑战了以往由弱监督（图像-文本对）方法主导的领域。

* **图像分类**：在线性探测评估中，DINOv3是首个在ImageNet及其各种分布外（OOD）变体数据集上，性能达到与顶尖的弱监督模型（如SigLIP 2, PE）以及超大规模全监督模型（如ViT-22B）相媲美水平的自监督模型。这表明纯粹基于图像的自监督学习，在经过充分的规模化和精细的算法设计后，同样可以学习到强大的、可泛化的语义表示 1。  
* **细粒度与实例识别**：在需要区分细微差别的任务上，DINOv3同样表现出色。它在挑战性的iNaturalist 2021细粒度物种分类任务上取得了89.8%的最高准确率。在牛津和巴黎地标检索等实例识别基准测试中，DINOv3也以显著优势超越了包括DINOv2在内的所有先前模型，证明了其特征具有极高的辨识度 1。

### **3.3 跨域应用：DINOv3在地球观测领域的突破**

为了验证DINOv3学习框架的通用性，研究团队进行了一项极具说服力的跨域实验：将完全相同的自监督学习配方应用于一个由4.93亿张卫星影像组成的庞大数据集（SAT-493M）1。这一实验揭示了关于基础模型泛化能力的一个深刻二元性。

* **领域预训练的优势**：实验结果表明，在需要精确物理量测的任务上，领域内预训练至关重要。例如，在冠层高度估计这一回归任务中，使用卫星数据预训练的“DINOv3 Sat”模型表现最佳。这是因为该任务与卫星传感器的特定成像物理特性（如光照反射、大气噪声）紧密相关，在领域内数据上进行预训练能够让模型更好地学习到这些物理先验，从而做出更准确的度量预测 1。  
* **通用预训练的泛化力**：然而，一个出人意料的发现是，在许多语义理解任务上（如土地覆盖分类、地理目标检测），原始的、在通用网络图像上预训练的“DINOv3 Web”模型，其性能竟然与甚至**超越**了在卫星数据上训练的“DINOv3 Sat”模型 1。这背后的逻辑是，像“建筑”、“水体”或“森林”这样的语义概念，其视觉表现形式是极其多样的。在16.89亿张通用网络图像上训练的DINOv3 Web模型，接触过的相关概念的实例在光照、视角、纹理和上下文方面远比4.93亿张俯视视角的卫星图像要丰富得多。这种海量的、多样化的视觉经验构建了一个更强大、更具泛化能力的语义特征空间，使其在面对新的、未见过的领域（如卫星图像）中的语义任务时，依然能够表现出色。

这一发现对基础模型的设计具有重要启示：不存在一个“放之四海而皆准”的最优预训练数据集。最佳的数据选择取决于下游任务的性质——是更依赖于通用的、抽象的语义理解，还是更依赖于领域特定的、物理接地的度量预测。而DINOv3框架的真正强大之处在于，它提供了一套足够鲁棒和通用的学习方法，能够成功地应用于截然不同的数据域，并在这两种类型的任务上都取得SOTA级别的成果。

---

## **第四部分：总结与展望**

DINOv3的发布是自监督视觉学习领域的一个重要里程碑。它系统性地解决了随着模型和数据规模扩大而出现的核心技术瓶颈，将纯视觉自监督方法推向了新的性能高峰。

**核心贡献的综合回顾**：

1. **解决了密集特征退化问题**：通过创新的**Gram锚定**机制，DINOv3成功地在不牺牲全局特征学习的前提下，稳定并提升了模型在超长训练周期下的密集特征质量。  
2. **实现了SSL的超大规模化**：通过精细的数据策展、现代化的模型架构以及务实的训练策略，DINOv3成功地将自监督学习扩展到了近70亿参数的模型和超过16亿张图像的数据集上。  
3. **树立了密集任务的新标杆**：DINOv3及其模型家族在语义分割、深度估计、3D理解和视频分析等一系列密集视觉任务上，以无需微调的冻结骨干网络形式，全面超越了以往的自监督、弱监督乃至部分专用模型，设立了新的性能基准 1。

未来影响与展望：  
DINOv3的成功极大地缩小了自监督学习与依赖图文对的弱监督学习之间的性能差距，尤其是在密集任务上实现了反超。这有力地证明了，只要有恰当的算法设计和足够的计算资源，纯视觉信号本身就足以学习到与多模态信号相媲美甚至更优的通用视觉表示。  
这一成就为未来在缺乏高质量文本配对数据的领域（如医学影像、科学计算可视化、工业检测等）训练强大的基础模型开辟了新的道路。DINOv3所代表的这条技术路线——即系统性地识别并解决规模化过程中的瓶颈——为未来构建更大、更强的通用视觉编码器提供了宝贵的经验和坚实的基础。它标志着自监督学习已经从一个充满潜力的研究方向，成长为能够真正构建SOTA基础模型的核心技术。

#### **Works cited**

1. DINOv3.pdf  
2. Emerging Properties in Self-Supervised Vision Transformers \- CVF Open Access, accessed September 12, 2025, [https://openaccess.thecvf.com/content/ICCV2021/papers/Caron\_Emerging\_Properties\_in\_Self-Supervised\_Vision\_Transformers\_ICCV\_2021\_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf)  
3. \[2104.14294\] Emerging Properties in Self-Supervised Vision ..., accessed September 12, 2025, [https://ar5iv.labs.arxiv.org/html/2104.14294](https://ar5iv.labs.arxiv.org/html/2104.14294)  
4. Emerging Properties in Self-Supervised Vision Transformers (DINO) — Paper Summary | by Anuj Dutt | Medium, accessed September 12, 2025, [https://medium.com/@anuj.dutt9/emerging-properties-in-self-supervised-vision-transformers-dino-paper-summary-4c7a6ed68161](https://medium.com/@anuj.dutt9/emerging-properties-in-self-supervised-vision-transformers-dino-paper-summary-4c7a6ed68161)  
5. DINO: Self-Supervised Vision Transformers and Their Emerging Properties | by Jim Canary, accessed September 12, 2025, [https://medium.com/@jimcanary/dino-self-supervised-vision-transformers-and-their-emerging-properties-7f9e5f4adac4](https://medium.com/@jimcanary/dino-self-supervised-vision-transformers-and-their-emerging-properties-7f9e5f4adac4)  
6. DINO \- Emerging Properties in Self-Supervised Vision Transformers \- AIdventure, accessed September 12, 2025, [https://aidventure.es/blog/dino/](https://aidventure.es/blog/dino/)  
7. DINO \- A Foundation Model for Computer Vision | Towards Data Science, accessed September 12, 2025, [https://towardsdatascience.com/dino-a-foundation-model-for-computer-vision-4cb08e821b18/](https://towardsdatascience.com/dino-a-foundation-model-for-computer-vision-4cb08e821b18/)  
8. DINO | self\_supervised, accessed September 12, 2025, [https://keremturgutlu.github.io/self\_supervised/15%20-%20dino.html](https://keremturgutlu.github.io/self_supervised/15%20-%20dino.html)  
9. SSL with Vision Transformers | Rohit Bandaru, accessed September 12, 2025, [https://rohitbandaru.github.io/blog/SSL-with-Vision-Transformers/](https://rohitbandaru.github.io/blog/SSL-with-Vision-Transformers/)  
10. Paper Walkthrough: DINO \- Erik Storrs, accessed September 12, 2025, [https://storrs.io/dino/](https://storrs.io/dino/)  
11. Emerging Properties in Self-Supervised Vision Transformers \- Hugging Face, accessed September 12, 2025, [https://huggingface.co/papers/2104.14294](https://huggingface.co/papers/2104.14294)  
12. DINOv2: Learning Robust Visual Features without Supervision \- arXiv, accessed September 12, 2025, [https://arxiv.org/html/2304.07193v2](https://arxiv.org/html/2304.07193v2)  
13. DINOv2: Learning Robust Visual Features without Supervision \- arXiv, accessed September 12, 2025, [https://arxiv.org/pdf/2304.07193](https://arxiv.org/pdf/2304.07193)  
14. Understanding DINOv2: Engineer's Deep Dive \- Lightly, accessed September 12, 2025, [https://www.lightly.ai/blog/dinov2](https://www.lightly.ai/blog/dinov2)  
15. DINOv2 from Meta AI: Data pipeline, model training and results explained \- YouTube, accessed September 12, 2025, [https://www.youtube.com/watch?v=RZEkdOc3szU](https://www.youtube.com/watch?v=RZEkdOc3szU)  
16. Harness DINOv2 Embeddings for Accurate Image Classification | by Lihi Gur Arie, PhD, accessed September 12, 2025, [https://pub.towardsai.net/harness-dinov2-embeddings-for-accurate-image-classification-f102dfd35c51](https://pub.towardsai.net/harness-dinov2-embeddings-for-accurate-image-classification-f102dfd35c51)