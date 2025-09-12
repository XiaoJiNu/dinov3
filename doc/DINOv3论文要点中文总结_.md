

# **DINOv3技术报告深度解读：无监督学习的视觉基础模型新里程碑**

## **第一部分：摘要与核心洞察**

### **核心成果概述：通用视觉编码器的新定义**

DINOv3代表了自监督学习（SSL）领域的一个重大飞跃，它通过简单但高度有效的策略，实现了一个通用视觉基础模型的宏大愿景 1。该模型能够在不进行微调的情况下，在广泛的任务和领域中超越许多专业化的最先进模型 1。DINOv3的核心突破在于，它不仅将模型和数据集的规模扩展到了前所未有的水平，更通过引入创新的技术，解决了长期困扰大规模训练的一个关键瓶颈——致密特征图（dense feature maps）的性能退化问题 1。

这一成就使得DINOv3能够作为一个“开箱即用”的统一视觉骨干网络，为下游任务提供高质量的特征表示 6。这意味着研究人员和开发者无需为特定任务重新训练或微调整个模型，只需在其上构建轻量级的适配器（adapters）或线性分类器，即可获得卓越的性能 3。这种模式极大地降低了开发和部署的成本，并使计算资源得以复用和共享，从而为计算机视觉的实际应用和创新提供了强大的推动力 3。

### **DINOv3的三大支柱**

DINOv3的突破性表现是建立在三个核心技术支柱之上的，它们共同确保了模型在规模化、稳定性和通用性上的卓越表现。

1. **规模化突破：** 论文将自监督视觉模型成功扩展到了一个前所未有的规模。其核心骨干网络参数量达到了67亿（约7B），并基于一个包含17亿张图像的精心策划数据集（LVD-1689M）进行训练 4。这种规模化策略借鉴了大型语言模型（LLM）的成功经验，旨在通过扩大模型和数据的容量，解锁更强大的泛化能力和“涌现能力” 3。  
2. **稳定器：Gram Anchoring：** 论文引入了一种名为Gram Anchoring的新型正则化策略，有效解决了困扰大规模、长时间训练中致密特征图退化的已知问题 4。通过确保补丁（patch）特征之间的相对相似性结构保持一致，该方法能够在不牺牲高层语义理解的情况下，维持低层局部特征的质量和稳定性 6。这一创新使得DINOv3能够同时在全局和局部任务上都实现卓越性能 1。  
3. **多学生蒸馏：** 为了将70亿参数教师模型的强大知识转化为可实际应用的工具，论文采用了一种高效的“单教师多学生”（single-teacher multiple-students）蒸馏流程 1。该方法能将大型教师模型的知识高效地传递给一系列更小、更实用的模型变体，包括ViT-S、ViT-B、ViT-L以及ConvNeXt系列 1。这为应对不同的计算资源限制和部署场景提供了灵活的解决方案 1。

### **规模、退化与正则化的深层逻辑**

DINOv3的成功并非偶然，其背后是对自监督学习训练范式局限性的深刻理解。论文明确指出，当模型规模超过ViT-Large（3亿参数）并进行长时间训练时，其致密特征图的性能会逐渐下降 1。这一现象的本质是模型内部优化目标的冲突：一方面，全局损失（如DINO损失）鼓励模型学习图像级别的宏观语义，使得所有补丁令牌（patch tokens）都与代表全局信息的CLS令牌趋同；另一方面，局部损失（如iBOT损失）则要求补丁特征保留其局部特异性和几何细节 1。在长期训练中，前一个目标会占据主导地位，导致补丁特征失去其局部性，从而在分割、深度估计等任务上表现不佳 8。

Gram Anchoring的提出正是为了解决这一核心矛盾。它的巧妙之处在于，它不直接强制学生模型的补丁特征与教师模型完全一致，而是通过正则化Gram矩阵——一个编码了所有补丁特征之间两两点积的矩阵——来约束特征的**相对相似性结构** 1。这种方法允许局部特征自由演化，只要它们与其他特征的相对关系保持不变，就如同在保持一张照片“风格”不变的前提下，可以改变其具体内容。这是一种更高层次的知识蒸馏，既能“修复”退化的特征图，又不影响模型对高级语义的持续学习，从而提供了一个优雅而强大的解决方案。

## **第二部分：引言：背景、挑战与核心目标**

### **自监督学习的崛起与DINOv2的成功**

基础模型已经成为现代计算机视觉（CV）的核心构建模块，它们通过一个统一、可复用的模型实现了跨任务和领域的广泛泛化 1。自监督学习（SSL）是一种强大的方法，它通过直接从原始像素数据中学习并利用图像中模式的自然共现，来训练此类基础模型，从而摆脱了对昂贵人工标注的依赖 1。这一范式尤其适用于训练大规模的视觉编码器，因为可用的原始训练数据几乎是无限的 1。

DINOv2是这一发展路径上的一个重要里程碑。该模型在图像理解任务中取得了令人印象深刻的成果，并且首次在某些任务上达到甚至超越了CLIP等弱监督预训练模型 1。DINOv2的成功证明了SSL模型的强大泛化能力，它们对输入分布偏移具有鲁棒性，能够提供强大的全局和局部特征，并生成有助于物理场景理解的丰富嵌入 1。

### **大规模训练的三大挑战**

尽管SSL具有巨大的潜力，即能够利用海量非受限数据来训练任意规模和能力的模型，但在大规模实践中仍然面临着严峻的挑战。论文明确指出了几个亟待解决的问题 1：

1. **数据选择的模糊性：** 从未标注的海量数据集中筛选出对模型训练真正有用的数据，其方法尚不明确。  
2. **模型稳定性：** 在训练过程中，模型可能会变得不稳定甚至崩溃。虽然DINOv2中的一些启发式方法可以缓解这些问题，但当模型规模进一步扩大时，新的不稳定问题随之出现。  
3. **致密特征图的退化：** 这是论文重点关注的、未解决的关键问题。在早期训练后，模型致密特征的性能会逐渐下降，这一现象在参数量超过ViT-Large（3亿）的长期训练中尤为明显，严重影响了模型的实用性 1。

### **DINOv3的宏大目标**

为了解决上述挑战，DINOv3被提出，旨在推动大规模SSL训练的发展 1。该研究的核心目标包括：

* **训练一个通用基础模型：** 训练一个在任务和领域之间都具有高度通用性的基础模型。  
* **改进密集特征：** 解决现有SSL模型在致密特征图方面的不足，通过优化使其在密集任务上表现卓越。  
* **发布模型家族：** 提供一系列可即时使用的模型，以满足多样化的部署需求 1。

DINOv3旨在证明，一个冻结的SSL骨干网络可以作为通用的视觉编码器，在具有挑战性的下游任务上取得最先进的性能，超越依赖元数据的监督和预训练策略 3。

## **第三部分：规模化训练的核心策略**

### **数据准备：LVD-1689M数据集的构建**

大规模数据是大型基础模型成功的关键因素之一 1。然而，简单地增加训练数据量并不一定能带来更高的模型质量 1。因此，DINOv3采用了精心的混合式数据精选和准备流程，以在数据的通用性和任务相关性之间取得平衡 1。

该数据集的构建始于一个包含约170亿张Instagram公开图片的大型数据池 1。从中，论文通过两种互补的方法构建了最终的大规模预训练数据集（LVD-1689M）1：

1. **聚类精选（Clustering）：** 利用DINOv2模型学习到的嵌入，通过分层k-means聚类方法，筛选出一个16.89亿张图像的子集 1。这种方法确保了数据集能够平衡地覆盖网络上出现的各种视觉概念，从而提升模型的泛化能力 1。  
2. **检索精选（Retrieval）：** 类似于DINOv2的策略，通过检索与特定种子数据集（如ImageNet-1k）相似的图像，构建了一个与常见下游任务高度相关的子集 1。

在训练过程中，论文采用了混合采样策略。每批次数据中，10%的批次专门来自ImageNet-1k数据集，其余90%则混合了所有其他精选数据 1。通过一项数据消融实验，论文证明了这种混合方法在各项基准上均能取得最佳综合性能，优于仅使用单一精选或原始数据的方法 1。

### **模型架构与训练配方**

为了实现前所未有的规模化，DINOv3在模型架构和训练配方上进行了重要升级 1。

1. **架构升级：ViT-7B**  
   * DINOv3的主模型是一个参数量高达67亿（7B）的Vision Transformer（ViT），相比于DINOv2的ViT-g模型（11亿参数）有了显著的提升 1。该模型深度为40层，并增加了嵌入维度、前馈网络（FFN）和注意力头数量，以捕获更复杂的视觉关系 1。  
   * 在位置嵌入方面，论文采用了**旋转位置嵌入（ROPE）**，并通过坐标抖动（jittering）来增强模型对不同分辨率、尺度和长宽比的鲁棒性，使其能够无缝处理从256到4096像素的各种输入 1。  
   * 为了解决高范数补丁异常值问题，论文沿用了DINOv2中的**寄存器令牌（Register tokens）**，这些特殊的令牌能有效吸收模型内部的全局信息交流，从而保持补丁特征的稳定性 1。  
   
2. **训练配方：恒定超参数与混合损失**  
   * 与DINOv2中复杂的余弦调度（cosine schedules）不同，DINOv3采用了恒定的学习率、权重衰减和教师EMA动量 1。这种简化使得模型可以根据性能表现无限期地进行训练，从而摆脱了预先设定优化周期的限制 1。  
   
   * 模型的学习目标是一个混合损失函数：  
     * LPre​=LDINO​+LiBOT​+0.1∗LKoLeo​  
     
     * 其中，$L_{{DINO}}$是基于CLS令牌的图像级蒸馏损失，用于学习全局语义；$L_{iBOT}$是基于补丁令牌的局部重建损失，用于学习局部细节；$L_{{KoLeo}}$是一种正则化损失，用于鼓励批次内特征在空间中均匀分布，从而防止模型崩溃 1。
     
       **疑问：损失函数具体是如何计算的？每个函数的具体公式是什么？**

### **数据、架构与训练配方之间的微妙关系**

DINOv3的成功证明了训练超大规模视觉模型是一项多层面协同优化的工程，而非单一因素的胜利。当模型规模和训练数据量都达到一定阈值时，一些在小规模下需要复杂技巧才能解决的问题会以新的形式出现。例如，DINOv2的启发式方法虽然能缓解部分不稳定性，但在70亿参数的规模下仍然不够 1。

DINOv3通过一个系统性的解决方案来应对：精心策划的数据集确保了输入的质量和多样性，而像ROPE和寄存器令牌这样的架构升级则为模型提供了**处理大规模、高分辨率数据的结构性保障** 1。正因为模型和数据已经足够强大和稳定，简单的恒定学习率策略才得以奏效，不再需要复杂的调度来引导训练。这种多因素的协同作用是DINOv3实现突破性性能的关键。

## **第四部分：致密特征的守护者：Gram Anchoring机制**

为了充分利用大规模训练的优势，论文旨在让模型进行长时间的训练 1。然而，正如引言所指出的，随着训练周期的延长，致密特征图的性能会显著下降 1。DINOv3的Gram Anchoring策略正是为此而生，它是一种全新的训练阶段，能够有效缓解这一问题。

### **深入解析：密集特征退化问题**

在长时间训练中，一个显著的现象是模型在全局任务（如图像分类）上的性能单调提升，但在密集任务（如语义分割）上的性能却在达到峰值后开始下降 1。这表明，模型的优化目标在高层语义理解与低层局部一致性之间产生了冲突 1。

通过对特征图的余弦相似度进行可视化，论文发现了这一退化的根本原因：随着训练的进行，本应具有局部特异性的补丁特征变得“不干净”且失去局部性 1。例如，一个代表花瓣的补丁特征开始与不相关的草地补丁表现出高相似度 8。这直接导致了密集任务性能的下降，因为这些任务需要精确的局部特征来支持像素级别的预测 6。

Gram Anchoring的核心哲学在于，它不直接强制学生模型的补丁特征与教师模型完全一致，而是通过正则化**Gram矩阵**（Gram matrix）来约束特征的**相对相似性结构** 1。Gram矩阵编码了所有补丁特征之间的两两点积，本质上是特征图的“风格”或“结构”表示 5。这一方法允许局部特征自由演化，只要它们与其他特征的相对关系保持不变。这是一种更高层次的“知识蒸馏”，它能够“修复”退化的特征图，同时不影响模型对高级语义的持续学习 1。

### **Gram Anchoring原理与效果**

Gram Anchoring的核心工作机制是引入一个额外的损失函数 L\_Gram 1。该损失旨在最小化学生模型（Student）的Gram矩阵与一个“Gram教师”（Gram Teacher）的Gram矩阵之间的弗罗贝尼乌斯范数（Frobenius norm） 1。

* **“Gram教师”的选择：** Gram教师并非固定的，而是取自模型早期训练阶段的一个检查点，因为此时的特征图质量最佳 1。Gram教师会每隔10k次迭代更新一次，以保持指导的有效性 1。  
* **高分辨率Gram Anchoring：** 论文还引入了高分辨率Gram Anchoring，即在计算Gram矩阵时，使用更高分辨率的教师模型特征图，再将其下采样到学生模型的尺寸 1。这进一步提升了密集特征的平滑性和一致性 1。

定量实验表明，采用Gram Anchoring后，DINOv3在语义分割（ADE20k）和深度估计（NYUv2）等密集任务上的性能立竿见影地得到提升 1。高分辨率Gram Anchoring则在此基础上带来了额外的性能增益，例如在ADE20k上带来了额外的2个mIoU提升 1。

## **第五部分：后处理策略：从7B到模型家族的演进**

DINOv3的强大不仅在于其大规模预训练，更在于其一系列旨在提升实用性和灵活性的后处理策略。这些策略共同构建了一个完整的“DINOv3生态系统” 7。

### **高分辨率适应性训练**

尽管DINOv3的核心架构（如ROPE）使其能够原生支持可变分辨率，但为了在更高分辨率下实现最佳性能，论文引入了一个专门的后训练阶段 1。在主要训练之后，模型会进行1万次额外的迭代训练，使用混合分辨率的图像对（例如，全局裁剪大小为512/768，局部裁剪大小从112到336），以适应更丰富的空间信息 1。

在这一阶段，Gram Anchoring再次被证明是至关重要的，它确保了模型在高分辨率下也能维持特征图的一致性 1。实验结果显示，这一短暂而有针对性的训练步骤显著提升了模型在密集任务上的表现，并使其能够在4K等超高分辨率下保持稳定和语义一致的特征图 1。

### **知识蒸馏与模型家族构建**

70亿参数的模型虽然强大，但在推理和部署上并不实用 3。为了解决这一挑战，DINOv3采用了高效的“单教师多学生”蒸馏流程，将70亿参数教师模型的知识高效地转移到一系列更小、更实用的学生模型上 1。

该方法通过一个高效的并行蒸馏管线来实现，该管线允许多个学生模型同时训练，并通过共享教师模型的推理结果来节省计算资源 1。通过调整每个学生模型分配到的GPU数量，该管线能够最小化闲置时间，从而实现高效的知识传递 1。

最终，该过程构建了一个全面的模型家族，包括：

* **ViT系列：** 从紧凑的ViT-S到高性能的ViT-H+。论文指出，ViT-H+模型（0.84B参数）的性能已经非常接近其7B教师模型，这证明了知识蒸馏的有效性 1。  
* **ConvNeXt系列：** DINOv3还首次实现了知识从ViT到ConvNeXt的跨架构蒸馏，发布了CNX-T、S、B、L等变体，为资源受限的部署场景提供了高效的卷积网络选择 1。

### **与文本对齐：dino.txt**

DINOv3的后处理策略还包括赋予其与文本的对齐能力，从而实现多模态和零样本（Zero-shot）应用 1。该方法遵循了LiT（Locked-image Text tuning）范式，即在保持DINOv3视觉骨干网络冻结的前提下，训练一个轻量级的文本编码器，并通过对比损失使其与视觉编码器对齐 1。

该方法的核心创新在于，它同时结合了代表全局语义的CLS令牌和平均池化后的补丁嵌入，实现了对全局和局部视觉特征的同时对齐 1。这使得模型在无需任何微调的情况下，在ADE20k和Cityscapes等任务上实现了卓越的零样本分割性能，同时在全局分类和检索任务上与CLIP等SOTA模型保持了竞争力 1。

## **第六部分：实验成果与性能评估**

### **致密特征的卓越性能**

DINOv3的核心突破在于其在致密特征图上的卓越表现。其性能在多项基准测试中得到了验证。

* **线性探测：** 在不进行微调的情况下，仅在冻结的DINOv3骨干网络上训练一个线性分类器，即可在语义分割和深度估计任务上取得SOTA性能 1。  
  * 在ADE20k上，DINOv3的mIoU达到55.9，比其前身DINOv2高出6点，并显著超越了依赖标注的聚合模型（如AM-RADIOv2.5）和弱监督模型（如PEspatial） 1。  
  * 在单目深度估计上，DINOv3在NYUv2和KITTI数据集上同样大幅领先所有竞争对手 1。

**表1：致密线性探测结果**

| 方法 | ViT | ADE20k (mIoU) | Cityscapes (mIoU) | VOC (mIoU) | NYUv2 (RMSE) | KITTI (RMSE) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **DINOv3** | **7B/16** | **55.9** | **81.1** | **86.6** | **0.309** | **2.346** |
| DINOv2 | g/14 | 49.5 | 75.6 | 83.1 | 0.372 | 2.624 |
| Web-DINO | 7B/14 | 42.7 | 68.3 | 76.1 | 0.466 | 3.158 |
| AM-RADIOv2.5 | g/14 | 53.0 | 78.4 | 85.4 | 0.340 | 2.918 |
| PEspatial | G/14 | 49.3 | 73.2 | 82.7 | 0.362 | 3.082 |
| SigLIP 2 | g/16 | 42.7 | 64.8 | 72.7 | 0.494 | 3.273 |
| PEcore | G/14 | 38.9 | 61.1 | 69.2 | 0.590 | 4.119 |

*注：部分表格数据已根据报告需求进行了精选和整合。*

* **非参数方法：** DINOv3在无需任何训练的非参数任务上也表现出色 1。在3D关键点匹配（NAVI、SPair）和视频跟踪（DAVIS、YouTube-VOS）等任务上，其性能大幅超越所有竞争对手 1。在无监督物体发现任务上，DINOv3的性能同样显著优于其前辈DINO、DINOv2以及其他所有模型 1。

### **鲁棒而通用的全局特征**

DINOv3的性能提升并非以牺牲全局任务表现为代价 1。在评估模型全局特征的质量时，DINOv3展现出了强大的鲁棒性和通用性。

* **线性探测：** 在ImageNet-V2和ImageNet-ReaL等“干净”测试集上，DINOv3的线性探测准确率与SigLIP 2和Perception Encoder等弱监督模型相当 1。在对模型鲁棒性要求极高的OOD（Out-of-Distribution）基准（如ImageNet-R、ObjectNet）上，DINOv3的表现显著优于所有先前的SSL模型，并接近甚至超越了部分弱监督模型 1。

**表2：全局线性探测结果**

| 方法 | ViT | IN-Val | IN-V2 | IN-ReaL | IN-R | IN-A | Obj. |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **DINOv3** | **7B/16** | **88.4** | **81.4** | **90.4** | **91.1** | **86.9** | **79.0** |
| DINOv2 | g/14 | 87.3 | 79.5 | 89.9 | 81.1 | 81.7 | 66.4 |
| Web-DINO | 7B/14 | 85.9 | 77.1 | 88.6 | 64.0 | 71.6 | 69.7 |
| PEcore | G/14 | 89.3 | 81.6 | 90.4 | 92.2 | 89.0 | 80.2 |
| SigLIP 2 | g/16 | 89.1 | 81.6 | 90.5 | 92.2 | 84.6 | 78.6 |

*注：IN-R/A/Obj.代表ImageNet-Rendition/Adversarial/ObjectNet。*

* **实例识别：** 在非参数检索任务上，DINOv3同样取得了压倒性优势 1。在Oxford、Paris、Met和AmsterTime等数据集上，DINOv3的性能大幅超越所有竞争对手 1。

### **作为复杂视觉系统的基石**

DINOv3的一个关键发现是，其学习到的特征质量极高，以至于它可以作为一个**冻结的骨干网络**，在不进行微调的情况下，为复杂的下游系统提供强大的支持 3。

* **目标检测：** 使用冻结的DINOv3骨干网络，结合轻量级的Plain-DETR解码器，在COCO数据集上实现了66.1的mAP，达到了SOTA水平 1。在对鲁棒性要求更高的COCO-O数据集上，其表现也远超其他模型 1。  
* **语义分割：** 同样采用冻结的DINOv3骨干网络和Mask2Former解码器，在ADE20k上取得了63.0的mIoU，与SOTA模型持平 1。  
* **单目深度估计：** 在Depth Anything V2（DAv2）框架中，用冻结的DINOv3替换DINOv2，在所有5个真实世界数据集上均实现了SOTA，这证明了DINOv3在桥接模拟-现实（sim-to-real）差距方面的强大能力 1。  
* **3D理解：** 在VGGT（Visual Geometry Grounded Transformer）中用DINOv3替换DINOv2，在相机姿态估计、多视图估计和视图匹配等3D任务上获得了稳定提升 1。

这些实验共同证明了DINOv3所代表的范式转变：将重计算集中在一次性的大规模预训练上，而下游应用只需进行轻量级的适配，从而极大地加速了研发周期，降低了部署成本，实现了“一个前向传播服务多个应用”的愿景 3。

## **第七部分：跨领域应用：以地球观测为例**

自监督学习的训练范式本质上是通用的，可以应用于任何图像领域 1。为了验证DINOv3的领域通用性，论文通过在卫星图像上训练了一个DINOv3 7B模型，并将其性能与自然图像模型进行比较，因为卫星图像与自然图像具有截然不同的特点（如传感器噪声、视角和光谱带） 1。

该卫星模型在名为SAT-493M的Maxar卫星数据集上训练，并使用了与自然图像模型完全相同的训练配方（除了归一化参数） 1。

### **冠层高度估计**

冠层高度估计是一项关键的地球观测任务，需要精确的连续空间结构恢复 3。DINOv3在这一任务上取得了显著的成果。

* **性能：** DINOv3卫星模型在SatLidar1M和Open-Canopy等基准上实现了SOTA性能 1。与DINOv2相比，DINOv3在肯尼亚地区的冠层高度测量中，平均误差从4.1米降至1.2米，这是一个巨大的提升 3。  
* **领域专长：** DINOv3卫星模型在物理任务（如高度估计）上的表现优于自然图像模型，这证明了领域特定的预训练对于捕捉传感器特定的先验知识至关重要 1。

### **地理空间任务基准（GEO-Bench）**

在对更广泛的地球观测任务进行的GEO-Bench基准测试中，DINOv3同样展现了强大的实力 1。

* **性能：** 冻结的DINOv3卫星和自然图像模型在15项GEO-Bench任务中的12项上创造了新SOTA 1。  
* **跨领域泛化：** DINOv3自然图像模型（在网络图片上训练）在许多地理空间任务上表现出极强的竞争力，甚至超越了专为多光谱数据和微调设计的模型 1。这支持了“领域不可知的预训练可以为专业领域提供强大的泛化能力”这一观点 3。

这些发现表明，DINOv3自然图像模型在**语义**任务上表现出色，因为它学习了更通用、更多样的视觉概念；而卫星模型在**物理**任务（如深度/高度估计）上更优，因为它学习了特定的传感器和物理先验 1。这一发现为未来的视觉基础模型发展指明了方向，即可能存在一种“一主多专”的模式，以在效率和性能上实现最佳平衡。

## **第八部分：结论、局限性与未来方向**

### **核心结论：通用视觉基础模型的诞生**

DINOv3通过其规模化、Gram Anchoring和多学生蒸馏等创新，成功克服了自监督学习在大规模训练中的主要障碍 1。它首次证明，一个

**冻结的骨干网络**可以作为通用的视觉编码器，在不牺牲任何性能的情况下，在密集和全局任务上均达到甚至超越SOTA 3。DINOv3重新定义了视觉基础模型的概念，其强大的泛化能力和“开箱即用”的特性，使其成为一个通用且实用的视觉骨干，为计算机视觉领域的未来发展提供了强大的基石 3。

### **模型的局限性与挑战**

尽管取得了巨大成功，DINOv3也存在一些局限性，为未来的研究提供了方向 1。

1. **对OCR任务的挑战：** DINOv3在需要字符识别的OCR类任务上，其性能仍落后于依赖图文对的弱监督模型 1。这是因为DINOv3的训练没有接触到任何文本信息，难以学习字符等细粒度信息。  
2. **公平性与偏见：** 在地理公平性分析中，DINOv3在低收入地区的表现仍有显著下降，尽管比DINOv2有所改善 10。这提醒我们，即使是无标注数据也可能包含偏差。

### **未来研究方向展望**

基于上述发现，未来的研究可以探索以下方向：

* **更深度的多模态融合：** 探索如何将DINOv3的强大视觉特征与语言模型更深度地融合，以提升其在零样本和多模态任务上的性能，并解决OCR等难题。  
* **更高效的蒸馏：** 进一步优化多学生蒸馏流程，以适应边缘计算和移动设备等更多样化的部署场景 3。  
* **更普适的训练数据：** 研究如何构建更大、更具普适性的无标注数据集，以进一步提升模型的泛化能力和公平性 1。

#### **Works cited**

1. DINOv3.pdf  
2. Paper page \- DINOv3 \- Hugging Face, accessed September 12, 2025, [https://huggingface.co/papers/2508.10104](https://huggingface.co/papers/2508.10104)  
3. DINOv3: Self-supervised learning for vision at unprecedented scale \- Meta AI, accessed September 12, 2025, [https://ai.meta.com/blog/dinov3-self-supervised-vision-model/](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)  
4. DINOv3 Explained: Technical Deep Dive \- Lightly AI, accessed September 12, 2025, [https://www.lightly.ai/blog/dinov3](https://www.lightly.ai/blog/dinov3)  
5. DINOv3: Self-Supervised Vision Model by Meta AI | by DhanushKumar | Aug, 2025 \- Medium, accessed September 12, 2025, [https://medium.com/@danushidk507/dinov3-self-supervised-vision-model-by-meta-ai-45bdcba3527e](https://medium.com/@danushidk507/dinov3-self-supervised-vision-model-by-meta-ai-45bdcba3527e)  
6. DINOv3 Explained: Scaling Self-Supervised Vision Transformers | Encord, accessed September 12, 2025, [https://encord.com/blog/dinov3-explained-scaling-self-supervised-vision-tr/](https://encord.com/blog/dinov3-explained-scaling-self-supervised-vision-tr/)  
7. DINOv3: why vision foundation models deserve the same excitement as LLMs \- Towards AI, accessed September 12, 2025, [https://pub.towardsai.net/dinov3-why-vision-foundation-models-deserve-the-same-excitement-as-llms-630cd469d1f3](https://pub.towardsai.net/dinov3-why-vision-foundation-models-deserve-the-same-excitement-as-llms-630cd469d1f3)  
8. DINOv3 Explained: The Game-Changing Vision Transformer That's Redefining Computer Vision | by Abhishek Selokar | Aug, 2025 | Medium, accessed September 12, 2025, [https://medium.com/@imabhi1216/dinov3-explained-the-game-changing-vision-transformer-thats-redefining-computer-vision-cd63646141e6](https://medium.com/@imabhi1216/dinov3-explained-the-game-changing-vision-transformer-thats-redefining-computer-vision-cd63646141e6)  
9. Introducing DINOv3: Self-supervised learning for vision at unprecedented scale \- YouTube, accessed September 12, 2025, [https://www.youtube.com/watch?v=-eOYWK6m3i8](https://www.youtube.com/watch?v=-eOYWK6m3i8)  
10. facebook/dinov3-convnext-small-pretrain-lvd1689m \- Hugging Face, accessed September 12, 2025, [https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-small-pretrain-lvd1689m)  
11. \[2508.10104\] DINOv3 \- arXiv, accessed September 12, 2025, [https://arxiv.org/abs/2508.10104](https://arxiv.org/abs/2508.10104)  
12. DINOv3: Meta's AI That Learns to See Without Labels | by Edwin Lisowski \- Medium, accessed September 12, 2025, [https://medium.com/@elisowski/dinov3-metas-ai-that-learns-to-see-without-labels-e5a8443a9b5f](https://medium.com/@elisowski/dinov3-metas-ai-that-learns-to-see-without-labels-e5a8443a9b5f)  
13. Global-Scale Forest Height Estimation | Max Zimmer, accessed September 12, 2025, [https://maxzimmer.org/blog/2025/estimating-canopy-height-at-scale/](https://maxzimmer.org/blog/2025/estimating-canopy-height-at-scale/)  
14. A Deep Learning Approach to Estimate Canopy Height and Uncertainty by Integrating Seasonal Optical, SAR and Limited GEDI LiDAR Data over Northern Forests \- arXiv, accessed September 12, 2025, [https://arxiv.org/html/2410.18108v1](https://arxiv.org/html/2410.18108v1)  
15. 14.5 kB \- Hugging Face, accessed September 12, 2025, [https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m/resolve/main/README.md?download=true](https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m/resolve/main/README.md?download=true)