# 第一方向报告与少量复核

这里的第一方向是 proposal 4.1 的 SC 语义因果干预。六组64条每任务的 benchmark
属于此前的实现检查和背景结果，不代替语义干预证据。

## 方法计数口径

干预按4类基本操作整理：候选logits交换、SC分布替换/混合、SC清零、canvas token替换。
162是已完成的扰动配置数，另有1个baseline；位置、步数、强度、范围和通道组合不计为
新的方法。详见[方法分类](INTERVENTION_TAXONOMY.md)及../results/intervention_taxonomy/。

## 已有材料

中期报告的四类操作、现象解释、匹配对照和31个代表分支已整理为
[可版本管理的结果存档](../results/interim_sc_case_study/README.md)。它包含全部166条
条件索引、三组图和原执行源码包；可用scripts/archive_interim_results.py --verify离线校验。

MBPP277 的完整快照包已在本地校验SHA256，544份逐步导出均通过 native 前缀、
采样与接受、记录的canvas编辑连续性检查。去重后166种操作序列中，163种已经自然停止，
3种只是重复脉冲分支的中途前缀，不算未完成的正式实验，也不计入完成数量。
18组重复干预逐数组一致，另有1项原始轨迹对照。六项评分/绑定回归测试通过。

可展示三类主例：SC内容影响跨位置变量协调；错误算符被接受后恢复或在停止时保留；
SC扰动引起局部再加噪而全局停止，从而损坏代码。对照、无效与修复分支均保留。
GSM480已有算术修复及区域再加噪个例，可作为不同任务和不同结局的对照。

主表：../outputs/sc-debug-session-14004156/review/case277_interventions.csv
原始记录及所有条件：../outputs/sc-debug-session-14004156/
图：同目录review/operator_recovery.png、rename_propagation.png、late_rename_dose.png。

## 报告前剩余工作

从DIRECTION1_CASE_SHORTLIST.csv选少量例子，而非把每个样本都做163组。
建议总共3个MBPP与2个GSM例子，其中277/480复用已有材料，其余3个目前仅完成选择，
尚未声称已跑干预。新例先重放baseline，核实答对、目标含义、实际接受状态与停止点。

每个新例先用无干预、SC-only、canvas-only、joint及原内容对照，比较接受前后两个时机，
观察1/3/5步。统一记录预测偏移、实际接受、再加噪、恢复、停止及函数/答案正确性。
保留全部预先选择的条件；需要微调位置或强度时另记探索批次。

单断点重放验证可复现；跨样本、必要时不同随机状态的复核提供泛化依据。
不要把快照数、同一seed重复数当作独立样本数，也不从本例探索网格估计总体干预成功率。

第二方向的交接指南见../exp_code/positional_intervention/README.md。
