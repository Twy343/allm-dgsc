# 第二方向操作指南

2026-11-19交接：第一方向中期材料见
[结果与现象说明](../../results/interim_sc_case_study/README.md)。基础调试器已放入正常代码目录，
实际执行版本也保留在结果目录的execution-source.tar.gz中。位置操作仍是待接入、待验证原型，
不能将本指南当作位置实验已完成的声明。

## 环境与输入前提

当前Console沿用集群allm环境和已有权重，入口为
`python -m exp_code.semantic_intervention.debug_console --verify --idle-seconds 0`。
需要在已分配的GPU上运行；启动前检查现有作业，不要重复申请或启动。
入口优先读取已有`outputs/sc-benchmark64-deferred-v2/run.json`；该运行文件不随Git分发。
新clone没有它时，自动使用已提交的`configs/debugger_run.json`，其中只含两个示例batch，
包括GSM480与MBPP277，并保留原batch编号与seed。模型权重仍需单独准备。

明确指定配置和自己的权重路径（在已配置allm环境、已取得GPU的仓库根目录运行）：

```bash
python -m exp_code.semantic_intervention.debug_console --run-config configs/debugger_run.json --checkpoint /your/path/diffusiongemma-26B-A4B-it --verify --idle-seconds 0 --output outputs/position-session-01
```

如果使用原服务器的现有模型路径，可以省略`--checkpoint`。配置不是结果文件或GPU断点；
GSM480 seed=0、focus=3；MBPP277 seed=100008、focus=2。不要重新排序batch成员。
不要把CSV里的top16候选当作可恢复的完整GPU断点。

第一方向的方法分类见 `../../docs/INTERVENTION_TAXONOMY.md`。位置/区域作为已有内容操作
的参数，与本方向真正沿位置轴搬动同一候选的概率质量，应分别登记。

接收人：队友 1。目标是按 proposal 4.2 检验位置耦合：保持候选 token 的身份，
把它在 SC 中的概率质量从位置 i 移到附近 j，观察邻居是否跟随、补偿或恢复原位。
先在一个清楚的样本上做完整对照，再选少量例子复核；无需重新训练或重跑 benchmark。

## 可以复用什么

| 文件 | 用途与当前状态 |
| --- | --- |
| `../common/canvas_debugger.py` | 已通过实际 checkpoint 的 native 一致性、断点恢复、重复分支验证。复用原生 while-loop 后端。 |
| `../semantic_intervention/debug_console.py` | 已有常驻模型、load/save/restore/inspect/step/export，以及 swap/mix/canvas/zero/region。 |
| `intervention.py` | 已有两行之间转移单个候选概率质量的原型和 PositionalShift；尚未接入常驻控制台并完成真实 checkpoint 验证。 |
| `metrics.py` | 已有概率质心计算；需要加上局部窗口、总质量和逐步输出。 |
| `../../scripts/review_filter_operator.py` | MBPP277 函数测试与表达式定位示例，可复用记录与判读思路。 |

**第一方向的大部分基础设施已经可用。第二方向仍需补一个原子两行 SC 操作和跟踪器。**
现有 `swap POS SOURCE_ID TARGET_ID` 交换的是同一位置的词表候选，不能当成空间位移。
`region` 写入固定位置模板，也不等价于守恒的候选质量转移。

旧 `../common/dg_runner.py` 使用单独 JIT 的 sample_step。此前真实模型上发现这种
编译上下文可能偏离原生生成，因此不要直接沿用旧 README 的 fixed-trajectory 路线
作为已验证的正式后端。保留 CanvasDebugger 的原生 while-loop；不重写 forward/采样循环。

## 一个分支的时间顺序

所有 canvas 位置采用0基索引；step指已完成的轮数。完成 step t 后，C_t 已经采样，P_t 已经用于生成下一轮 SC，但尚未被 step t+1 消费。
在这个断点修改 SC，首次受影响的是 t+1。恢复断点时必须一起恢复 canvas、SC、RNG、
step、done 与记录前缀，不要只重设 seed 或重新写当前文本。

已接受的位置下一轮仍会预测；accepted 不是永久锁定。整个样本停止后，现有控制台不能
直接重开它。先从停止前的断点开始。`step 2` 是推进两轮，不是设定绝对 step2；
`until 4` 才是推进到完成第4轮。命令逐行输入，不支持分号串命令。

## 先做一个最小空间干预

固定样本、原始 batch、seed 和窗口。先看无干预轨迹，选一个候选在相邻位置都有概率
的阶段，并确认它的语义角色。可先看 MBPP277 的 return 附近；原轨迹 return=2060、
位置42，附近43为开括号。GSM233 的数字竞争和措辞位移也可作为第二个候选观察例。
这些位置只在已保存的原 batch/seed 轨迹中核实，换轨迹必须重新定位。

对 p_i=P_t[i,v]、p_j=P_t[j,v]，定义

    delta = fraction * min(p_i, 1 - p_j)
    p_i_new = p_i - delta
    p_j_new = p_j + delta

分别在两行内按比例缩放其余候选，使每行仍归一化。当前 sc_logits 已做温度整形，不能再除一次温度。编码修改后的 shaped logits 为 SC，
只替换 i、j 两个 SC embedding 行；其他 SC 行、当前 canvas、RNG、步数和前缀 KV 均不改。
它移动一个候选的质量，不是平移整行 logits，也不是搬动整个 SC 向量。

队友首先在独立的新文件中实现 `apply_position_move(engine, ...)`，再写一个继承现有
Console 的小包装器，增加拟议的 `posmove` 命令。**posmove 目前尚不存在。**
不必重写 prepare_canvas、模型加载或 native 采样后端；新命令直接调用 engine 上的操作。
修改 state 之前保留旧 state，任何检查失败都回滚，记录到 engine.edits，沿用 save/restore。

原型接入前必须补齐：Float32 softmax 与质量计算；fraction=0 或实际delta=0 真正原样返回；有效范围和
forbidden-token 掩码；源/目标相邻且在同一 canvas；实际 dtype 舍入后的质量变化。
只改两行 embedding，避免全量重编码覆盖别处已有 SC 覆盖操作。delta 受目标容量限制，
比较不同条件时使用实际移动量，不能只比较 fraction 数值。高置信行应直接计算其他候选
的质量或用掩码log-softmax，避免1-p的消减误差把真实尾部分布误变成均匀分布。

## 对照与首轮规模

先取 delta_position=-1/+1，fraction=.5/1，观察干预后1/3/5步。
每个独立分支从同一断点恢复，不把上一个干预的结果当作下一个分支起点。

| 条件 | SC | 当前 canvas |
| --- | --- | --- |
| baseline | 不改 | 不改 |
| SC-only | 候选 v 的质量从 i 转到 j | 不改 |
| canvas-only | 不改 | 交换 i、j 的输入 token，保持 token 清单 |
| joint | 与 SC-only 相同 | 与 canvas-only 相同 |
| no-op | fraction=0，逐数组恒等 | 不改 |
| 对照位置 | 同候选、相近实际质量，换无关位置 | 按条件固定 |

如果 i 处尚未接受 v，canvas-only 只是偏移位置写入候选，不能称为搬动已有 token。
记录真实操作。单次 `canvas` 输入编辑也不等于把输出硬锁定一整轮。
proposal 中“多保持一轮”的硬锚定需要另加 clamp 钩子、释放时机与独立记录；强制写入的
位置不能冒充模型自己 accepted。首轮先完成 SC-only 与输入 canvas 交换即可。

## 当前控制台上可以直接复用的操作

先检查自己的队列和已有会话；一块研究分配中保持同一模型进程，不为每组重新申请卡。
不要修改另一个正在运行的 worker 所用的公共模块；在独立分支/工作目录开发新包装器。
新开会话时用自己有权限的项目副本、环境和 checkpoint 路径。

```bash
export SLURM_CONF=/etc/slurm/slurm.conf
squeue -u "$USER"
# 先进入 tmux；按实际研究时长申请一张卡，下面是1小时研究块示例
srun -A edu -p short --gres=gpu:R6000:1 -c 4 --mem=32G -t 01:00:00 --pty /bin/bash
cd /insomnia001/depts/edu/users/yc4953/projects/allm-dgsc
source scripts/activate_allm.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
export JAX_COMPILATION_CACHE_DIR=/insomnia001/depts/edu/users/yc4953/hf_cache/jax_sc_test256
run_id=$(date +%Y%m%d-%H%M%S)
python -u -m exp_code.semantic_intervention.debug_console --verify --idle-seconds 0 --output "outputs/position-debug-${run_id}"
```

上面启动的是**已有通用控制台**，用于熟悉原流程；路径目前沿用本项目部署，自己的副本
需相应配置。真实 checkpoint gate 通过后才开展新干预。下面是已支持的 canvas 空间交换
示例，不是尚未实现的 SC posmove。先 inspect 确认实际 token 对应，再执行。

```text
load mbpp:277
until 4
save base
inspect 42
inspect 43
step 1
export position_demo_baseline_h1
restore base
canvas 42 642
canvas 43 2060
step 1
export position_demo_canvas_swap_h1
step 2
export position_demo_canvas_swap_h3
restore base
```

inspect 必须看到 source42=return/2060、target43=开括号/642，且样本尚未停止；否则重选。
每组用新的 export 名称，保留原结果。不要在仍计划继续研究时 quit；Slurm 到时会终止分配，
内存断点随之消失。CSV 不是可冷恢复的完整模型 checkpoint。

## 怎样测位置耦合

提前固定局部窗口，例如 i±8，并选定 anchor 与2到3个相关邻居 token。
当前导出只保存 top16；候选不在 top16 时不能当作概率0。新包装器应在每步读取完整的
当前 logits，计算并保存所选 token 在窗口每个位置的**精确概率**，不必保存整个词表历史。
分别记录 before_edit、after_edit、next_model_prediction，避免把输入 SC 与新输出混淆。

    M_v(t) = sum_position p_t(position, v)
    COM_v(t) = sum_position position * p_t(position, v) / M_v(t)

同时报告 M 与 COM；质量几乎消失时 COM 无意义。标明这是局部窗口的质心。
同一 token 多次出现时，先匹配同一语义实例，不用全局质心混合多个变量引用。
邻居统计须和 baseline 同步比较：anchor 是否保持位移、邻居是否同向移动、是否通过插入/
删除或空白调整补偿，还是回到原布局。按语义 span 对齐；固定位置换了 token 可能只是措辞
长度变化。单次一整段损坏只能称扰动影响，不能直接当作有组织的协同位移。

## 逐步记录与交付

保留原 steps.csv、tokens.csv、NPZ、branch.json；新增：

- `position_design.csv`：case、prompt hash、batch/seed、checkpoint/source、t、i/j、token_id、
  fraction、实际 delta、两行前后概率/entropy、embedding delta、窗口与条件。
- `position_probabilities.csv`：condition、completed_step、phase、canvas_position、token_id、
  exact_probability；据此生成质心和位置热图。
- `position_summary.csv`：anchor/邻居 COM 与质量、实际 accepted/renoised/reopened、停止/上限、
  局部编辑与跟随/补偿/恢复/不确定分类。

先通过原型的归一化/质量守恒/未编辑行不变测试，再在同一实际 checkpoint 上验证 no-op
逐数组一致、同断点重复位移一致、仅两行 SC 改变且 canvas/RNG 不变，最后开始正式条件。
数值函数 CPU 测试通过不等于位置机制已在 DG 上复现。避免重新引入单独 JIT 步进的旧偏差。

当前 adaptive stop 在第一方向中会影响是否有机会恢复；主比较先保留原策略并明确停止点。
若做固定步数/禁用停止对照，应另设经验证的模式，记录配置与步数/RNG，不暗中重开 done。
不合并不同停止策略的“恢复时间”。

队友交付：新两行操作及控制台命令、测试与真实 gate 记录、一个样本的完整对照CSV、
一张 anchor/邻居位置概率热图、一个质心轨迹图、2到3个有代表性的分支说明。
保留恢复、无效和损坏分支；跨样本结论需要少量独立样本复核，1/3/5步快照不是独立样本。

## 现有资料

- `../../docs/SC_DEBUGGER.md`：通用断点操作与导出范围。
- `../../docs/SC_MECHANISM.md`：SC 的实际计算、接受和再加噪机制。
- `../../outputs/sc-debug-session-14004156/review/CASE277_NOTES.md`：第一方向个例经验。
- `../../outputs/full-sc-steptrace-v1/`：30条已有原始轨迹；以新复跑 CPU 分数判断是否答对。
- `../../docs/DiffusionGemma_Final_Project_Proposal.docx`：以4.2的空间干预问题为准。
