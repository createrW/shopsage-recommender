# ShopSage Recommender — 计划与进度

[English](plan.md) | **简体中文**

**更新：** 2026-10-24，S2 批量训练完成后 · **进度：** S0、S1 和 S2 已完成（批量训练记为 E007）；下一步是 S3 · **代码：** `05f74c4` 已 push 到 [createrW/shopsage-recommender](https://github.com/createrW/shopsage-recommender)；S2 已在本地提交为 `fefcf6c`（未 push）；汇总修复和 E007 文档尚未提交 · **配套文档：** [memory.zh-CN.md](memory.zh-CN.md)（事实、规则、决策）

- **目标：** 在 MovieLens-20M 上做一个两阶段推荐系统（Two-Tower 稠密召回和 BM25 稀疏召回、LightGBM 重排、可选的 LLM 查询扩展、Milvus Lite 检索、Streamlit demo），所有对外报告的数字都来自实测。
- **架构：** 离线部分，PyTorch 在云 GPU 上训练双塔，numpy 和 scipy 为评估计算精确候选，LightGBM 对两条召回路径的并集重排。在线部分，demo 只用 CPU：numpy 实现的用户塔、Milvus Lite（稠密、稀疏 BM25 和 RRF 混合检索）、LightGBM，以及用于查询扩展的 Claude Haiku。
- **技术栈：** Python 3.11–3.12、Polars、DuckDB、PyTorch 2、LightGBM、Milvus Lite（`pymilvus`）、scipy、W&B、Streamlit、Typer、pytest 加 hypothesis、Claude Haiku 5.5。
- **需求来源：** 你 2026-10-23 更新的范围（§2）。设计理由见 [design.md](design.md)；memory.zh-CN.md §6 列出的章节在 S7 之前视为已被取代。

> **工作循环（你在 2026-10-23 定的规则）。** 每一步开始前：先读 plan.md 和 memory.md。每一步结束后：在这里打勾，更新 memory.md 和两份中文版，然后告诉你接下来要做什么。**效率优先：** 选能如实测出目标指标的最简路径，不在这条路径上的一律砍掉（§3）。执行方式：由我直接逐步实现，测试先行。

云端部署遵循你的托管方案 `cloud_hosting_plan_2026-10-17.md`（不在本仓库内：`/Users/zy/Project/new_sys/cloud_hosting_plan_2026-10-17.md`）中的 P02 项目。§6 把它落到本仓库上。

## 1. 一览

| 方面 | 已完成 | 下一步 |
|---|---|---|
| 数据 | ML-20M 下载；Liang 划分，精确复现了已发表的计数（E004）；Day 1 的划分保留 | — |
| 评估 | 带屏蔽的全量 top-k、截断 Recall / NDCG / HR、百分位 bootstrap 与配对 bootstrap；`shopsage evaluate` | S2：用同一套代码给 Two-Tower 打分 |
| 基线 | MostPop 和 EASE，已对照已发表数字完成校准（E004） | — |
| 模型 | Two-Tower A0–A4 已在 Modal 上训练完（E007）：A1 Recall@50 0.5361、NDCG@10 0.3351，高于 EASE；消融 (a) 完成 | S4 对它的候选重排 |
| 检索 | Milvus Lite 探测：在笔记本上实测了稠密、稀疏和混合检索（E004） | S3：BM25 路径、Milvus 服务索引、消融 (a) 和 (b) |
| 排序 | — | S4：基于两条召回路径的 LightGBM |
| LLM | 设计 | S5：Claude Haiku 扩展，消融 (c) |
| Demo | 探测已部署到 Community Cloud：峰值 482 MB，混合检索 p95 176 ms（E005） | S6：完整 demo；已规划延迟兜底方案（§8） |
| 仓库 | 122 个测试通过；lint、类型检查和 lock 都干净；S1 已上 GitHub，CI 通过 | 提交并 push S2（由你来做，§7） |

目前的实测结果（`Liang-SG-strict`，10,000 个测试用户）：MostPop Recall@50 0.2352、NDCG@10 0.1275；EASE Recall@50 0.5210、NDCG@10 0.3280（E004）。Two-Tower A1，3 个种子（E007）：Recall@50 0.5361 ± 0.0005、NDCG@10 0.3351 ± 0.0005，Recall@50 比 EASE 高 +0.0151 [+0.0119, +0.0184]。困难负样本：A3 − A1 = −0.0020 [−0.0033, −0.0007]，+0.08 的目标没有达到；logQ 校正带来 +0.225（A1 − A0）。

## 2. 范围与目标（你 2026-10-23 的需求）

- **数据：** MovieLens-20M（138,493 个用户、27,278 部电影、20,000,263 条评分）。Amazon Reviews 2023 Toys 砍掉（§3）。
- **召回：** 两条路径，Two-Tower（稠密）和 BM25（稀疏），外加二者的 RRF 混合。**排序：** LightGBM 排出 Top-10。**LLM：** 用 Claude Haiku 做查询扩展，是可选路径。**服务端检索：** Milvus Lite。**Demo：** Streamlit。**实验追踪：** W&B。**指标：** NDCG@10、Recall@50 和 HitRate@50，带 bootstrap 95% 置信区间。

以下是目标而不是结果；每一项都对照已发表的参考数字汇报：

| 目标 | 数值 | 测量方式 | 已发表的参考 | 评估 |
|---|---|---|---|---|
| Recall@50 | ≥ 0.55 | 10,000 个 Liang 测试用户上的截断 Recall@50 | EASE 0.521；RecVAE 0.553，已核实的最高值（`Liang-SG-strict`） | 仓库内 EASE：0.5210（E004），目标比它高 0.029；最可能的路线是排序器重排约 600 个候选之后的系统级 Recall@50 |
| 排序后的 NDCG@10 | ≥ 0.32 | 同一批用户 | EASE 0.329（`Liang-SG-fit`） | 仓库内 EASE：0.3280（E004），已经高于目标，所以排序器只有超过 EASE 才算数 |
| 延迟：召回加排序到 10 | p95 ≤ 200 ms | demo 主机，CPU，不含 LLM | — | 大概率能达到：S1 探测在笔记本上测得 Milvus Lite 混合检索 p95 为 34–44 ms（E004）；主机上的数字在 S6 测 |
| 难负样本消融 | Recall@50 +0.08（A3 − A1） | 同一批用户，每组 3 个种子 | — | 偏乐观，因为 A1 已经有 logQ 和 accidental-hit 屏蔽；按实测汇报 |

## 3. 效率优先的改动（2026-10-23）

| 方面 | 之前 | 现在 | 原因 |
|---|---|---|---|
| 评估协议（D-001） | `LOO-full` 和 `LOO-asof`；Liang 可选，Day 4 由你决定 | 只用 `Liang-SG-strict`，精确复现 Liang 等人的划分；除非你否决，这就是默认方案 | 目标数字只在这个协议下有意义；只需建一个协议；随机 80/20 留出不需要 as-of 机制 |
| 特征 | as-of 访问器、泄漏金丝雀、61/51 个用户的缺失值规则 | 特征只来自 fold-in 行，物品统计只来自训练用户，一个金丝雀测试 | as-of 机制是为时间戳划分准备的 |
| 用户塔（D-004） | 待定 | 归纳式：对 fold-in 历史做池化 → MLP；没有用户 ID 表 | strict 协议下的留出用户从不参与训练 |
| 负样本（D-005） | 第 2 轮 = 均匀 MNS；挖掘的负样本可选 | 主对比是 A1（in-batch + logQ）对 A3（A1 + 挖掘的难负样本），各 3 个种子；A0、A2 和 A4 各 1 个种子 | 需求明确要求难负样本挖掘的消融 |
| 检索引擎（D-008） | 精确检索加 FAISS ANN 研究 | 离线用 numpy 精确检索；服务端用 Milvus Lite（FLAT；稠密、稀疏 BM25 和 RRF）；FAISS 不再是直接依赖（Milvus Lite 3.x 底层用 faiss-cpu） | 需求里有 Milvus；Milvus Lite 只支持 FLAT，而 2 万个物品本来就该用精确检索 |
| BM25（D-010） | scipy BM25 | 离线用 scipy；同一套权重存成 Milvus 稀疏向量；不用 Elasticsearch | 不需要 JVM 服务，而且能在 Streamlit Cloud 上运行 |
| 排序器（D-006） | 稠密 Top-500 | 稠密 Top-500 ∪ BM25 Top-100；约 11 个特征，含一个稀疏 EASE 分数 | 两条召回路径；最强又便宜的信号 |
| LLM | Haiku，vLLM 跑 Qwen 作为备选 | 只用 Haiku；消融 (c) 约 1,000 个查询，对比 BM25 用原始查询和用扩展后的查询 | 不需要 GPU；约 \$1 |
| 训练 | 很多次运行，一次一个 | 一次 Modal 批量并行训练，墙钟约 25 分钟（估计） | 只交接一次 |
| 种子（D-007） | 每个组 ≥ 3 个 | 每个要下结论的对比 3 个；背景组 1 个，并注明 | 凡是下结论的地方，严谨程度不变 |
| 砍掉 | — | BGE-M3、用户 ID 消融组、`LOO-*` 和 `Liang-SG-fit` 的表、分组诊断、ItemKNN、ANN 研究、vLLM、MLflow、Elasticsearch、Amazon Reviews、构建 Docker 镜像 | 不在通往目标的路径上 |
| 文档 | 每一步都更新所有文档 | 每一步更新 plan.md 和 memory.md；每次运行写 experiment_log.md；README、design.md 和 api_spec.md 只在 S1（加横幅、更新决策）和 S7（重写）时更新 | 减少维护量 |

## 4. 步骤

工作量是对工作时间的粗略估计。每一步结束时测试全部通过，并执行上面的工作循环。

### S0. Day 1 ✅（2026-10-18）

脚手架、Polars 流水线、带 md5 校验的下载、leave-one-out 划分、EDA notebook、文档、CI、云端初始化脚本和 MPS 基准；53 个测试；提交 `3e0f08f`。详见 memory.zh-CN.md §5 和 experiment_log.md 的 E001–E003。

### S1. Liang 划分、评估、基线、打包、Milvus 探测 ✅（2026-10-23，笔记本）

- [x] `src/shopsage_rec/data/liang.py`：`liang_split(positives, *, seed=98765, n_heldout_users=10_000, test_prop=0.2) -> LiangSplit`，逐个调用复现作者的预处理；测试对照了逐行照搬作者 notebook 的参考实现（固定数据和 hypothesis 生成的数据），并在真实数据上对照已发表的计数：9,990,682 个正样本、136,677 个用户、物品从 20,720 到 20,108。`load_liang()` 返回 CSR 矩阵；`partition_val_users()` 给出 V_A / V_B。
- [x] `configs/data/movielens_20m_liang.yaml`（`split.strategy: liang`）成为默认配置：`shopsage prepare` 用 5 秒写出 `data/processed/liang/`（峰值 RSS 2.2 GB）。Day 1 的划分加一个 `--config` 参数即可生成。
- [x] `evaluation/metrics.py`（`topk_masked`、`topk_for_users`、`relevance`、`recall_at_k`、`ndcg_at_k`、`hit_rate_at_k`、`evaluate_topk`）和 `evaluation/bootstrap.py`（`bootstrap_ci`、`paired_bootstrap`，即 api_spec.md §8.4 描述的过程）。
- [x] `models/baselines.py`（MostPop、float32 的 EASE）和 `shopsage evaluate`（`evaluation/runner.py`）：在 V_A 上选 λ，测试指标带置信区间，每次运行输出 `per_user.parquet` 和 `summary.json`。
- [x] **校准关卡已通过（E004）：** MostPop Recall@50 0.2352（发表值 0.235），EASE 0.5210（发表值 0.521）；Recall@20 和 NDCG@100 与 Steck 2019 Table 1 的差距在 0.002 以内。
- [x] 打包：不再直接依赖 FAISS；`pymilvus[milvus-lite]` 放进 `demo` 和 `dev` extra；scipy 进入核心依赖；删掉 `cpu` / `gpu` extra 及其冲突声明；`bootstrap_cloud.sh`、Makefile、CI、`Dockerfile` 和 `requirements.txt` 都已跟进；`uv.lock` 重新锁定（232 个包）。
- [x] 文档同步：experiment_log.md（评估协议、TARGETS、决策 D-001 到 D-014、实验矩阵、E004）；README 围绕 Liang 协议重写并放入实测的基线；design.md 和 api_spec.md 加上横幅，api_spec 的 §2.7、§6.1 和 §9 已改为描述 S1；GitHub 用户名 `createrW` 已填好。
- [x] 探测：`serving/probe.py` 和 `serving/app.py`，以及由 `uv export` 生成的 `serving/requirements.txt`。在笔记本上、20,108 个物品的规模下，Streamlit 加应用的峰值内存约 450 MB；混合检索 p95 为 34–44 ms（E004）。Milvus Lite 在载入数据后必须 `flush()`：没 flush 的行每次查询都会在 Python 里逐行扫描（flush 前稠密检索 p50 181 ms，flush 后 9–17 ms）。CI 会在 Linux 上运行探测，并检查 `serving/requirements.txt` 是否与 `uv.lock` 一致。
- [ ] 在 Streamlit Community Cloud 上部署探测，记录主机上的内存和延迟：需要你提交、push 并创建 Community Cloud 应用（§7）。
- **完成标准：** 已达到，只差主机上的测量，它要等部署完成。

### S2. Two-Tower、负样本、Modal 批量训练 ✅（2026-10-24；代码在笔记本上，批量训练在 Modal 上）

- [x] `features/items.py`：类型 multi-hot、5 年分桶、带缺失标记的 genome PCA-64，缓存到 `item_features.npz`。
- [x] `models/two_tower.py`：物品塔（`item_idx` embedding ⊕ 特征 → MLP → 128 维，L2 归一化）；归纳式用户塔（对最多 50 个采样历史物品的 embedding 取均值 ⊕ 历史长度的对数 → MLP → 128 维）；`sample_history` 会排除目标物品。
- [x] `models/losses.py`：带 logQ 校正和意外命中屏蔽（`PositiveSet`）的 `sampled_softmax`；额外负样本可整批共享（均匀、全量）或按行（挖掘）。
- [x] `models/trainer.py`：A0–A4 五个组、fused AdamW（embedding 不做衰减）、bf16/fp16 规则、按 V_A Recall@50 选最佳 epoch、导出（`item_embeddings.npy`、`user_tower.npz`、`model.pt`）、带置信区间的测试指标（`per_user.parquet`、`metrics.json`）、峰值显存、W&B（没有 key 时离线记录）。`models/mining.py`：每个训练用户的第 101–500 名。`models/sweep.py`：任务计划。
- [x] 一致性测试：在 1,000 个用户上对比 numpy 用户塔和 torch（< 1e-5）。
- [x] `shopsage train / mine / sweep`；`scripts/train.sh`；`scripts/modal_train.py`（在 Volume 上准备数据 → 4 次 τ/lr 检查 → 第一阶段并行 → 挖掘 → 第二阶段 → `sweep_summary.json`；重跑时会跳过已完成的运行）。
- [x] MPS 冒烟运行（A1、A2、A4，10% 用户）和一次全量数据 3 个 epoch 的 A1 检查（E006）。
- [x] 在 Modal 上批量训练（你，2026-10-24）：13 次运行，2.5 个 A10 小时，约 \$2.75；已拷回 `artifacts/runs/`；记为 E007，含消融 (a) 和配对 Δ。

### S3. BM25、Milvus 服务索引、消融 (a) 和 (b)（约 4 小时；笔记本；可以和你的批量训练同时进行）

- [ ] `retrieval/bm25.py`：物品文档（片名、类型、top genome 标签，以及只来自训练用户的标签）；BM25 权重存为 scipy.sparse 矩阵；用户查询是其 fold-in 物品的词项。
- [ ] `retrieval/milvus_store.py`：一个同时含稠密和稀疏向量字段的 collection，载入后 flush；`search_dense`、`search_sparse` 和 `search_hybrid`（RRF，k = 60）；按历史长度多取，再去掉看过的物品。与 numpy 和 scipy 的 top-k 做一致性测试。
- [ ] `retrieval/fusion.py`：离线评估用的 RRF。
- [ ] 表 (b)：MostPop、EASE、BM25、稠密（A1-s42）、混合。记为 E008。（表 (a) 已完成：E007。）
- **完成标准：** 表 (a) 和 (b) 写进 experiment_log.md；一致性测试通过。

### S4. LightGBM 排序器（约 4 小时；笔记本 CPU）

- [ ] 先执行 `brew install libomp`。
- [ ] `ranking/candidates.py`：为 V_B 和测试用户生成稠密 Top-500 ∪ BM25 Top-100。
- [ ] `ranking/features.py`：稠密分数和名次、BM25 分数和名次、稀疏 EASE 分数（每个物品取 top-100 邻居）、训练用户上的物品流行度和平均评分、类型偏好、与历史中心的 genome 相似度、年份差、log 历史长度。金丝雀测试：改变一个用户的目标集，任何特征都不应改变。
- [ ] `ranking/train.py`：以 NDCG@10 为目标的 LambdaRank；V_B 的 10% 用于早停；3 个种子；分别带和不带 EASE 特征。
- [ ] 测试指标：NDCG@10、系统级 Recall@50 和 HR@50，分别与 EASE 和稠密路径做配对比较；离线测完整服务路径的 CPU 延迟 p50/p95。记为 E007。
- **完成标准：** §2 目标表的每一行都能填上。

### S5. LLM 查询扩展，消融 (c)（约 3 小时；笔记本和 Anthropic API）

- [ ] `llm/expand.py`：`claude-haiku-5-5`，结构化输出（`titles`、`keywords`、`genres`），effort 设为 `low`，不传采样参数，用 sha256(model id + prompt + params) 作为缓存键。缺少 key、被拒绝、超时，或扩展结果对应不到任何物品时，退回原始查询；用假客户端测试。
- [ ] 约 1,000 个 (user, tag) 查询的基准，相关物品是该用户打过这个标签的电影：C0 用原始标签做 BM25，C1 用扩展结果做 BM25（有时间再加 C2 混合）；成本取自 `usage`。记为 E008。
- **完成标准：** 表 (c)；成本约 \$1。

### S6. Demo 与部署（约 5 小时；先在笔记本上，再到 Streamlit Community Cloud）

- [ ] 落实 D-003：片名从哪里来，资源包里可以放什么。
- [ ] 导出资源包（物品向量、`user_tower.npz`、稀疏 EASE、BM25 权重、LightGBM 模型、许可范围内的物品元数据）到 GitHub Release；把它的 tag 和 sha256 固定在 `configs/deploy/streamlit.yaml`。Milvus Lite collection 在启动时由资源包构建。
- [ ] `serving/app.py`：选 5 部电影 → Top-10，带分数以及召回每个物品的路径；dense / BM25 / hybrid 开关；自由文本查询，带一个显示扩展结果的 LLM 开关；延迟读数和一个基准测试按钮。
- [ ] 部署到 `yi-shopsage.streamlit.app`；内存在 690 MB 以内；主机上 p95 ≤ 200 ms；录制 GIF。记为 E009。
- **完成标准：** 在线地址可用；§2 的延迟一行已填上。

### S7. 整理与发布（约 3 小时）

- [ ] README 的结果表和消融表（带置信区间）、W&B 报告链接、GIF 和在线地址；重写 design.md 和 api_spec.md 中被取代的部分；简历描述只用实测数字。
- [ ] 最终评审；由你 push。

## 5. 重点检查项

各步主流程都覆盖不到的故障模式；每一项都在所列步骤里配一个测试。

1. 留出用户的目标泄漏进训练或特征（留出用户的标签、在全部用户上统计的流行度）：S1 和 S4 的金丝雀测试。
2. top-k 截断前没有屏蔽 fold-in 物品：S1 的屏蔽 top-k 测试，S3 的 Milvus 多取测试。
3. Milvus 结果与精确路径不一致（并列、过滤、返回少于 k 个）：S3 的一致性测试。
4. Community Cloud 上 demo 的内存或延迟（Debian 11 上的 Milvus Lite，690 MB）：S1 的探测和 S6 的主机基准。
5. LLM 路径故障（没有 key、被拒绝、超时、片名对应不到任何物品）：S5 的假客户端测试。

## 6. 云端部署（遵循托管方案）

### 6.1 托管方案对本项目的要求

- **Demo，必须部署：** Streamlit Community Cloud，免费，从 GitHub 仓库部署，地址形如 `https://yi-shopsage.streamlit.app`；只做 CPU 推理，不在那里跑 PyTorch 训练。备选：Hugging Face Spaces（免费）。
- **GPU 训练：** 电脑没有 GPU 时用 Modal；RunPod 被列为最便宜的 GPU 选项。
- **预算：** \$0 档覆盖本项目；Modal 的每月 \$30 是可选项。规则：开 \$5 的账单提醒，工作一做完就立即关停云资源。

### 6.2 工作负载与平台

| 工作负载 | 平台 | 本仓库中的实现 | 成本 |
|---|---|---|---|
| 数据、EDA、测试、EASE、BM25、LightGBM、LLM 调用 | 笔记本 | `make` 目标、`shopsage` CLI | \$0（LLM 调用约 \$1） |
| Two-Tower 批量训练 | Modal A10；RunPod 作为备选 | `scripts/modal_train.py`；`scripts/bootstrap_cloud.sh` 和 `scripts/train.sh` | 约 \$3（按实测的 A1 epoch 估计），在每月 \$30 的额度之内 |
| Demo | Streamlit Community Cloud | `serving/app.py`、`.streamlit/`、`packages.txt`、`configs/deploy/streamlit.yaml` | \$0 |
| Demo 备选 | Hugging Face Spaces（CPU basic：2 vCPU、16 GB） | `Dockerfile` | \$0 |
| Demo 资源包 | 固定 sha256 的 GitHub Release 附件 | S6 | \$0 |
| 实验追踪 | W&B 免费版 | `WANDB_API_KEY` | \$0 |

### 6.3 训练交接：你要运行的内容

**显存要求：最低 8 GB，建议 16 GB 及以上；默认用 24 GB 的 A10（Modal 写作 `"A10"`）。** 笔记本上的实测（E006，MPS，B = 4,096；MPS 报告的是驱动分配量，是 CUDA 数字的上界）：A1 2.1 GB，A2 2.2 GB，A4（全量 softmax）5.0 GB。A3 每行多 8 个挖掘负样本，大小与 A2 相当，另外 GPU 上要放挖掘池（116,677 × 400，int64 约 0.37 GB；磁盘上 int32 为 0.19 GB）。

| 组 | 峰值显存（MPS 实测） | 每 epoch 秒数，M5 MPS（实测） | 每 epoch 秒数，A10 |
|---|:-:|:-:|:-:|
| A0、A1（in-batch） | 2.1 GB（A10：0.98 GB，实测） | 129–146 | **21.5（实测，Modal 试跑）** |
| A2（+2,048 个均匀负样本） | 2.2 GB | ~150（10% 用户的运行按 ×10 换算） | ~25（估计） |
| A3（每行 +8 个挖掘负样本） | ≈ A2 | 未实测 | ~30（估计） |
| A4（全量 softmax） | 5.0 GB | ~420（10% 用户的运行按 ×10 换算） | ~75（估计） |

- **批量训练估计：** 4 次调参 × 10 个 epoch，然后 9 次 × 30 个 epoch（A4 最长，约 1 小时），约 2.5 个 A10 小时，按 Modal A10 的价格（约 \$1.10 每小时）约 \$3；同一阶段的运行并行，墙钟时间约 1–1.5 小时。以上都是估计；`metrics.json`（`wall_s`、`peak_memory_gb`、`gpu`）会给出实测值。

**方案 A：Modal（托管计划的选择）。**

```bash
uv tool install modal && modal setup                 # 一次性：装 CLI，然后浏览器登录
modal secret create wandb WANDB_API_KEY=<你的 key>   # 一次性；不用 W&B：modal secret create wandb WANDB_MODE=offline
cd /Users/zy/Project/new_sys/projects/shopsage-recommender
modal run scripts/modal_train.py                     # 数据准备 + 整个批量训练；最后打印各组汇总
modal volume get shopsage-data artifacts/runs artifacts/   # 13 个运行目录
```

`modal run` 期间要保持连接（别让笔记本睡眠）；如果断了，再运行一次即可：已完成的运行会被跳过。换 GPU：`SHOPSAGE_GPU=L4 modal run scripts/modal_train.py`。想先试跑的话：`modal run scripts/modal_train.py --epochs 2 --tune-epochs 1`（正式跑之前用 `modal volume rm -r shopsage-data artifacts/runs` 删掉 Volume 上的 `artifacts/runs`，因为运行 id 会被复用）。

**方案 B：RunPod 或任何 Linux GPU 机器，** 顺序运行，A10 上约 4–5 小时。

```bash
git clone https://github.com/createrW/shopsage-recommender.git && cd shopsage-recommender
export WANDB_API_KEY=...                             # 或 WANDB_MODE=offline
bash scripts/bootstrap_cloud.sh                      # uv、train + llm extras、torch 检查、下载、Liang 划分
bash scripts/train.sh --sweep                        # 整个批量训练，逐个运行
```

把 `artifacts/runs/` 拷回来，然后**终止**（terminate）pod：停止（stop）的 pod 仍按卷磁盘计费（RunPod：闲置 \$0.20 每 GB·月）。Community Cloud 的 RTX 4090（24 GB，\$0.34 每小时）就够用。

**方案 C：Colab 或 Kaggle 的免费 T4，** 只适合冒烟运行（fp16，耗时约为两倍）。

### 6.4 Demo 部署：Streamlit Community Cloud

S1 的探测和 S6 的 demo 用同样的方式部署：

1. 仓库已经在 GitHub 上公开（§7）。
2. 在 share.streamlit.io 用 GitHub 登录，从该仓库创建应用：repository 填 `createrW/shopsage-recommender`，branch 填 `main`，main file 填 `src/shopsage_rec/serving/app.py`，App URL 填 `yi-shopsage`（自定义子域名，之后可以改）。
3. Advanced settings：Python 3.12（默认值）。Secrets：`ANTHROPIC_API_KEY = "..."`，只有从 S5 起的在线 LLM 路径需要；没有它 demo 也能运行。
4. Community Cloud 先搜索入口文件所在目录，再搜索仓库根目录，所以它安装的是 `src/shopsage_rec/serving/requirements.txt` 而不是根目录的 `uv.lock`，另外会从根目录的 `packages.txt` 安装 `libgomp1`。push 到 `main` 会更新应用。
5. S6：把 demo 资源包上传到 GitHub Release，把它的 tag 和 sha256 写进 `configs/deploy/streamlit.yaml`，然后 push。
6. 检查项：不导入 torch；Milvus Lite 能启动并返回结果；内存在 690 MB 的保证额度之内（上限 2.7 GB）；一次 Top-10 请求的 p50/p95 延迟；应用休眠（闲置 12 小时）之后的冷启动。

**备选：** Hugging Face Spaces（Docker 或 Streamlit SDK；CPU basic，2 vCPU、16 GB，免费），比如 Milvus Lite 或资源包放不下的时候。D-003 在那里同样适用。

### 6.5 成本护栏（托管方案的规则在本项目中的落实）

- **Modal：** 设置 workspace budget（Modal FAQ 原文："If you want to limit your monthly spend, you can set a Workspace budget"）；Starter 每月 \$30 的额度足够覆盖本计划。
- **RunPod：** 只用预充值额度，不开自动充值；产物拷出之后就终止 pod。
- **云上没有长期运行的资源：** 训练机器用完即弃，唯一常驻的服务是免费的 Streamlit 应用（相当于托管方案里的"做完立刻 `terraform destroy`"）。
- **Anthropic：** 在 Console 里设置较低的月度消费上限；响应有缓存，重跑不花钱。
- **密钥不进 git：** `.streamlit/secrets.toml` 已被 git 忽略，也不在 Docker 构建上下文里；密钥来自环境变量或 Streamlit secrets。
- **demo 的 Docker 镜像只留在本地，** 在 D-003 决定之前不推到公共镜像仓库。

### 6.6 本仓库对托管方案的细化

厂商数据在所示日期重新核对过；平台选择沿用托管方案。

| 托管方案 | 核对后的事实 | 本仓库的做法 |
|---|---|---|
| Streamlit Community Cloud：1 GB 内存 | 保证 690 MB，最高 2.7 GB；≤ 2 个 CPU 核；闲置 12 小时后休眠（Streamlit 文档，2026-10-18） | demo 按 690 MB 设计，并且从不导入 torch |
| Modal A10 每小时 \$0.60，10 小时约 \$6 | A10 \$0.000306/s ≈ \$1.10/h；T4 \$0.000164/s ≈ \$0.59/h；Starter 每月含 \$30（modal.com/pricing，2026-10-23） | 批量训练约 \$1.1–1.7，预算 3 小时约 \$3.3：都在额度之内，自付 \$0 |
| Modal A100 40 GB 每小时 \$1.50 | \$0.000583/s ≈ \$2.10/h | 不需要：最坏情况的峰值是 6.1 GB |
| Hugging Face Spaces GPU 每小时 \$0.60 | T4 small \$0.40/h；CPU basic 2 vCPU、16 GB，免费（huggingface.co/pricing，2026-10-23） | demo 留在 CPU 上；Spaces 作为备选平台 |
| RunPod RTX 4090 约 \$0.40/h（spot） | Community Cloud \$0.34/h，Secure Cloud \$0.89/h；闲置卷盘每 GB 每月 \$0.20（runpod.io/pricing，2026-10-23） | 作为 Modal 的替代；用完终止 pod |
| P02 训练：8–12 GB 显存，10–15 小时 | MPS 实测（E006）：峰值 ≤ 5.0 GB（A4）；批量训练约需 3–5 个 A10 小时（估计） | 最低 8 GB，建议 16 GB 及以上 |
| P02 checkpoint：约 500 MB | 按 Day 1 的规模，每个 checkpoint 约 78 MB 参数加 155 MB AdamW 状态（估计） | 每次运行只保留最佳 checkpoint；产物绝不进 git |
| Demo：面试官输入 user_id | GroupLens 许可禁止未经许可的再分发（D-003） | 访客"选 5 部电影"画像；只有拿到 GroupLens 的许可才提供 user-id 选择器 |
| 简历描述："<200 ms CPU inference; 50+ test users" | 目前什么都还没测 | 延迟在线上平台实测（S6）；用户数只写真实的，托管方案的 Part 1 自己也提醒过这一点 |

## 7. 你的待办事项

**现在**

- [x] 回复"go S1"（2026-10-23；没有否决任何默认方案，所以 D-001 和 §3 的其他默认方案都已确定）。
- [x] 创建 GitHub 仓库并 push Day 1（`createrW/shopsage-recommender`；CI 通过）。
- [x] 提交并 push S1（`05f74c4`，2026-10-24；CI 通过，包括 Linux 上的 Milvus Lite 探测）。

- [x] 在 Community Cloud 上部署探测（2026-10-24；E005：482 MB，混合检索 p95 176 ms）。
- [x] 提交 S2（`fefcf6c`，2026-10-24）。
- [ ] 提交批量训练结果并 push：

  ```bash
  git add -A
  git commit -m "S2: Modal sweep results (E007), plain-JSON run summaries"
  git push
  ```

- [x] 在 Modal 上运行 S2 批量训练（2026-10-24；E007）。注意：入口程序会先在本地写 `artifacts/runs/sweep_summary.json`，`modal volume get` 不肯覆盖它；改为下载到一个空目录。
- [ ] 就 D-003 给 GroupLens 发邮件（回复可能要等一段时间）。邮件是写给 GroupLens 的，所以保留英文：

  ```text
  To: grouplens-info@umn.edu
  Subject: Permission request: public non-commercial demo built on MovieLens-20M

  Hello GroupLens team,

  I am a student at the University of Southern California building a non-commercial
  portfolio project, a movie recommender trained on MovieLens-20M. I would like to host
  a public demo on Streamlit Community Cloud and would like to know whether the following
  is permitted:
  1. publishing model artifacts derived from the data (item embeddings, model weights, a
     BM25 index built from titles, genres and tags);
  2. including the titles and genres from movies.csv;
  3. optionally, showing a small sample of anonymised user rating histories.
  The repository does not redistribute the dataset (users download it from grouplens.org),
  cites Harper & Konstan (2015), and implies no endorsement.

  Thank you,
  Yi Zhao
  ```

**跑批量训练之前**（上一步需要）

- [ ] 注册 Modal 账号：`uv tool install modal && modal setup`，然后在 Settings → Usage 设置 workspace budget（例如 \$15）。
- [ ] 注册 W&B 并获取 API key，用 `modal secret create wandb WANDB_API_KEY=...` 存到 Modal（或用 `WANDB_MODE=offline` 跳过 W&B）。

**之后**

- [ ] S4 之前：在笔记本上执行 `brew install libomp`。
- [ ] S5 之前：准备一个设置了月度消费上限的 Anthropic API key（`ANTHROPIC_API_KEY`）。

## 8. 风险与砍需求顺序

进度落后时的砍需求顺序：先砍消融 (c) 里的 C2，然后依次是 A0 / A2 / A4 这几个背景组、不带 EASE 的排序器变体、demo 里的在线 LLM 调用（只保留缓存的示例）、demo 里的 Milvus（换成 numpy 实现；Milvus 保留在离线一致性基准里），最后是云 GPU（改在 MPS 上训练，每次约 40 分钟）。

| 风险 | 应对 |
|---|---|
| Liang 划分复现有偏差（数量对不上） | S1 的 golden 测试和真实数据计数测试，在 S2 之前修好 |
| EASE 达不到校准数字 | 在报告任何模型数字之前修好指标和划分的 bug |
| 目标达不到 | 对照已发表的参考数字汇报；绝不为了凑目标而改协议或指标 |
| 批量训练在 Modal 上失败或卡住 | 重跑时跳过已完成的运行；先做 2 个 epoch 的试跑；RunPod（方案 B）作为备选 |
| Milvus Lite 在 Community Cloud 上偏慢：实测混合检索 p95 176 ms，占了 200 ms 预算的大部分（E005） | S6 在主机上实测 numpy 精确稠密 top-k 和 scipy BM25（结果相同；FLAT 重合度 1.000）；如果 Milvus 加排序器达不到 D-013，demo 改走 numpy / scipy，Milvus 作为备选引擎展示并附上其延迟 |
| demo 超过 690 MB | 探测峰值 482 MB（E005）；精简资源包；Hugging Face Spaces |
| 许可不允许附带片名或衍生产物 | 现在就发 D-003 邮件；S6 里准备了后备方案 |

## 9. 完成标准

- 每个结果都来自有记录的运行，写成 `mean ± sd [95% CI]`，协议为 `Liang-SG-strict`。
- 表 (a)、(b)、(c) 和排序器的表都已完成，四个目标都对照参考数字汇报，无论达到与否。
- 在线 demo 有实测的 p95 延迟；README 里有 GIF；附上 W&B 报告链接；GitHub 上的 CI 通过。
- README 和简历描述里的每一条结论背后都有实测数据。
