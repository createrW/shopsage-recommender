# ShopSage Recommender — 项目记忆

[English](memory.md) | **简体中文**

**更新：** 2026-10-24，S2 批量训练完成后（E007）。先读这份，再读 [plan.zh-CN.md](plan.zh-CN.md)；两份合起来就是下一步的依据。这份文档记录事实、规则和决策；plan 里是步骤和勾选项。

## 1. 项目概况

- **是什么：** 基于 MovieLens-20M 的两阶段推荐系统：Two-Tower 稠密召回和 BM25 稀疏召回，外加二者的 RRF 混合；LightGBM 重排器；可选的 Claude Haiku 查询扩展；Milvus Lite 作为服务端检索引擎；只用 CPU 的 Streamlit demo。所有对外报告的数字都来自有记录的运行。
- **为什么做：** 这是你 MLE 作品集的旗舰项目；README 是评审者看到的页面。
- **负责人：** Yi Zhao（USC），所有 GitHub push 也由本人完成。
- **位置：** `/Users/zy/Project/new_sys/projects/shopsage-recommender`，`main` 分支。提交 `3e0f08f`（Day 1）和 `05f74c4`（S1）已 push 到 `https://github.com/createrW/shopsage-recommender`（CI 通过）；S2 已在本地提交为 `fefcf6c`（未 push）；汇总修复和 E007 文档尚未提交。
- **状态：** S0、S1 和 S2 已完成；批量训练记为 E007，含消融 (a)。下一步是 S3（BM25、Milvus 服务索引、表 (b)）。2026-10-23 按效率优先重新规划（plan.zh-CN.md §3）。

## 2. 仓库之外的输入

| 输入 | 位置 | 说明 |
|---|---|---|
| 最初的 brief：范围、固定技术栈、10 天计划、目标 | 第一次工作会话中粘贴（2026-10-18） | 摘要见 §3 |
| 更新后的范围：Milvus、一条 BM25 召回路径、新的目标 | 2026-10-23 粘贴 | 摘要见 §3；与最初 brief 不同之处以它为准 |
| 代码规范 | `/Users/zy/Project/new_sys/code_standard.md` | 目录结构已遵循；偏差见 design.md §15 |
| 托管方案 | `/Users/zy/Project/new_sys/cloud_hosting_plan_2026-10-17.md` | 决定云端部署方式（plan.zh-CN.md §6）；`/Users/zy/Project/new_sys/cloud_sync/personal_extras/` 下有一份逐字节相同的副本（sha256 一致） |

## 3. 需求摘要

- **更新后的范围（2026-10-23），以它为准：** MovieLens-20M（Amazon Reviews 2023 Toys 为可选项，已砍掉）。两条召回路径：Two-Tower 稠密召回和 BM25 稀疏召回；LightGBM 排序器；LLM 查询扩展（用 vLLM 跑 Qwen2.5-0.5B、GPT-4o-mini 或 Claude Haiku API，选了 Haiku）。技术栈：PyTorch 2 加 torchrec 或手写双塔、LightGBM、sentence-transformers BGE-M3、bitsandbytes；FAISS 或 Milvus Lite，外加 Elasticsearch BM25；Polars 和 DuckDB；W&B 或 MLflow；Streamlit 和 Jupyter EDA notebook；NDCG@10、Recall@50 和 HitRate@50，带 bootstrap 95% 置信区间。目标：Recall@50 ≥ 0.55；排序后的 NDCG@10 ≥ 0.32；每次查询端到端延迟 ≤ 200 ms（召回 50，排序到 10）；难负样本挖掘消融显示 Recall@50 提升 +0.08 以上。资源：一块 RTX 3060/4060（8–12 GB）训练 10–15 小时，16 GB 内存，15 GB 磁盘。
- **最初的 brief（2026-10-18）：** 同一个系统，用 FAISS，有一轮难负样本、Top-500 → LightGBM 排序器、LLM 扩展、三个消融（(a) 负样本，(b) dense vs BM25 vs hybrid，(c) 有无 LLM 扩展）、Streamlit demo（user-id 下拉框、该用户最近 5 条评分、带分数的 Top-10、LLM 开关），以及带表格和置信区间的 README。不在范围内：在线服务、深度排序模型、强化学习、A/B 基础设施。当时的目标是 Recall@50 0.42 → 0.58 和 NDCG@10 0.32。
- **按效率优先做出的取舍：** 见 plan.zh-CN.md §3。

## 4. 长期规则

1. **没有运行就没有数字。** 指标只能来自有记录的运行（experiment_log.md 规则 1）；在此之前结果单元格写 `pending`；文献数字要标注并注明出处。
2. **目标不是结果。** 你定的目标对照已发表的参考数字汇报，无论达到与否。
3. **许可。** MovieLens 数据永不提交，因为 GroupLens 禁止未经许可的再分发；`data/` 已被 git 忽略；除非 D-003 允许，公开 demo 不附带任何逐用户评分数据。
4. **demo 从不导入 torch**（Streamlit Community Cloud：只有 CPU，保证 690 MB 内存）；CI 会检查这一点。
5. **由负责人 push。** agent 只在被要求时提交，从不 push；git 命令一条一条地执行（Day 1 时一条 `mkdir && git init` 的复合命令被拒绝过）。
6. **训练以交接方式完成：** 提供一条云端命令和显存表，由负责人运行。
7. **云端部署遵循托管方案**（plan.zh-CN.md §6）；厂商数据写进文档前要重新核对。
8. **本项目代码用 Polars，不用 pandas**（pymilvus 自己会导入 pandas）；用 DuckDB 对 parquet 做 SQL 查询。
9. **fold-in 规则**（Liang 协议）：留出用户的用户向量和特征只来自该用户的 fold-in 物品；物品统计、BM25 标签和所有训练出的模型只用训练用户的数据。Day 1 设计中的 as-of 访问器只适用于 leave-one-out 划分，而这个划分已不再使用。
10. **测试先行（TDD）。** 在把 warning 当作错误的设置下测试保持通过；ruff、mypy 和 `uv lock --check` 保持干净。
11. **实验日志只追加。** 更正写成新条目；决策行可以修订，但要附上带日期的说明。
12. **状态文档中英双语。** plan.md 和 memory.md 各有一份 `.zh-CN.md` 对应版本；两者一起更新。
13. **工作循环。** 每一步开始前读 plan.md 和 memory.md；每一步结束后更新这两份文档（连同中文版），并告诉负责人接下来要做什么。
14. **效率优先。** 选能如实测出目标指标的最简路径，每一处删减都记在 plan.zh-CN.md §3。速度绝不能换来更弱的评估：全量候选排序、不泄漏目标、不写未经实测的结论。

## 5. 关键事实

### Liang 划分（S1 实测，E004；与已发表的计数完全一致）

| 事实 | 数值 |
|---|---|
| 预处理 | 评分 ≥ 4.0；用户至少 5 个正样本；不过滤物品：9,990,682 个正样本、136,677 个用户、20,720 部电影（均与发表值一致） |
| 划分 | 116,677 个训练用户，10,000 个验证用户和 10,000 个测试用户；训练物品库 20,108 个物品（与发表值一致）；训练行 8,538,846 |
| 留出行 | 验证 fold-in 589,887 / 目标 142,514；测试 fold-in 579,008 / 目标 139,775；每个留出用户都有目标 |
| `split_sha256` | `4f93a0e6f6119f20d9aabedc633c82bf37965bf5fbf8a1d2ba72ed94316e4886` |
| 配置哈希 | `2f24b2c594e1`（`configs/data/movielens_20m_liang.yaml`） |
| `meta.json` 的 `created_at` | 2026-10-23T20:06:05+00:00 |
| `prepare` | M5 上 5.1 秒，峰值 RSS 2.2 GB |

### 目前的结果（E004，`Liang-SG-strict`，10,000 个测试用户；mean [95% CI]）

| 模型 | Recall@20 | Recall@50 | NDCG@10 | NDCG@100 | HR@50 |
|---|---|---|---|---|---|
| MostPop | 0.1618 | 0.2352 [0.2310, 0.2396] | 0.1275 [0.1241, 0.1311] | 0.1906 | 0.7425 |
| EASE，λ = 1,000（在 V_A 上选出） | 0.3893 | 0.5210 [0.5156, 0.5264] | 0.3280 [0.3231, 0.3331] | 0.4201 | 0.9246 |
| 发表值（Steck 2019，Table 1）：popularity / EASE | 0.162 / 0.391 | 0.235 / 0.521 | — / 0.329（`Liang-SG-fit`） | 0.191 / 0.420 | — |

- 校准关卡已通过：两个基线都复现了发表的数字（Recall@50 差距在 0.0002 以内）。
- EASE 已经达到 NDCG@10 的目标（0.32）；Recall@50 的目标（0.55）比 EASE 高 0.029。该协议下已发表的最高 Recall@50 是 RecVAE 的 0.553。
- `shopsage evaluate`：MostPop 加三个 λ 的 EASE 网格用时 211 秒，峰值 RSS 3.5 GB（EASE 的 Gram 矩阵和权重矩阵在 float32 下各 1.6 GB）。10,000 个用户时 Recall@50 的 bootstrap 半宽约 ±0.005。
- 笔记本上的部署探测（20,108 个物品，合成的稠密和稀疏向量）：混合检索 p95 34–44 ms，稠密检索 p95 18–24 ms；Streamlit 加应用的峰值内存约 450 MB。
- **Streamlit Community Cloud 上（E005）：** import pymilvus 1.9 s，建库 7.9 s；稠密 p50/p95 76.3/135.2 ms，稀疏 63.0/94.6 ms，混合 118.0/176.3 ms；重合度 1.000；峰值 RSS 482 MB。内存在 690 MB 之内；但光混合检索的 p95 就占了 200 ms 预算中的 176 ms，所以 S6 要实测 numpy / scipy 服务路径，必要时改用它。
- **Two-Tower 批量训练（E007，Modal A10，A1 和 A3 各 3 个种子）：** A1 Recall@50 0.5361 ± 0.0005、NDCG@10 0.3351 ± 0.0005、Recall@20 0.3997、HR@50 0.9371；A1 − EASE：Recall@50 +0.0151 [+0.0119, +0.0184]，NDCG@10 +0.0071。A3 − A1 = −0.0020 [−0.0033, −0.0007]（困难负样本 +0.08 的目标**没有达到**）；A0 0.3111（logQ 带来 +0.225）；A2 0.5342；A4 0.5333。τ 0.05，lr 3e-3；最佳 epoch 22–30。GPU 峰值 0.98 GB（A1）到 3.21 GB（A4）；A1 每 epoch 21.5 s；2.5 个 A10 小时，约 \$2.75。S3–S6 使用的代表运行：`A1-s42`（`artifacts/runs/A1-s42/`）。
- **Two-Tower 检查（E006，MPS，不算结果）：** A1 训练 3 个 epoch 后测试 Recall@50 0.5145 [0.5088, 0.5199]、NDCG@10 0.3163，仍在上升；M5 上每 epoch 129–146 s；峰值显存 A1 2.1 GB、A2 2.2 GB、A4 5.0 GB。

### Day 1 运行（原始数据，以及保留但不用于结果的 leave-one-out 划分）

| 事实 | 数值 |
|---|---|
| 原始数据 | 20,000,263 条评分（0.5–5.0 星）；138,493 个用户（每人 ≥ 20 条评分；没有人口统计信息）；27,278 部电影（其中 26,744 部有评分）；465,564 条标签记录；1995-01-09 至 2015-03-31 |
| 压缩包 md5 | `cd245b17a1ae2cc31bb14903e1204af3` |
| Day 1 划分 | 9,977,455 个正样本；136,674 个用户 × 13,681 个物品；带 splitmix64 并列打破的 leave-one-out；`split_sha256` 为 `5fa6cbe8f702…`（2026-10-23 重新运行 EDA 后不变）；配置哈希 `aec7ddd4dc98` |
| `prepare` | M5 上热缓存 3.3 s、冷启动 9.1 s，峰值 RSS 2.3 GB（E001） |
| MPS（M5，B = 4,096，bf16） | 每步 37.25 ms → in-batch 每 epoch 88 s；带 4,096 个采样负样本时峰值 2.30 GB（in-batch 为 1.16 GB）；30 个 epoch 约 44 分钟（E002） |
| 流行度 | 前 100 部电影占正样本的 23.5%；在 26,744 部有评分电影上的 Gini 为 0.90 |
| 本地磁盘（2026-10-23） | `data/raw` 1.0 GB，`data/processed` 138 MB，`data/processed/liang` 146 MB，`.venv` 946 MB |
| 测试 | 122 个通过（2026-10-24），含较慢的真实数据测试 |

## 6. 决策

当前的决策；完整文字和修订说明见 experiment_log.md 的 Decisions 表。当前使用的协议是 `Liang-SG-strict`（experiment_log.md 的 Protocols 一节）。

| 编号 | 主题 | 状态 | 决策 |
|---|---|---|---|
| D-001 | 评估协议 | 2026-10-23 已定（负责人没有否决，直接开始了 S1） | `Liang-SG-strict` 是唯一的主表协议：精确复现 Liang 等人的划分（种子 98765，10,000 个验证用户和 10,000 个测试用户，fold-in 与目标按 80/20 划分）。校准已通过（E004）。leave-one-out 划分和 `Liang-SG-fit` 砍掉。 |
| D-002 | 物品过滤 | 2026-10-23 修订 | 不过滤物品（`min_item_interactions: 1`），即 Liang 的预处理。正样本 ≥ 5 的过滤只适用于不再使用的 Day 1 划分。 |
| D-003 | Demo 数据许可 | 待定，S6 之前 | 默认：访客"选 5 部电影"画像，启动时从 GroupLens 获取片名，不附带逐用户数据；由负责人给 GroupLens 发邮件（plan.zh-CN.md §7） |
| D-004 | 用户塔 | 2026-10-23 已定，随 D-001 | 归纳式：≤ 50 个 fold-in 物品的历史物品 embedding 均值加 log 历史长度 → MLP → 128 维；没有用户 ID 表 |
| D-005 | 难负样本消融 | 2026-10-23 修订 | 主对比：A1（in-batch + logQ，"无"）对 A3（A1 + 每行的挖掘负样本，取自 A1 种子 42 模型的第 101–500 名，"有"），各 3 个种子；A0（无 logQ）、A2（A1 + 2,048 个均匀负样本，即 MNS）和 A4（全量 softmax）各 1 个种子，作为背景 |
| D-006 | 排序器数据 | 2026-10-23 修订 | V_A / V_B = 10,000 个 Liang 验证用户按 50/50 划分（种子 42）；候选 = 稠密 Top-500 ∪ BM25 Top-100；标签 = 每个用户留出的 20%；约 11 个特征，含一个稀疏 EASE 分数；LambdaRank，3 个 LightGBM 种子，分别带和不带 EASE 特征 |
| D-007 | 指标与置信区间 | 2026-10-23 修订 | 主表为截断 Recall@50 和 NDCG@10，另报 Recall@20、NDCG@100 和 HR@50，基于 10,000 个测试用户；百分位 bootstrap（2,000 次重采样，种子 42），组间配对；每个要下结论的对比 3 个种子，背景组 1 个，并注明 |
| D-008 | 检索引擎 | 2026-10-23 修订 | 离线：numpy 精确 top-k，在截断前屏蔽 fold-in 物品；服务端：Milvus Lite（FLAT；稠密、稀疏 BM25 向量和 RRF 混合检索），载入后 flush，并与精确路径做一致性测试；FAISS 不再是直接依赖（Milvus Lite 3.x 底层用 faiss-cpu），ANN 研究砍掉 |
| D-009 | 基线 | 2026-10-23 修订 | 每张表都有 MostPop（训练用户上的流行度）和 EASE（稠密 B，λ 在 V_A 上调）；ItemKNN 砍掉 |
| D-010 | BM25 与 LLM | 2026-10-23 修订 | BM25 基于片名、类型、top genome 标签和训练用户的标签，用 scipy 实现（不用 Elasticsearch）；用户查询是其 fold-in 物品的文本；LLM：`claude-haiku-5-5`，结构化输出，effort 设为 `low`，有缓存；消融 (c) = 约 1,000 个标签查询，C0 BM25 原始查询对 C1 BM25 + 扩展（有时间再加 C2 混合）；不用 vLLM、Qwen 和 GPT-4o-mini |
| D-011 | 时间戳 | 2026-10-23 起不再适用 | 只适用于 leave-one-out 划分 |
| D-012 | 排期 | 2026-10-23 被取代 | 由步骤 S1–S7 取代（plan.zh-CN.md §4） |
| D-013 | 延迟目标 | 2026-10-23 已定 | 召回（稠密 + BM25）加 LightGBM 重排到 10 的 p95 ≤ 200 ms，在 demo 主机上测量，不含 LLM，LLM 单独汇报 |
| D-014 | 效率优先的范围 | 2026-10-23 已定 | 砍掉的内容见 plan.zh-CN.md §3；一次 Modal 批量训练，取代逐次运行 |

**在 S7 重写之前视为已被取代的部分：** design.md 的 §4.1、§4.3、§5–§7、§9、§11、§16 和 §17，以及 §4.4、§8、§10、§12 和 §13 中与评估协议相关的部分（文件开头的横幅已说明）；api_spec.md 的 §5.6、§6.2 和 §7（它的 §2.7、§6.1 和 §9 已改为描述 S1）。README 在 S1 中已重写，是最新的。旧章节与本节不一致时，以本节为准。

## 7. 环境与坑

- **笔记本：** Apple M5，16 GB 统一内存，没有 NVIDIA GPU。uv 0.12.22；项目 venv 用 Python 3.12.15，装了核心依赖和 `dev` extra（包括 scipy、pymilvus、milvus-lite、pandas 和 faiss-cpu；没有 torch，也没有 streamlit）。
- **S1 之后的 extra：** `train`、`llm`、`demo`、`eda`、`dev`；`cpu` / `gpu` 这两个 FAISS extra 及其冲突声明已删掉。`uv sync` 是精确同步：一条命令里要写全所有 extra。`uv run --extra demo ...` 会临时加上 Streamlit 跑一次（`make demo`），之后再执行 `uv sync --extra dev` 会把它删掉。
- **Milvus Lite（3.2.2，配 pymilvus 3.0.2）：** 基于 faiss-cpu 的纯 Python 实现，运行在调用它的进程里。只支持 FLAT 索引；稀疏向量和混合检索可用。**插入数据后要调用 `client.flush(collection)`**：没 flush 的行每次查询都会在 Python 里逐行扫描（20,108 个物品时稠密检索 p50 181 ms，flush 后 9–17 ms）。pymilvus 会导入 pandas（热启动约 1 秒，第一次冷启动约 10 秒）。文档列出的系统是 Ubuntu ≥ 20.04 和 macOS ≥ 11，没有 Debian 11，所以主机上能否运行要靠部署探测确认。
- **无界面运行 Streamlit：** `streamlit.testing.v1.AppTest.from_file("src/shopsage_rec/serving/app.py").run()` 会在当前进程里执行应用；本地测量探测时用的就是它。
- **macOS 上的 LightGBM** 需要 `brew install libomp`（截至 2026-10-23 尚未安装）；Linux 和 Streamlit Cloud 用 `libgomp1`。
- **Linux 上的 torch** 来自 cu126 索引（NVIDIA 驱动 ≥ 525）；Blackwell 架构的 GPU 需要 cu130 索引。
- **AMP：** 只有 `torch.cuda.is_bf16_supported(including_emulation=False)` 为真（计算能力 ≥ 8.0）时才用 bf16；T4 用 fp16 加 `GradScaler`；MPS 上跑的是 bf16 autocast。
- **MPS 不支持 float64：** 放到 MPS 上的张量必须是 float32（logQ 用的物品频率向量踩过这个坑）。
- **Modal 1.6：** `Image.uv_sync` 只装依赖、不装项目本身，所以源码用 `add_local_dir` 加 `PYTHONPATH` 放进去；GPU 名称是 `"T4"`、`"L4"`、`"A10"`、`"L40S"`、`"A100"` 等（截至 2026-10-24 文档里没有 `"A10G"`）；Volume 写入后要 `commit()`，读之前要 `reload()`。
- **检查 `sys.modules` 的测试** 要放在子进程里跑：同一会话里的其他测试会 import torch。
- **MPS 计时：** 计时步数少于 30 噪声太大（3 步的测试看起来比实际快了 30%）。
- **Polars 的 `is_in`：** 用同类型的 Series 去测一列已被弃用；要写成 `.is_in(series.implode())`。
- **Docker：** 2026-10-18 和 2026-10-23 时 daemon 都没有运行，所以镜像从未在本地构建过；只通过了静态检查。
- **Git：** 没有配置全局身份；Day 1 的提交是在命令里指定作者的。`gh` 已经能访问 `createrW/shopsage-recommender`（用 `gh run list` 查看 CI 状态）。
- **Streamlit Community Cloud** 从它找到的第一个依赖文件安装，先找入口文件所在目录，再找仓库根目录，优先级为 `uv.lock` > `Pipfile` > `environment.yml` > `requirements.txt` > `pyproject.toml`；因此用 `src/shopsage_rec/serving/requirements.txt`（用 `make demo-requirements` 重新生成；它和 `uv.lock` 不一致时 CI 会失败）。
- **`claude-haiku-5-5`：** 提示不超过 100K token 时，每百万输入 / 输出 token 为 \$0.10 / \$0.50；结构化输出用 `output_config.format`；默认开启自适应思考且 effort 默认为 `medium`，所以要设为 `low`；从 2026-10-21 起，任何 `temperature`、`top_p` 或 `top_k` 都会返回 400；没有服务端的拒绝回退，所以 `refusal` 这个停止原因要在客户端代码里处理（claude-api 技能，缓存日期 2026-10-06）。
- **厂商数据（核对结果）：** Streamlit Community Cloud（2026-10-18）：保证 690 MB，最高 2.7 GB，≤ 2 个核，默认 Python 3.12，Debian 11，闲置 12 小时后休眠。Modal（2026-10-23）：Starter 每月含 \$30 额度；A10 \$1.10/h，T4 \$0.59/h，L4 \$0.80/h，A100 40 GB \$2.10/h；支持 workspace budget。Hugging Face（2026-10-23）：CPU basic，2 vCPU、16 GB，免费；T4 small \$0.40/h。RunPod（2026-10-23）：RTX 4090 在 Community Cloud 上 \$0.34/h，在 Secure Cloud 上 \$0.89/h；闲置卷盘每 GB 每月 \$0.20。

## 8. 文件位置

| 路径 | 内容 |
|---|---|
| `README.md` | 首页：含实测基线的结果、架构、快速上手、云端训练与显存、数据集、实验、demo、设计理由、局限性、路线图 |
| `docs/experiment_log.md` | 规则、评估协议、TARGETS、决策 D-001 到 D-014、实验矩阵、条目 E001–E007 |
| `docs/design.md`、`docs/api_spec.md` | Day 1 的设计，以及产物 / CLI 约定（v0.4），部分已被取代（§6） |
| `docs/plan.md`、`docs/memory.md` | 步骤计划与进度；本记忆文档（各有一份 `.zh-CN.md` 对应版本） |
| `src/shopsage_rec/cli.py`、`config.py` | Typer CLI（`prepare`、`stats`、`evaluate`；默认值跟随 Liang 划分）；`DataConfig`、`SplitConfig`（`leave_one_out` 或 `liang`） |
| `src/shopsage_rec/data/` | `loader`、`preprocess`、`split`（Day 1）、`liang`（`liang_split`、`load_liang`、`partition_val_users`）、`pipeline`（两种划分共用的 `run_prepare`） |
| `src/shopsage_rec/evaluation/` | `metrics`（带屏蔽的 top-k、Recall / NDCG / HR）、`bootstrap`、`runner`（`evaluate_baselines`） |
| `src/shopsage_rec/models/baselines.py` | `MostPop`、`gram_matrix`、`EASE` |
| `src/shopsage_rec/serving/` | `app.py`（Streamlit 入口：探测程序）、`probe.py`（`run_probe`）、`requirements.txt`（自动生成） |
| `src/shopsage_rec/features/items.py` | 物品塔输入：类型、年份分桶、genome PCA-64，缓存到 `<processed>/item_features.npz` |
| `src/shopsage_rec/models/` | `baselines.py`；S2：`two_tower.py`（`TwoTower`、`sample_history`）、`losses.py`（`sampled_softmax`、`PositiveSet`）、`numpy_tower.py`（不依赖 torch 的用户塔）、`trainer.py`（`TrainConfig`、`ARMS`、`train_two_tower`）、`mining.py`、`sweep.py`（任务计划，纯 Python） |
| `src/shopsage_rec/{retrieval,ranking,llm}/` | 只有 docstring 的占位包（S3–S5） |
| `configs/data/` | `movielens_20m_liang.yaml`（默认，哈希 `2f24b2c594e1`）和 `movielens_20m.yaml`（Day 1，哈希 `aec7ddd4dc98`） |
| `configs/train/lgbm_ranker.yaml` | 排序器草稿（S4）；Two-Tower 没有 YAML：用 `TrainConfig` 和 CLI 选项，快照写进每个 `metrics.json` |
| `configs/deploy/streamlit.yaml` | demo 配置草稿：资源包、产物、Milvus Lite 检索、LLM 扩展 |
| `scripts/` | `download_data.sh`、`prepare_data.sh`（默认用 Liang 配置）、`bootstrap_cloud.sh`（train + llm extras，Liang 划分）、`bench_two_tower_step.py`、`train.sh`（顺序批量训练）、`modal_train.py`（Modal 批量训练：Volume `shopsage-data`，GPU `A10`） |
| `tests/` | 122 个测试，其中有 `test_two_tower.py`、`test_trainer.py`、`test_sweep.py`、`test_item_features.py`、 `test_liang.py`（含真实数据计数测试）、`test_liang_pipeline.py`、`test_metrics.py`、`test_baselines.py`、`test_probe.py` |
| `notebooks/01_eda.ipynb` | 基于 Day 1 划分的 EDA，已执行；图在 `docs/figures/` |
| `Makefile` | 22 个目标（`make help`），包括 `evaluate` 和 `demo-requirements` |
| `.github/workflows/ci.yml` | ruff、ruff format、mypy、pytest；不装 torch 安装 demo 依赖；核对 `serving/requirements.txt` 与 `uv.lock`；在 Linux 上运行 Milvus Lite 探测 |
| `Dockerfile`、`.dockerignore` | CPU 版 demo 镜像（还没构建过）；Hugging Face Spaces 的后备方案 |
| `.streamlit/`、`packages.txt` | Community Cloud 设置、secrets 模板、`libgomp1` |
| `data/`、`artifacts/`（被 git 忽略） | `data/raw/ml-20m/`、`data/processed/`（Day 1）、`data/processed/liang/`；`artifacts/eval/local-20261023-1306/`（E004）；`artifacts/runs/`（E007：13 个批量训练运行和 `sweep_summary.json`，0.6 GB；E006 的检查：`smoke-A1-mps`、`check-A1-full3-mps`） |

## 9. 恢复工作

1. 读 plan.zh-CN.md 和本文件（规则 13）；从 plan.zh-CN.md §4 里第一个没打勾的步骤继续。
2. 检查工作区：

```bash
cd /Users/zy/Project/new_sys/projects/shopsage-recommender
git status && git log --oneline -3
uv sync --extra dev                              # 核心 + dev；要在 MPS 上用 torch 再加 --extra train
make check                                       # ruff、mypy 和快速测试
uv run shopsage stats | grep split_sha256        # Liang 划分：4f93a0e6f611...
```

3. 这一步完成后：在 plan 里打勾，更新本文件（事实、决策、历史），把两份文档同步到 `.zh-CN.md` 版本，在 experiment_log.md 里记录运行，然后告诉负责人接下来要做什么。

## 10. 历史

| 日期 | 事件 |
|---|---|
| 2026-10-17 | 负责人写下托管方案 |
| 2026-10-18 | 完成 Day 1：脚手架、数据流水线、划分、EDA、文档、CI；MPS 基准（E002）；最后一轮评审（104 条问题）处理完毕，D-005 和 D-006 随之修订；提交 `3e0f08f`（21:13 PDT） |
| 2026-10-23 | 用中英文写好 plan.md 和 memory.md；托管方案改为引用新路径；重新核对厂商价格 |
| 2026-10-23 | 负责人给出更新后的范围（Milvus、一条 BM25 召回路径、新的目标）和两条规则（效率优先；工作循环）；重新规划为步骤 S1–S7，以 Liang 协议为默认方案 |
| 2026-10-23 | 负责人把 Day 1 push 到 `createrW/shopsage-recommender`（CI 通过），并回复"go S1" |
| 2026-10-24 | 负责人把 S1 push 为 `05f74c4`；CI 通过，包括 Linux 上的 Milvus Lite 探测 |
| 2026-10-24 | 负责人在 Community Cloud 上部署了探测（E005）并说"go s2"；S2 代码完成：Two-Tower A0–A4、挖掘、批量训练计划、Modal 脚本；MPS 检查（E006）；122 个测试通过 |
| 2026-10-23 | S1 完成：Liang 划分复现了已发表的计数；`shopsage evaluate`；MostPop 和 EASE 复现了已发表的 Recall@50（E004）；去掉 FAISS，加入 Milvus Lite；在笔记本上实测了 Milvus Lite 探测；README 和 experiment_log 已同步 |
| 2026-10-24 | Modal 试跑（`--epochs 2 --tune-epochs 1`）：镜像构建和 Modal 上的数据准备都正常；A10 上 A1 每 epoch 21.5 s，峰值 0.98 GB；之后本地反序列化汇总里的 `torch.__version__`（`TorchVersion` 类型）失败；已改为返回纯 JSON |
| 2026-10-24 | 负责人在 Modal 上跑完整套批量训练（13 次运行，2.5 个 A10 小时）；记为 E007：A1 超过 EASE（Recall@50 0.5361 对 0.5210），挖掘的困难负样本没有提升（A3 − A1 −0.0020），logQ 带来 +0.225；S2 完成 |
