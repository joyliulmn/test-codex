# V1.X 全A股 OHLCV 数据源 V0.2

这是V1.X从“看到股票后分析”走向“系统主动从全市场发现”的第一层数据地基。

## V0.1已经做什么

- 东方财富/AKShare一次抓取**全部沪深京A股**当日行情快照。
- 把日线写入本地 SQLite：`data/v1x_market.sqlite`。
- 支持从2020年开始逐股历史回填，默认保存**不复权**日线。
- 历史回填支持断点续跑；网络失败股票会单独记录。
- 初版V1.X客观特征扫描：位移、量能、振幅、攻击K、攻击后接受、缩量/收敛，以及“第二次点火前置窗口”。
- 输出CSV，后续可交给V1.X深度分析进一步压缩到执行池。

## V0.2防漏审计

- 新增`price_attack_k`，按收盘相对前收盘的正向位移识别；涨幅达到5%即进入
  攻击登记，不再因`C <= O`的跳空假阴而漏扫。为兼容既有消费者，
  `attack_k == fire_k`保留原定义，新增逻辑只读取`price_attack_k`。
- 优先使用逐日权威涨跌停元数据，缺失时才按主板/ST/创业板/科创板/北交所
  的标准规则回退，并显式标注来源；据此计算`is_limit_up`、`is_first_limit_up`
  与`limit_up_streak`，避免直接混排10%/20%/30%。
- 生产`update`会在收盘后额外抓取东方财富当日封板池，把封板收盘价和接口报告的
  `连板数`落为`EASTMONEY_LIMIT_UP_POOL`时点证据。池代码必须属于当日完整快照，
  池价格必须与收盘价逐只一致；行数、来源和内容哈希写入更新审计，扫描时复核。
  空池、日期错配、字段漂移或价格错配均失败关闭，不能用“0只涨停”掩盖接口故障。
- 上述接口只承诺“近期”数据且不包含ST。程序因此只请求已确认的中国市场当日，
  不拿它任意回放历史；主板ST的当日5%状态另由同日全市场名称落库。启用后次日可
  自然计算ST连板；若此前只有当前名称回填、无法证明前一日ST状态，则输出
  `UNKNOWN_PREVIOUS_LIMIT_STATE`，不伪称首板或一板，但仍强制进入实名台账。
- 所有首板/连板进入`mandatory_accounting`；高质量首板、连板和点火前置窗口
  进入`mandatory_compare_seed`。强制比较资格只防漏，不自动授予1U。
- 同时输出原全局排名`global_rank`、防漏优先级`priority_rank`和通道内排名
  `lane_rank`；日报不得在强制比较集合形成前执行Top-N截断。
- `report_audit.build_report_plan()`执行“首板/连板实名台账 ∪ 强制比较全集 ∪
  常规Top-N”，并校验`silent_drop_count == 0`。所有未呈现行都必须具有拒绝码。
- 写报告前从当日完整横截面独立核对攻击、首板、连板和强制比较集合；任一代码
  未进入候选即中止扫描并报差集。`first_seen_date/first_planned_date`只在最近一期
  仍有该代码时延续，历史回放不会读取未来日期文件。
- `report_included`仅表示计划呈现，不等于已经送达。发布器成功后必须调用
  `v1xdata receipt --scan <CSV> --receipt-id <ID> --all-included`（也可逐个`--code`）；
  只有带回执 ID 和 UTC 时间戳的行才会生成并延续`first_reported_date`。
- 交易日由持久化`trade_calendar`给出；每日更新只有在上证指数已经形成当日收盘
  日线后才允许写入。休市日、三地任一市场缺失、总覆盖显著少于前一交易日都会
  fail closed，并在`daily_update_audit`留下原因。独立证券全集、带日期个股收盘锚、
  前收连续性、代码集合哈希和完整快照内容哈希共同防止昨日缓存、部分快照或PASS后改写
  自证通过；扫描只接受目标
  会话的PASS审计，并拒绝任何不在日历中的日线日期；前收参考证据覆盖率不足95%
  或普通参考价与库内前收的一致率不足95%同样失败；前一官方会话缺少整日行情、或前一
  会话已记账代码未被当日100%延续时也会失败并转人工复核。已交叉印证的除权参考差异单列
  审计；允许单只例外，但系统性差异超过参考样本5%即失败。官方全集覆盖按“可用行情
  ∪ 东方财富当日全时段停牌台账”逐码核对；停牌接口、分页、字段、日期或区间证据不可验证
  时失败关闭，未被台账证明的缺失/坏行情仍会失败；官方全集内全日停牌比例超过5%也会
  失败并要求人工复核。`suspension_rows`、`suspension_source`、`suspension_hash`、
  停牌比例及官方/前日代码全集指纹写入更新审计，扫描前会从`daily_suspension`和行情表
  重算复核。停牌证据先与同日官方A股全集求交；非停牌官方代码的点时名称必须非空，且
  官方全集名称与行情名称的ST状态必须一致。同日坏重跑不会覆盖已存的有效行情；日历刷新同时撤销
  权威区间内已纠正的伪会话，并把相应原始行移入`daily_bar_quarantine`保留审计。

## 最短启动方式（Windows）

第一次：双击 `scripts/bootstrap_windows.bat`。

首装若当日`update`未通过验证，脚本仍会完成可续跑的历史回填和`doctor`，但不会
执行`scan`或生成扫描文件；请在下一个已收盘交易日运行`scripts/daily_windows.bat`。

以后每个交易日收盘后：双击 `scripts/daily_windows.bat`。

也可以命令行运行：

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -e .

v1xdata update
v1xdata bootstrap --start 20200101 --resume
v1xdata scan
v1xdata doctor
```

> 历史回填是第一次的大任务。AKShare历史接口按股票返回，因此全市场回填会耗时；脚本会断点续跑，不需要一次完成。

## 每日逻辑

收盘后 `v1xdata update` 取得一次全市场快照，并额外取得一次当日封板池，因此
**日常更新不是5000多次请求**。历史逐股接口主要用于第一次建库和补洞。

## 输出

扫描结果写到：

```text
output/v1x_scan_YYYY-MM-DD.csv
```

重点字段：

- `attack_k`（兼容字段，仍等于`fire_k`）
- `price_attack_k`（V0.2收盘位移攻击）
- `effective_pre_close` / `pre_close_source`
- `daily_limit_pct` / `daily_limit_pct_source`
- `limit_up_price` / `limit_up_price_source`
- `reported_limit_up_streak` / `reported_limit_up_streak_source`
- `limit_up_streak_source` / `limit_up_sequence_status`
- `name_source` / `point_in_time_state_unknown` / `future_data_status`
- `days_since_attack`
- `retains_attack_close`
- `center_not_falling_5d`
- `volume_contracting_5d`
- `range_contracting_5d`
- `pre_ignition_window`
- `attack_registry`
- `is_first_limit_up`
- `limit_up_streak`
- `mandatory_accounting`
- `mandatory_compare_seed`
- `mandatory_reason`
- `global_rank` / `priority_rank` / `lane_rank`
- `report_included` / `report_reject_code` / `permission_status`
- `report_mandatory_total` / `report_accounting_total` / `report_included_total`
- `silent_drop_count` / `unrendered_mandatory_codes`

其中 `pre_ignition_window` 对应当前新增的V1.X执行问题：**第一次攻击已被市场接受时，不机械等第二次涨停/一字板才买，而是寻找点火前最后一个可执行窗口。**

## 重要边界

这不是“万能选股公式”。数据层负责发现、分轨和防静默遗漏；最终的L1/H1、
边界迁移、速度/加速度、语义板块共振和0U/1U/3U仍由V1.X解释层继续迭代。

完整设计见 [`docs/DESIGN.md`](docs/DESIGN.md)。
