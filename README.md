# V1.X 全A股 OHLCV 数据源 V0.1

这是V1.X从“看到股票后分析”走向“系统主动从全市场发现”的第一层数据地基。

## V0.1已经做什么

- 东方财富/AKShare一次抓取**全部沪深京A股**当日行情快照。
- 把日线写入本地 SQLite：`data/v1x_market.sqlite`。
- 支持从2020年开始逐股历史回填，默认保存**不复权**日线。
- 历史回填支持断点续跑；网络失败股票会单独记录。
- 初版V1.X客观特征扫描：位移、量能、振幅、攻击K、攻击后接受、缩量/收敛，以及“第二次点火前置窗口”。
- 输出CSV，后续可交给V1.X深度分析进一步压缩到执行池。

## 最短启动方式（Windows）

第一次：双击 `scripts/bootstrap_windows.bat`。

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

收盘后 `v1xdata update` 使用 `stock_zh_a_spot_em()` 一次取得全市场快照，因此**日常更新不是5000多次请求**。历史逐股接口主要用于第一次建库和补洞。

## 输出

扫描结果写到：

```text
output/v1x_scan_YYYY-MM-DD.csv
```

重点字段：

- `attack_k`
- `days_since_attack`
- `retains_attack_close`
- `center_not_falling_5d`
- `volume_contracting_5d`
- `range_contracting_5d`
- `pre_ignition_window`

其中 `pre_ignition_window` 对应当前新增的V1.X执行问题：**第一次攻击已被市场接受时，不机械等第二次涨停/一字板才买，而是寻找点火前最后一个可执行窗口。**

## 重要边界

这不是“万能选股公式”。V0.1先把事实计算正确：数据层负责发现异常和候选，最终的L1/H1、边界迁移、速度/加速度、订单流异常、板块共振和0U/1U/3U仍由V1.X解释层继续迭代。

完整设计见 [`docs/DESIGN.md`](docs/DESIGN.md)。
