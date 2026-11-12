# tool-funnel · 让大工具集的 Agent 少花 Token、少绕弯路

> 一个**漏斗式工具路由器**：当 Agent 手里有几十、几百上千个工具时，不再把全部
> 工具定义塞进每一轮对话。先用「业务域 → 阶段」把范围收窄到几个工具，再按
> 「已调用工具」推荐下一步，并把参数预先填好。**调用越多，链路越准、越快。**

[![tests](https://img.shields.io/badge/tests-136%20passing-brightgreen)](#测试)
[![python](https://img.shields.io/badge/python-3.8%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![deps](https://img.shields.io/badge/runtime%20deps-0-orange)](#设计原则)

**v0.2 新增（重点：MCP）**

- 🔌 **MCP 适配器** —— 把一个或多个 MCP server 的工具目录直接灌进漏斗，
  调用回传自动进入学习闭环。`order.search` 产出的 `orderId` 会落进上下文，
  下一次 `order.get` 直接预填。支持同步/异步 session，零 `mcp` 依赖。
- 🪄 **自动索引（v0.3）** —— 扔进一堆**原始工具**（MCP 列表 / OpenAPI / 一串名字），
  自动推断业务域、阶段、链路、触发词，直接产出可路由的目录。**不用手写配置**，
  自由任务/无边界任务也能立刻用起来。
- 📈 **自适应预算** —— `max_tools` 变成「天花板」而非目标，工具少时不硬凑；
  支持按 token 预算收窄。
- 🧠 **链路探索（bandit）** —— UCB 探索让偶发的更短链路有机会被发现，打破
  「先入为主」。
- 🪓 **链路蒸馏** —— 一键剪掉低权重边，长期运行也不会退化。
- 💾 **冷启动缓存** —— 索引与学习状态落盘，冷启动省去重建。
- 🔍 **阶段自动发现** —— 从历史调用序列聚类推断业务域的阶段，免手写。
- 🧭 **语义域兜底** —— embedding 只兜住「一个触发词都没命中」的长尾查询。
- ✅ **预填接受率学习** —— 记录预填值是否被采纳，好的来源上浮。

---

## 落地即用：扔进工具，自动建好路由（v0.3）

前面的四层漏斗需要一个「已经划分好」的工具目录。现实里你往往只有**一堆工具**：
一个 MCP server 发布的上百个工具、一份 OpenAPI、某个框架的注册表。**自动索引器**
就是把 codebase-memory 对代码库做的事搬到工具集上——**扫一遍，推断结构，交回一个
已经能路由的漏斗**，并把推断结果写成一份普通目录文件，你可以看、可以改、可以版本管理。

```bash
# 三种输入，同一条流水线，不需要写任何配置
tool-funnel index my_mcp_listing.json --out indexed.json   # MCP list_tools 结果
tool-funnel index openapi.json        --out indexed.json   # OpenAPI 文档
tool-funnel index tool_names.json     --out indexed.json   # 就一串工具名
tool-funnel route indexed.json "帮我查一下订单"             # 立刻可用
```

Python 里一行：

```python
from toolfunnel.index import index_tools, tools_from_openapi

funnel, report = index_tools(tools_from_openapi(openapi_doc))
print(report.summary())
# indexed 8 tools into 4 domains (order, shipment, refund, general); 14 stages; ...
result = funnel.route("客户申请退款")     # 直接路由，已经学会 order/refund/shipment
```

它会自动推断四件事，**每一条都写进输出文件**（可解释、可覆盖）：

| 推断项 | 依据 |
| --- | --- |
| **业务域** | 工具名/描述的词频相似度聚类（可选 embedding）；名字前缀是最强信号 |
| **阶段** | 从名字里的**动词**映射：search/list → 定位，get → 查看，create/update → 变更，verify → 核对 |
| **链路（next_hint）** | 同域内「读」类工具 → 指向共享同一对象的「写/核对」类工具 |
| **触发词** | 从各域自己的工具里挖「对外独有」的对象词（用 IDF 抑制套话），层 1 立刻能命中 |

**自由任务 / 无边界任务**：漏斗从不假设任务只属于一个域。当没有触发词命中时，
桥接工具与 `expand()` 仍然可用；索引器还会把**没能清晰归类的工具**统一放进
`general` 域，所以「随便看看」类的查询也有地方落，而不是拿到空集。

**可选让模型帮忙命名**（纯粹锦上添花，不配也能跑）：

```python
def namer(tool_names, descriptions):
    return "售后处理"           # 给这一簇工具起个业务域名

funnel, report = index_tools(raw_tools, llm_namer=namer)
```

在 1000 个真实结构的合成工具上：**索引耗时约 0.7 秒**，推断出 8 个业务域，
与真实划分的**纯度约 89%**，5 个典型查询里 4 个能靠挖出的触发词直接命中。

---

## 问题：工具太多，模型花了冤枉钱

一个真实的业务 Agent（订单、物流、售后、库存、客户、营销、商品、素材、报表、
财务……）很容易拥有 **500～2000 个工具**。业界常见的做法是每轮把**全部**工具
schema 都发给模型。这带来三重代价：

| 代价 | 具体表现 |
| --- | --- |
| **Token 成本** | 1000 个工具 ≈ 5 万+ schema tokens / 轮，每轮都在为「永远不会用到」的工具付费 |
| **推理质量** | 工具越多，模型越容易选错、漏选，或把参数填错 |
| **响应延迟** | 更长的 prompt = 更慢的首 token、更慢的整轮完成 |

`tool-funnel` 的答案不是「丢弃工具」，而是**按需收窄**：每一轮只给模型它这轮
真正用得上的那几个，同时留一条随时可走的逃生通道。

## 效果（可复现）

`python benchmarks/bench.py` 在 1000 个工具的真实规模合成目录上：

```
catalog: 1000 tools | 55142 schema tokens if sent whole | 150 queries
--------------------------------------------------------------------------------
phase             tools surfaced  tok/turn  tok/full    saved  top1 dom   p50 ms
--------------------------------------------------------------------------------
before learn       1000     9.0       504     55142    99.1%    84.5%     ~1-3
after  learn       1000     9.0       504     55142    99.1%    84.5%     ~1-3
--------------------------------------------------------------------------------
next-tool top-3 accuracy (held-out sessions):
  declared transitions  (n= 57): 100.0% ->  86.0%
  emergent transitions  (n= 95):   0.0% ->  33.7%   <- only learnable from observed traffic
```

- **每轮工具 schema token 下降 ~99%**（55k → 约 500）。
- **纯 CPU 路由，单次毫秒级**（本机 1000 工具下 p50 约 1–3 ms，随机器而定），
  不额外调用模型。
- **经验积累看得见**：那些「没有写在配置里、只能从真实调用里学出来」的跨域
  链路（emergent transitions），从 0% 提升到 ~34%。

> 数字随目录/查询集不同而变，但量级一致：**工具越多，漏斗省得越多，而漏斗
> 自身的开销几乎不变。**

---

## 核心思路：四层漏斗

```
用户/Agent 一轮输入
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ L1 业务域网关   DomainGate                                   │
│    关键词 / 策划好的触发词 / 学习到的先验                     │
│    → 命中「订单」「售后」等业务域，缩小到 1~2 个              │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ L2 阶段解析     StageResolver                                │
│    同一业务域内有不同阶段：定位→查看→变更→核对               │
│    → 只暴露当前阶段的几个工具。（关键词命中 / 链路继承 /     │
│       可选地让模型自选，三条路按性价比依次尝试）              │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ L3 链路 + 词法排序   ChainGraph + BM25                       │
│    「刚调用了 A，接下来最可能是 B」——来自跨会话学习的图      │
│    → 既排序当前工具集，又推荐「下一个工具」                  │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ L4 参数预填     PreFiller                                    │
│    order.search 已经产出了 order_id → 直接填进 order.get     │
│    → 少一轮问答、少一次模型往返                              │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
   交给模型的：约 5~10 个工具 schema + 已预填的参数
```

**关键设计：业务域之间没有硬隔离。** 漏斗只做「排序与收窄」，不做「封锁」：

- 每一轮都会附带若干**桥接工具**（bridge），让模型仍能跨域伸手；
- 模型若确实需要某个没被选中的工具，调用 `funnel.expand(name)` 即可取回，
  **而这次纠正本身会被记录下来**，下次同样的场景它就会排得更靠前。

> 换句话说：漏斗猜错时的代价是「一次廉价的补充调用」，而不是「走进死胡同」。

---

## 安装：三种方式，任选其一

零运行时依赖，只需 Python 3.8+。完整说明见 [INSTALL.md](INSTALL.md)。

**① 单文件，无需安装（推荐先试试）** —— 从
[Releases](https://github.com/1428514215-create/tool-funnel/releases) 下载 `tool-funnel.pyz`：

```bash
python tool-funnel.pyz demo      # 自带示例，直接跑给你看
```

不需要 pip、不需要联网、不需要虚拟环境。

**② pip 安装（推荐用于项目集成）**

```bash
pip install tool_funnel-0.2.1-py3-none-any.whl   # 从 Releases 下载
# 或：pip install git+https://github.com/1428514215-create/tool-funnel.git
tool-funnel demo
```

**③ 克隆源码**

```bash
git clone https://github.com/1428514215-create/tool-funnel.git && cd tool-funnel
./install.sh          # 自动装；失败则回退构建 .pyz
```

可选增强（都不装也能跑）：`tool-funnel[yaml]`、`tool-funnel[openai]`、
`tool-funnel[mcp]`。

## 快速开始

先跑一次内置演示看效果（不用写任何代码）：

```bash
tool-funnel demo
tool-funnel catalog --tools 1000 --out big.json   # 造一个 1000 工具的目录
tool-funnel route big.json "客户申请退款"          # 看漏斗怎么收窄
```

再看 Python API：

```python
from toolfunnel import Funnel, ToolSpec, ParamSpec, DomainSpec, SessionContext

tools = [
    ToolSpec(
        name="order.search", domain="order", stage="locate",
        description="按关键字搜索订单",
        keywords=["订单", "搜索", "order"],
        params={"keyword": ParamSpec("keyword", required=True)},
        produces=["order_id"],              # 它会产出 order_id
        next_hint=["order.get"],            # 静态链路先验（冷启动用）
    ),
    ToolSpec(
        name="order.get", domain="order", stage="inspect",
        description="按订单号查看订单详情",
        keywords=["订单", "详情"],
        params={"order_id": ParamSpec("order_id", required=True, aliases=["id"])},
    ),
    # ... 另外 998 个工具
]

funnel = Funnel.from_tools(
    tools,
    domains=[DomainSpec("order", triggers=("订单", "order"), bridges=("logistics",))],
)

# 一轮对话：只拿到相关的几个工具
result = funnel.route("帮我查一下订单 12345 走到哪了")
print(result.summary())
# domains[order=4.24, logistics=1.48] stage=locate tools=6+3bridge of 1000 tokens=492/55142 (99% saved)

# 直接喂给你的 chat completion 调用
schemas = result.schemas()               # OpenAI / Anthropic 通用
```

### 让链路越用越准：把真实调用喂回去

```python
ctx = SessionContext()

# 模型调用了 order.search，产出 order_id
call = funnel.record(
    "order.search",
    args={"keyword": "12345"},
    produced={"order_id": "12345"},
    context=ctx,                          # 写入黑板，供后续预填
)

# 下一轮：漏斗知道「刚查完订单」，于是推荐 + 预填订单详情
print(funnel.chain.successors("order.search", limit=3))
# [ChainStep('order.get', 0.6, 'seed-hint'), ...]

nxt = funnel.route("看看这个订单详情", context=ctx)
print(nxt.get("order.get").prefilled)     # {'order_id': 'context:order_id'}
```

### 逃生通道：模型要一个漏斗没给的工具

```python
match = funnel.expand("refund.create")    # 取回被收窄掉的那个工具
# 这次纠正被记录为一条链路，下回同类场景它会自动上浮
```

### 用声明式文件描述整套工具目录（无需写 Python）

```jsonc
// compose.example.json —— 也支持 YAML（pip install pyyaml）
{
  "config": { "max_tools": 6, "top_domains": 2 },
  "domains": [
    { "name": "order", "triggers": ["订单", "order"], "bridges": ["logistics"] }
  ],
  "stages": [
    { "name": "locate", "domain": "order", "order": 0, "triggers": ["查", "搜索"] }
  ],
  "tools": [
    { "name": "order.search", "domain": "order", "stage": "locate",
      "keywords": ["订单"], "produces": ["order_id"],
      "next_hint": ["order.get"],
      "params": [{ "name": "keyword", "required": true }] }
  ]
}
```

```python
from toolfunnel import build_from_dict, load_document
funnel = build_from_dict(load_document("compose.example.json"))
```

### 命令行

```bash
tool-funnel index  my_tools.json --out indexed.json  # 自动索引原始工具
tool-funnel demo                               # 内置示例，直接看效果
tool-funnel catalog --tools 1000 --out c.json  # 生成合成目录
tool-funnel stats  c.json                       # 目录/业务域统计
tool-funnel route  c.json "查订单"               # 沿漏斗走一遍，看它怎么选
tool-funnel bench  c.json                       # Token 节省基准
tool-funnel learn  c.json corpus.json --out learned.json
tool-funnel discover-stages c.json corpus.json  # 从会话推断阶段
tool-funnel distill learned.json                # 蒸馏学习状态里的链路图
tool-funnel mcp-import listing.json --server ecom  # 展平 MCP list_tools 结果
```

未安装时用 `python -m toolfunnel ...` 或 `python tool-funnel.pyz ...` 完全等价。

### 接入真实 Agent 循环

```python
from toolfunnel.adapters import ToolRunner, openai_tools_payload

runner = ToolRunner(funnel, handlers={
    "order.search": lambda keyword: search_orders(keyword),
    "order.get":    lambda order_id: get_order(order_id),
})

route = runner.begin("帮我查订单 12345")
tools_payload = openai_tools_payload(route)   # 传给 client.chat.completions.create

call = runner.call("order.search", {"keyword": "12345"})
print(call.produced)   # 工具产出已写入上下文，链条与黑板都已更新
```

---

## MCP 接入（v0.2 重点）

[MCP](https://modelcontextprotocol.io) 已经成为给 Agent 暴露工具的标准方式，
而一个 MCP server 很容易发布几十上百个工具——正是本库要解决的场景。适配器
**不引入 `mcp` 依赖**：你把 session 传进来，它按鸭子类型调用，因此官方
`ClientSession`、测试用的假 session、手写 dict 都能用。

```python
import asyncio
from toolfunnel import Funnel, DomainSpec
from toolfunnel.mcp import MCPToolServer, MCPToolRunner, ingest_mcp_servers
from mcp.client.session import ClientSession   # 真实场景

async def main():
    async with ClientSession(...) as session:      # 你的 MCP 连接
        funnel = Funnel()
        server = MCPToolServer("ecom", session=session, default_domain="misc",
                               domain_hints={"order": "order", "ship": "logistics"})

        # 1) 把 server 的工具目录灌进漏斗（内部会调用 list_tools）
        n = await ingest_mcp_servers(funnel, [server])
        print(f"ingested {n} tools")

        # 2) 注册业务域触发词（MCP 不知道你的业务词汇，这一步由你补）
        funnel.gate.add(DomainSpec("order", triggers=("订单", "order")))
        funnel.gate.add(DomainSpec("logistics", triggers=("物流", "包裹")))

        runner = MCPToolRunner(funnel, [server])

        # 3) 每轮只把少数几个工具 schema 交给模型
        payload = runner.tools_payload("帮我查订单 12345")   # 保留原始 inputSchema

        # 4) 调用回传 → 自动记录链路 + 产出进上下文（供下一轮预填）
        await runner.acall("order.search", {"keyword": "12345"})
        # order.search 的 structuredContent 里的 orderId 已进入上下文，
        # 下一轮 logistics.track / order.get 会被预填，无需再向模型要 id
asyncio.run(main())
```

适配器做的事（不只是包一层）：

- **展平 schema** 成漏斗的 `ToolSpec`，并从名字前缀猜测业务域（`order.search` → `order`）；
- **保留原始 MCP `inputSchema`**，所以 `oneOf`、嵌套对象等复杂结构原样交给模型；
- **把 `structuredContent` 的标量字段提到顶层**，这是预填端到端可用的关键；
- **合并多个 server**，自动处理重名与命名空间（`server__tool`）。

不想连真实 server 也能跑通全套：`python examples/mcp_demo.py`（用内置假 session）。

---

## 它是怎么「越用越准」的

学习发生在三个地方，全部可在无网络环境离线运行：

1. **链路图（ChainGraph）** —— 每当模型先调 A、再调 B，就给有向边 `A→B` 加权。
   权重带**时间衰减**、按**成功/失败**打折、对**跨域边**轻微抑制（跨域是正常
   信号，但不该淹没域内链路）。支持 `dump / load`，可直接随仓库分发。
2. **触发词挖掘（Learner）** —— 从「这次查询最终落到了哪个业务域」反推：某些
   词反复出现在某个域的开头，它们就成为该域的学习触发词，让 L1 越来越准。
   只对**会话入口域**计入，避免把「订单→物流」的中途转手误学成「订单=物流」。
3. **参数流（arg_flow）** —— 记录「A 产出的哪个字段填进了 B 的参数」，让预填
   从「按名字猜」升级为「按真实数据流填」。

```python
from toolfunnel.learn import Learner, SessionRecord

report = Learner(funnel).learn(past_sessions)   # 批量回放历史会话
print(report.summary())
funnel.save_learning("learned.json")            # 只存学习到的状态（很小）
```

---

## 设计原则

- **零依赖内核。** 核心只用标准库；`openai`、`PyYAML`、`tiktoken` 都是可选
  加速项，不装也能跑。
- **默认不调模型。** 路由是确定性的、可复现的、毫秒级的。只有在配置了
  `StageSpec.llm_pick` 或启用语义重排时才会用到模型，而且**只对候选短名单**
  使用，成本上界可控。
- **绝不硬隔离。** 桥接工具 + `expand()` 保证模型永远出得去；猜错的代价是
  一次便宜调用，且会被学习纠正。
- **可解释。** `RouteResult.explanation` 逐层记录「为什么是这些工具」，方便
  调试一个夹在模型与真实工具之间的组件。
- **可序列化。** 工具目录（compose 文档）与学习状态（`save_learning`）都是
  纯 JSON，便于版本管理、A/B、跨进程复用。

## 什么时候不该用它

说清楚边界同样是设计的一部分：

- 工具数量少于 ~20，且都很短：直接全量发给模型更省事，漏斗收益有限。
- 业务域/阶段天然无法命名、边界高度流动：先把 L1 做成业务域无关的纯词法层
  再逐步引入域概念。
- 需要模型「自由探索」未知工具的开放式任务：本项目的假设是**工具集已知**，
  要解决的是「从中快速挑对」。

## 项目结构

```
toolfunnel/
  types.py       工具/参数/会话黑板/路由结果的数据模型 + token 估算
  registry.py    工具注册表 + 倒排索引 + BM25（支撑上千工具的检索）
  domains.py     L1 业务域网关
  stages.py      L2 阶段解析 + L4 参数预填（含预填接受率学习）
  chain.py       L3 链路图（学习、衰减、推荐、UCB 探索、蒸馏、序列化）
  funnel.py      四层编排 + 自适应预算 + 运行时学习 + 逃生通道 + 持久化
  learn.py       离线批量学习（链路重建、触发词挖掘）
  index.py       ★ 自动索引：原始工具 → 业务域/阶段/链路/触发词
  discovery.py   阶段自动发现（从调用序列聚类推断业务域的阶段）
  semantic.py    可选的 embedding 重排 + 语义域兜底
  mcp.py         MCP 适配器（工具目录灌入 + 执行回传 + 学习闭环）
  cache.py       冷启动缓存（索引 + 学习状态落盘）
  adapters.py    OpenAI / Anthropic / 通用 ToolRunner 适配
  compose.py     声明式目录（JSON/YAML/DSL）加载与导出
  cli.py         命令行入口（index/demo/catalog/stats/route/bench/learn/discover-stages/distill/mcp-import）
  __main__.py    `python -m toolfunnel` 入口
  data/          随包分发的示例目录（wheel 与 zipapp 都能读到）
tools/
  build_zipapp.py  构建单文件 tool-funnel.pyz
examples/        1000 工具合成目录生成器 + 端到端演示 + MCP 演示 + 自动索引演示
benchmarks/      Token 节省、链路预测与规模扫描基准
tests/           136 个单元测试
install.sh       一键安装 / 构建 .pyz
INSTALL.md       下载与安装指南
```

## 测试

```bash
python -m pytest -q             # 136 passed
python examples/demo.py         # 端到端演示（无需 API key）
python examples/mcp_demo.py     # MCP 端到端演示（内置假 session，无需 server）
python benchmarks/bench.py      # 复现基准数字
python benchmarks/sweep.py      # 扫不同工具规模的收益曲线
```

## 参与贡献

欢迎 issue / PR。比较有价值的方向：新的阶段解析策略、更强的链路学习
（如带探索的 bandit）、更多适配器（LangChain / LlamaIndex / MCP）、以及在
真实业务目录上的评测报告。

## License

MIT © tool-funnel contributors

---

## English

**tool-funnel** is a funnel-shaped **tool router for LLM agents that own many
tools**. Instead of shipping the entire tool catalog to the model on every turn,
it narrows the field in four cheap, deterministic layers:

1. **Domain gate** — keyword / curated-trigger matching picks the business
   domain(s).
2. **Stage resolver** — within a domain, resolve the workflow phase, exposing
   only that phase's tools.
3. **Chain + lexical rank** — a graph learned across sessions both ranks the
   current set and recommends the *next* tool after each call.
4. **Argument pre-filler** — fills arguments from a session blackboard, removing
   round-trips.

Domains are **not** hard-isolated: bridge tools stay reachable and
`funnel.expand(name)` is an explicit escape hatch whose use is itself learned.

Results at 1000-tool scale: **~99% fewer tool-schema tokens per turn**, sub-ms
pure-CPU routing, and measurable improvement in next-tool prediction on
transitions that were never declared in config — only learned from traffic.

**Get it:** download the single-file `tool-funnel.pyz` from Releases and run
`python tool-funnel.pyz demo` — no pip, no install, no network — or
`pip install` the wheel. See [INSTALL.md](INSTALL.md).

```python
from toolfunnel import Funnel, ToolSpec, DomainSpec

funnel = Funnel.from_tools(tools, domains=[DomainSpec("order", triggers=("order",))])
result = funnel.route("where is order 12345?")
schemas = result.schemas()          # hand these to your chat completion call
```

**MCP (v0.2).** Plug any MCP server's tool catalog straight into the funnel and
let calls feed the learning loop — no `mcp` dependency, works with the official
client's async session or a sync/embedded one:

```python
from toolfunnel.mcp import MCPToolServer, MCPToolRunner, ingest_mcp_servers

await ingest_mcp_servers(funnel, [MCPToolServer("ecom", session=session)])
runner = MCPToolRunner(funnel, [server])
payload = runner.tools_payload("find order 12345")   # only a few schemas
await runner.acall("order.search", {"keyword": "12345"})
# structuredContent scalars land on the blackboard, so the next call is
# pre-filled instead of round-tripping to the model for an id it just produced.
```

**Auto-indexing (v0.3).** Hand it a raw pile of tools — an MCP listing, an
OpenAPI document, or just a list of names — and it infers the domains, stages,
chains and triggers and returns a router that already works, with every guess
written into an ordinary, editable compose document:

```bash
tool-funnel index my_tools.json --out indexed.json
tool-funnel route indexed.json "find order 12345"
```

```python
from toolfunnel.index import index_tools
funnel, report = index_tools(raw_tools)      # no config to hand-write
print(report.summary())
```

Unclustered tools land in a catch-all `general` domain, so free-form / unbounded
tasks always have somewhere to go. An optional `llm_namer` hook lets a model
name the clusters — it works fully offline without one.

Also in v0.2: adaptive tool budget (ceilings, not targets), UCB exploration and
one-call distillation for the chain graph, on-disk cold-start cache, workflow
**stage auto-discovery** from call sequences, an embedding fallback for the
domain gate, and pre-fill acceptance learning.

Runtime dependencies: **none** (Python 3.8+). Optional extras: `openai`,
`PyYAML`, `tiktoken`. See `examples/` and `benchmarks/` for runnable demos.

MIT licensed.
