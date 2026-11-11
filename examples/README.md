# 示例

三个可独立运行的示例，全程**不需要 API Key**，纯本地跑通。

## 1. 生成合成工具目录 —— `generate_catalog.py`

按 10 个业务域 × 阶段 × 动词 × 名词生成一个大规模、贴近真实结构的工具目录，
并织入跨域桥接与阶段链路先验。

```bash
python examples/generate_catalog.py              # 生成 examples/catalog.json（默认 1000 工具）
python examples/generate_catalog.py --tools 3000 # 指定规模
```

生成得到的是 `compose` 文档（JSON），可直接喂给 `build_from_dict`，也可作为你
自己目录的模板改写。

## 2. 端到端演示 —— `demo.py`

一次跑完整个故事：漏斗路由 → 链路推荐 + 参数预填 → 逃生通道 → 学习前后对比 →
Token 节省统计。

```bash
python examples/demo.py
python examples/demo.py --tools 3000
```

## 3. 手写小型目录 —— `compose.example.json`

一个 4 工具、3 业务域的最小目录，用来理解声明式写法。可直接用 CLI 试：

```bash
tool-funnel stats  examples/compose.example.json
tool-funnel route  examples/compose.example.json "帮我查一下订单 12345"
tool-funnel bench  examples/compose.example.json
```

## 4. MCP 端到端 —— `mcp_demo.py`

演示把一个 MCP server 的工具目录灌进漏斗、路由、调用回传、下一轮预填的完整
闭环。**内置假 session，无需真实 server 或 API key**：

```bash
python examples/mcp_demo.py
```

换成真实 server 只需把 `FakeMCPSession` 替换成你的
`mcp.client.session.ClientSession`（两者方法签名一致）。

## 看一个工具目录「长什么样」

```jsonc
{
  "domains": [{ "name": "order", "triggers": ["订单"], "bridges": ["logistics"] }],
  "stages":  [{ "name": "locate", "domain": "order", "triggers": ["查"], "order": 0 }],
  "tools": [{
    "name": "order.search", "domain": "order", "stage": "locate",
    "description": "按关键字搜索订单", "keywords": ["订单", "搜索"],
    "produces": ["order_id"],        // 产出 order_id，供后续工具预填
    "next_hint": ["order.get"],      // 冷启动链路先验
    "params": [{ "name": "keyword", "required": true }]
  }]
}
```

字段含义见 [`../docs/DESIGN.md`](../docs/DESIGN.md) 与 `toolfunnel/compose.py`。
