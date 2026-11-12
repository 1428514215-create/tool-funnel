# 下载与安装

三种方式，任选一种。**运行时零依赖**——只需要 Python 3.8+。

---

## 方式 1：单文件（最简单，无需安装）

从 [Releases](https://github.com/your-org/tool-funnel/releases) 下载
`tool-funnel.pyz`，然后用任意 Python 3.8+ 解释器直接运行：

```bash
python tool-funnel.pyz demo                      # 自带示例，跑给你看
python tool-funnel.pyz catalog --tools 1000      # 生成 1000 工具的目录
python tool-funnel.pyz route catalog.json "帮我查一下订单"
python tool-funnel.pyz bench catalog.json        # token 节省基准
```

不需要 pip、不需要虚拟环境、不需要联网。这就是「下载下来就能用」。

> `.pyz` 是一个 zipapp：把整个包打成一个可执行 zip 文件。你也可以 `chmod +x
> tool-funnel.pyz` 后直接 `./tool-funnel.pyz demo`。

---

## 方式 2：pip 安装（推荐用于项目集成）

```bash
pip install tool_funnel-0.2.1-py3-none-any.whl    # 从 Releases 下载 wheel
# 或直接从源码
pip install git+https://github.com/your-org/tool-funnel.git
```

装好后有 `tool-funnel` 命令行，也能 `import toolfunnel`：

```bash
tool-funnel demo
python -c "import toolfunnel; print(toolfunnel.__version__)"
```

可选增强（都不装也能跑）：

```bash
pip install "tool-funnel[yaml]"      # 用 YAML 写目录
pip install "tool-funnel[openai]"    # embedding 语义重排 / 语义域兜底
pip install "tool-funnel[mcp]"       # 官方 MCP 客户端（适配器本身不需要它）
```

---

## 方式 3：克隆源码（想改代码或跑测试）

```bash
git clone https://github.com/your-org/tool-funnel.git
cd tool-funnel
./install.sh              # 自动选择：pip 装；失败则回退构建 .pyz
# 或
./install.sh --venv       # 装到 .venv
./install.sh --zipapp     # 只构建 dist/tool-funnel.pyz
./install.sh --user       # 当前用户安装
```

验证：

```bash
python -m pytest -q              # 测试
python examples/demo.py          # 端到端演示（无需 API Key）
python examples/mcp_demo.py      # MCP 演示（内置假 session，无需 server）
```

---

## 30 秒上手

不管用哪种方式装的，下一步都一样：

```bash
tool-funnel demo
```

它会用内置示例目录跑一次完整漏斗，打印「查订单」和「售后」两个查询命中了哪些
工具、省了多少 token。看完之后：

```bash
tool-funnel catalog --tools 1000 --out big.json   # 造一个大目录
tool-funnel route big.json "客户申请退款"           # 看它怎么收窄
tool-funnel bench big.json                        # 看省了多少
tool-funnel --help                                # 全部子命令
```

## 用你自己的工具目录

写一个 JSON（或 YAML）描述文件即可，无需写 Python：

```jsonc
{
  "domains": [{ "name": "order", "triggers": ["订单", "order"], "bridges": ["logistics"] }],
  "stages":  [{ "name": "locate", "domain": "order", "triggers": ["查"], "order": 0 }],
  "tools": [{
    "name": "order.search", "domain": "order", "stage": "locate",
    "description": "按关键字搜索订单", "keywords": ["订单"],
    "produces": ["order_id"], "next_hint": ["order.get"],
    "params": [{ "name": "keyword", "required": true }]
  }]
}
```

```bash
tool-funnel route my-catalog.json "查订单"
```

## 接 MCP server

```python
from toolfunnel import Funnel, DomainSpec
from toolfunnel.mcp import MCPToolServer, MCPToolRunner, ingest_mcp_servers

async def main():
    funnel = Funnel()
    server = MCPToolServer("ecom", session=your_mcp_session)
    await ingest_mcp_servers(funnel, [server])       # 灌入工具目录
    funnel.gate.add(DomainSpec("order", triggers=("订单",)))
    runner = MCPToolRunner(funnel, [server])
    payload = runner.tools_payload("帮我查订单")      # 只给模型少量工具
    await runner.acall("order.search", {"keyword": "12345"})
```

## 常见问题

**Q: 一定要 Python 3.8+ 吗？**
是。核心只用标准库，3.8 到 3.13 都可以。

**Q: 需要联网吗？**
不需要。核心与 zipapp 全程离线；只有你显式启用 embedding 重排时才会调用外部
服务。

**Q: pip 装不了（公司内网 / 无外网）？**
用方式 1 的 `.pyz`，或 `./install.sh --zipapp`。它绕开 pip 和 PyPI。

**Q: 怎么升级？**
重新下载新的 `.pyz` 覆盖，或 `pip install -U` 新 wheel。
