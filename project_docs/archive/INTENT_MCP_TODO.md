# Intent 与 MCP 接入开发清单

本文保留最初的开发计划，已归档到 `project_docs/`。**下方“尚未实现”等表述指编写计划时的状态，不能用于判断当前项目进度。** 意图、MCP、MBTI 和自动 Skill 的当前实现说明分别见 [意图与任务规划](../architecture/INTENT_PLANNING.md)、[MBTI 接入](../features/MBTI_MCP.md) 和 [自动 Skill](../features/AUTO_SKILL_TODO.md)。代码拆分对应关系见 [目录迁移说明](../README.md#本次目录迁移)，本次整理未执行测试或联调。

计划编写时状态：仅创建模块说明、详细 TODO 和配置草案，尚未实现或接入运行中的 Agent。由你逐项开发，全部接入完成后再统一验证。

本次范围是意图识别节点、路由节点、MCP 客户端及现有工具注册器适配。保留当前知识问答与手动 Skill 选择。MBTI、`workflows/`、自动 Skill 选择和图片问答留到后续。

## 新文件与职责

| 文件 | 职责 |
| --- | --- |
| `src/knowledge_agent/agent/intent.py` | 识别知识问题、简单对话、澄清请求和 MCP 工具请求；不执行工具 |
| `src/knowledge_agent/agent/router.py` | 将意图映射到知识、MCP 或直接回复分支；记录 Agent 接入 TODO |
| `src/knowledge_agent/mcp/client.py` | 本地 stdio 服务的配置、连接、发现、调用、超时和关闭 |
| `src/knowledge_agent/mcp/adapter.py` | 将白名单内的 MCP 工具适配到现有注册器与参数校验 |
| `config/mcp_servers.example.json` | 未来读取的配置模板；演示服务默认禁用，目前没有配置读取实现 |
| `examples/mcp_demo_server.py` | 后续联调用的无副作用演示服务 TODO，不包含 MBTI 工具 |

这些 Python 新文件只有注释和模块说明，不能作为已实现接口调用。

项目已按功能分包：意图识别和路由与执行入口放在 `agent/`，MCP 客户端与适配器放在 `mcp/`。以下开发顺序中的 Python 路径均相对 `src/knowledge_agent/`；目录调整不改变 CLI/API 启动命令，也没有实现本清单中的业务接入。

## 接入后的目标流程

```text
用户输入 + 当前会话有限上下文
  -> intent：识别意图，不注册为工具
  -> router：复核可用能力，决定分支
       -> knowledge：现有检索、Skill、知识工具、引用校验
       -> mcp：独立提示、授权 MCP 工具、工具结果校验
       -> direct：问候、范围说明、澄清或范围外回复
  -> 统一 AnswerResult、trace、会话保存
```

MCP 服务在应用生命周期内启动并复用，单轮问答不重复启动进程。知识分支继续以本轮知识来源生成引用；外部工具结果不转换成知识库的 `[S1]` 来源。

## 推荐开发顺序

1. **先开发 `agent/intent.py` 的 TODO 01—04。** 定义返回结构，迁移现有基础规则，处理知识追问，接收可用工具的描述。此时可以先传入空工具集合。
2. **开发 `agent/router.py` 的 TODO 01—04。** 在首次知识检索之前接入识别与路由，先接回现有知识/直接回复分支。未接入的 MCP 能力保持不可用，不能生成成功结果。
3. **开发 `mcp/client.py` 的实现阶段 TODO。** 完成配置读取与可选依赖、服务生命周期、工具发现和调用；未启用 MCP 时知识问答仍可独立使用。
4. **开发 `mcp/adapter.py` 的实现阶段 TODO。** 扩展注册器参数校验接口，将已允许的 MCP 工具转换为可绑定工具定义。
5. **完成 `agent/router.py` 的 TODO 05—09 和 MCP 生命周期接入。** 让 CLI/API 的实际问答入口使用 MCP 工具循环，同步结果、会话、执行记录和页面必要字段。
6. **完成 `agent/intent.py` 的 TODO 05—09。** 在规则模式的基础上提供可选模型识别，复用同一轮预算；保持普通问答和手动 Skill 的现有含义。
7. **完成演示服务，然后执行各模块最后一阶段的统一验证。** 先使用假模型/假传输，再做真实 DeepSeek 与本地演示 MCP 服务联调；不要在前面逐项开发时运行测试。

## 现有文件需要改哪里

以下都是待开发事项。Python 文件路径相对 `src/knowledge_agent/`，其他路径相对项目根目录；结构整理仅迁移文件和同步引用，不代表已完成下列业务改动。

| 现有文件 | 后续改动 |
| --- | --- |
| `config.py`、`.env.example` | 增加 `INTENT_MODE=rules` 与 `MCP_CONFIG_PATH`（默认留空）；默认规则识别、MCP 关闭；实际 `.env` 由用户管理 |
| `agent/runner.py` | `ask()` 在初次检索前识别/路由；迁移分类兼容入口；复用单轮状态、计数、消息配对与提交；增加 MCP 分支 |
| `agent/prompts.py` | 增加意图分类和 MCP 工具专用提示；保留知识回答的来源协议 |
| `tools/registry.py` | 给 `ToolRegistration` 增加可选 `argument_validator`；执行前完整校验 MCP JSON Schema；保留现有预算和 ID 检查 |
| `schemas.py` | 在回答结果中增加可选的意图/路由摘要，旧调用者可不传 |
| `session/store.py` | 保存并恢复可选摘要，兼容已有 JSON 记录；恢复会话不能重放工具动作 |
| `observability/trace.py` | 增加 intent/route 的公开事件和摘要，延续工具计数口径与旧 trace 兼容处理 |
| `api.py` | 在现有运行时启动/关闭链路接入 MCP；同步回答与历史公开字段，保留单工作线程和 SQLite 约束 |
| `cli.py` | 使用同一应用运行时管理 MCP；正常结束、异常和 Ctrl+C 均清理连接与进程 |
| `web/assets/app.js` | 在现有执行详情展示意图、路由和 MCP 工具状态；不新增 MBTI/图片页面 |
| `pyproject.toml` | 实现时明确声明选定版本的 MCP SDK 与 JSON Schema 校验依赖，建议放可选依赖组 |
| `README.md` | 开发完成后说明依赖安装、配置启用、启动方式、示例提问与实际限制，区分已实现和已验证 |

## 两个需要统一的契约

- **工具权限：** 本地配置的 `allowed_tools` → 启动后发现的可用工具 → 本轮路由候选集合 → 实际绑定的 registry。执行端仍要校验，不因模型生成了名字就获得权限。
- **状态归属：** Agent 持有单轮状态和模型/工具预算，应用运行时持有 MCP 连接，现有 SQLite 保存问答结果。此阶段没有 MBTI 答题状态，也不新增数据库服务。

详细的输入输出、异常处理、兼容改动和验收场景都在对应 Python 文件的编号 TODO 中。
