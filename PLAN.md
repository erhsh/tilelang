# TileLang Lower Trace 实施计划

> 基于 pass_diff (7174a26) 吸收 pass_trace (b3706c7) 的设计方案

**工作分支**：`main_lower-trace`
**日期**：2026-06-16

---

## 1. 命名约定

| 项目 | 值 |
|------|-----|
| 环境变量 | `TILELANG_LOWER_TRACE` / `TILELANG_LOWER_TRACE_DIR` |
| 模块路径 | `tilelang/tools/lower_trace/` |
| 工作分支 | `main_lower-trace`（不创建新分支） |
| 旧代码 | **完全删除** |
| 测试 | 最小集成测试 |

---

## 2. 文件结构

```
tilelang/tools/lower_trace/
├── __init__.py        # 公开 API：patch, uninstall, reset, lower_trace, LowerRecord, STATUS_*
├── core.py            # monkey-patch / AST pass 发现 / phase 管理 / 全局状态
├── diff.py            # terminal + HTML diff + 空白差异合并 + 内联高亮
└── html.py            # HTML 报告生成（嵌入式 CSS/JS，增量写入）

testing/python/debug/test_lower_trace.py  # 最小集成测试
tilelang/__init__.py                       # 集成 patch()
tilelang/env.py                            # 新增 EnvVar + get_lower_trace_mode()
docs/tutorials/debug_tools_for_tilelang.md # 更新文档
```

**删除文件：**
- `tilelang/utils/pass_diff.py`
- `tilelang/utils/pass_diff_hook.py`

---

## 3. 8 项特性吸收决策

| # | 特性 | 来源 | 实现方式 |
|---|------|------|----------|
| 1 | AST pass 预扫描 | pass_trace | `_discover_passes` + `_discover_passes_recursive` |
| 2 | 3 状态模型 | pass_trace | `completed` / `failed` / `skipped` |
| 3 | 崩溃安全 | pass_trace → 优化 | **增量 HTML 写入**（O(n) 总成本） |
| 4 | 线程安全 | pass_trace | `threading.RLock` |
| 5 | 双架构自适应 | pass_trace | PassPipeline.lower + phase fallback |
| 6 | `.tir` 落盘 | pass_trace | phase 子目录 + 前缀命名 |
| 7 | 空白差异合并 | pass_trace | `_merge_whitespace_diffs` |
| 8 | 增强 HTML 交互 | 融合方案 | 见下节 |

---

## 4. HTML UI 融合方案

### 保留（来自 pass_diff）：
- Catppuccin 主题 + dark/light 切换（localStorage 持久化）✓
- 原生 `<details>` 展开上下文（简洁、无 JS 依赖）
- Copy 按钮 ✓
- Word-level 内联高亮 ✓

### 吸收（来自 pass_trace）：
- 侧边栏（pass 导航、状态小圆点、+/- 统计）✓
- 侧边栏可折叠、可拖拽调整宽度 ✓
- Phase 标签页 ✓
- Summary bar 含 clickable filter badge ✓
- `j`/`k` 键盘导航 ✓
- `Shift+E` 全局展开 ✓
- **F7 手动对齐（Beyond Compare 风格）** ✓（已加入）
- 失败 pass 的 error-box + before IR ✓
- Skipped pass 的占位提示 ✓

---

## 5. 关键数据结构

```python
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

@dataclass
class LowerRecord:
    phase: str
    name: str
    index: int
    before_text: str
    after_text: str
    changed: bool
    add_lines: int = 0
    del_lines: int = 0
    status: str = STATUS_COMPLETED
    error_msg: str = ""
```

---

## 6. 全局状态

```python
_records: list[LowerRecord] = []
_section_cache: list[str] = []       # ★ 增量写入关键
_html_header_cache: str = ""
_original_pass_call: Callable | None = None
_original_pipeline_lower: object | None = None
_current_phase: str | None = None
_pass_index: int = 0
_records_offset: int = 0
_auto_flush: bool = False
_diff_dir: str | None = None
_lock = threading.RLock()
```

---

## 7. Env 配置

```python
# tilelang/env.py
TILELANG_LOWER_TRACE = EnvVar("TILELANG_LOWER_TRACE", "0")
TILELANG_LOWER_TRACE_DIR = EnvVar("TILELANG_LOWER_TRACE_DIR", "tmp/lower_trace_output")

def get_lower_trace_mode(self) -> str | None:
    value = str(self.TILELANG_LOWER_TRACE).lower().strip()
    if value in ("0", "false", "no", "off", ""):
        return None
    if value in ("1", "true", "yes", "on", "html"):
        return "html"
    if value in ("terminal", "both"):
        return value
    return "html"  # fallback
```

---

## 8. 公开 API

```python
# tilelang/tools/lower_trace/__init__.py
from .core import patch, uninstall, reset, LowerRecord
from .core import STATUS_COMPLETED, STATUS_FAILED, STATUS_SKIPPED

__all__ = [
    "patch", "uninstall", "reset", "lower_trace",
    "LowerRecord", "STATUS_COMPLETED", "STATUS_FAILED", "STATUS_SKIPPED",
]

def lower_trace(func_or_mod, passes, *, mode="terminal", context=3,
                html_path="lower_trace_report.html") -> list[dict]:
    """编程式 API — 向后兼容 pass_diff 接口签名"""
```

---

## 9. core.py 核心函数清单

| 函数 | 职责 |
|------|------|
| `patch()` | 双层 monkey-patch + 架构自适应 + atexit 注册 |
| `uninstall()` | 恢复原始函数，清空状态 |
| `reset()` | 清空 records（保留 patch） |
| `_traced_pass_call(self, mod)` | 拦截 Pass.__call__，捕获 before/after |
| `_traced_pipeline_lower(self, mod, target)` | 新架构入口，预注册 passes |
| `_wrap_phase(orig_func, phase_idx, total)` | 旧架构 phase 函数装饰器 |
| `_discover_passes(phase_func) -> list[str]` | AST 扫描 pass 名 |
| `_discover_passes_recursive(phase_func) -> list[str]` | 递归跟踪 helper 函数 |
| `_discover_phases(lower_func) -> list` | 旧架构字节码扫描 |
| `_save_raw_files(record)` | 落盘 .tir 文件 |
| `_get_pass_display_name(pass_obj) -> str` | 提取 pass 显示名 |
| `_ensure_diff_dir() -> str` | 懒初始化输出目录 |
| `_incremental_flush_html()` | 增量 HTML 写入（O(n) 总成本） |
| `_get_pass_name(p) -> str` | 从 pass_obj.info.name 提取名称 |

---

## 10. diff.py 函数清单

| 函数 | 职责 |
|------|------|
| `unified_diff(before, after, ..., color) -> str` | 终端 diff（可选 ANSI 颜色） |
| `print_diff(before, after, ...) -> bool` | 打印 diff |
| `_inline_diff(line_before, line_after)` | 字符级内联高亮 |
| `_merge_whitespace_diffs(opcodes, before, after)` | 空白差异合并 |
| `_make_diff_html(before, after, context) -> str` | GitHub 风格 side-by-side HTML |
| `_count_changes(diff_lines) -> (int, int)` | 统计 +/- 行数 |

---

## 11. html.py 核心逻辑

```python
_CSS = "..."            # ~400 行
_JS = "..."             # ~400 行（无 F7）
_HTML_TEMPLATE = "..."  # HTML 框架

def generate_html(records, output_path):
    """完整生成（首次或最终 flush 使用）"""

def render_pass_section(record) -> str:
    """渲染单个 pass 的 section HTML（增量使用）"""

def build_header(records) -> str:
    """构建动态统计 header（pass 总数、changed 数等）"""

def build_footer() -> str:
    """构建 JS 脚本尾部"""
```

### 增量写入策略：
```python
def _incremental_flush_html():
    """O(n) 总成本：只渲染新 section，拼接 header + cache + footer"""
    newly = len(_records) - len(_section_cache)
    if newly > 0:
        for rec in _records[-newly:]:
            _section_cache.append(render_pass_section(rec))
    full = build_header(_records) + "\n".join(_section_cache) + build_footer()
    write(full)
```

---

## 12. __init__.py 集成

```python
# tilelang/__init__.py 末尾
if not env.is_light_import():
    ...  # existing backend imports

    if env.get_lower_trace_mode() is not None:
        from .tools.lower_trace import patch as _lower_trace_patch
        _lower_trace_patch()
        del _lower_trace_patch
```

---

## 13. 测试计划（最小）

```python
# testing/python/debug/test_lower_trace.py (~200 行)

def test_env_modes():
    """env var 解析：off → None, 1 → html, terminal, both, case-insensitive"""

def test_env_default_off():
    """默认是 off"""

def test_lower_trace_api_single_pass():
    """单 pass 调用返回正确 list[dict]"""

def test_lower_trace_api_chain():
    """链式 passes 返回正确数量和命名"""

def test_patch_uninstall():
    """patch → uninstall 生命周期正确（无残留状态）"""

def test_html_output():
    """生成 HTML 文件，检查内容含 pass 名、CHANGED/NO-OP 等关键字"""

def test_crash_status():
    """patch 后触发抛异常的 pass，STATUS_FAILED 正确标记"""

def test_discover_passes():
    """AST 发现能在简单函数中提取 pass 名称"""
```

---

## 14. 实施顺序

| Step | 内容 | 依赖 |
|------|------|------|
| 1 | 创建模块骨架 + LowerRecord + STATUS 常量 | 无 |
| 2 | 实现 diff.py | Step 1 |
| 3 | 实现 html.py + 增量写入逻辑 | Step 2 |
| 4 | 实现 core.py（patch/traced_pass_call/AST 发现） | Step 3 |
| 5 | 接入 env.py | Step 4 |
| 6 | 接入 __init__.py | Step 5 |
| 7 | 实现 lower_trace() 公开 API | Step 4 |
| 8 | 删除旧文件 + 清理引用 | Step 6 |
| 9 | 跑 lint / typecheck / 测试 | Step 8 |
| 10 | 更新文档 | Step 9 |
| 11 | 手动验证（跑一个 example + 环境变量） | Step 9 |

---

## 15. 风险与缓解

| 风险 | 缓解 |
|------|------|
| AST 发现对 `tirx.transform` / `tilelang.transform` / `s_tir.transform` 匹配不全 | AST visitor 同时识别三种前缀；测试覆盖已知 pipeline |
| `str(mod)` vs `.script()` 选择不当 | 默认 `str(mod)`（更快、更稳定） |
| 首次 patch 时 backend 未 import 完 | __init__.py 中 backend 导入后再 patch |
| PassPipeline.lower 已被其他代码 monkey-patch | 检查 `_original_pipeline_lower is not None` 防重复 + warning |
| HTML 中 IR 文本过大 | 默认 context=3，仅展开 changes；全 IR 隐藏按需展开 |
| 并发编译 records 覆盖 | `reset()` 在 `_traced_pipeline_lower` 开始时调用 |

---

## 16. 产出清单

- [x] `tilelang/tools/lower_trace/__init__.py`
- [x] `tilelang/tools/lower_trace/core.py`
- [x] `tilelang/tools/lower_trace/diff.py`
- [x] `tilelang/tools/lower_trace/html.py`
- [x] `testing/python/debug/test_lower_trace.py`
- [x] `tilelang/env.py` 修改
- [x] `tilelang/__init__.py` 修改
- [x] `tilelang/utils/pass_diff.py` — 当前分支未包含，无需删除
- [x] `tilelang/utils/pass_diff_hook.py` — 当前分支未包含，无需删除
- [ ] `docs/tutorials/debug_tools_for_tilelang.md` 更新（TODO，低优先级）
- [x] lint + typecheck 通过
- [x] 端到端验证通过（CPU pipeline 49 pass 全追踪 + HTML 报告生成）
