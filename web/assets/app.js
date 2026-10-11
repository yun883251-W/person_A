/* 页面只保存当前会话 ID；问答正文由 API 写入本地 SQLite。 */
(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const ui = Object.fromEntries([
    "sidebar", "sidebar-backdrop", "menu-toggle", "sidebar-close", "new-session",
    "session-search", "session-list", "load-more-sessions", "refresh-sessions",
    "service-status", "current-title", "session-subtitle", "refresh-history",
    "clear-session", "delete-session", "welcome", "conversation", "conversation-scroll",
    "composer-form", "question", "skill-select", "send-button", "composer-hint", "char-count",
    "notice", "notice-text", "notice-dismiss", "confirm-dialog", "confirm-title",
    "confirm-description", "confirm-cancel", "confirm-accept",
  ].map((id) => [id, byId(id)]));
  const PAGE_SIZE = 100;
  const STORAGE_KEY = "person-a.current-session";
  const state = {
    sessions: [], sessionOffset: 0, hasMore: false, current: null, turns: [],
    operation: null, pendingQuestion: "", needsRefresh: false,
  };
  const statusLabels = {
    success: "基于知识库回答", insufficient_evidence: "资料不足",
    needs_clarification: "需要补充信息", out_of_scope: "超出知识库范围",
    service_error: "服务异常", limit_reached: "达到调用上限",
  };
  const errorMessages = {
    session_not_found: "这个会话已不存在，请新建会话。",
    conflict: "会话正在处理其他操作，请稍后再试。",
    service_unavailable: "服务暂时繁忙，请稍后刷新。",
    model_timeout: "模型响应超时，请稍后查看历史确认结果。",
    service_configuration_error: "模型配置有误，请检查服务端的模型配置和 API 密钥。",
    index_not_built: "知识库索引尚未准备好，请先执行 person-a build 并重启服务。",
    unknown_skill: "所选回答方式已不可用，请重新选择。",
    service_error: "模型服务暂时不可用，请检查 Ollama 和对话模型连接。",
    storage_error: "会话存储失败，请检查本地数据库是否可用。",
    persistence_failed: "回答已生成，但未能保存。请先复制需要保留的内容。",
    persistence_unavailable: "回答已生成，但数据库暂时无法写入。请先复制需要保留的内容。",
    validation_error: "请求格式不正确，请确认问题非空且不超过 10,000 个字符。",
    internal_error: "服务处理失败，请查看服务端日志后重试。",
    network_error: "无法连接服务，请确认 FastAPI 仍在运行。",
    request_timeout: "等待服务响应超时。",
    invalid_response: "服务返回了无法识别的数据，请检查 API 服务。",
  };

  class ApiError extends Error {
    constructor(code, status = 0, body = {}) {
      super(errorMessages[code] || `请求未完成（HTTP ${status || "未知"}），请稍后重试。`);
      this.code = code;
      this.status = status;
      this.body = body;
    }
  }

  async function request(path, { method = "GET", body, timeout = 20000 } = {}) {
    const controller = new AbortController();
    const timer = timeout ? setTimeout(() => controller.abort(), timeout) : null;
    try {
      const response = await fetch(path, {
        method, signal: controller.signal, cache: "no-store",
        headers: body === undefined ? {} : { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
      if (response.status === 204) return null;
      let data;
      try { data = await response.json(); }
      catch { throw new ApiError("invalid_response", response.status); }
      if (!response.ok) throw new ApiError(data.code || "http_error", response.status, data);
      return data;
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw new ApiError(error.name === "AbortError" ? "request_timeout" : "network_error");
    } finally {
      if (timer !== null) clearTimeout(timer);
    }
  }

  function rememberSession(id) {
    try {
      if (id) localStorage.setItem(STORAGE_KEY, id);
      else localStorage.removeItem(STORAGE_KEY);
    } catch { /* 禁用浏览器存储时仍可正常聊天。 */ }
  }

  function notice(message, kind = "error") {
    ui.notice.hidden = !message;
    ui.notice.dataset.kind = kind;
    ui["notice-text"].textContent = message;
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // 模型输出始终作为文本构建 DOM，不将 Markdown/HTML 交给 innerHTML 执行。
  function inline(parent, text, depth = 0) {
    if (depth > 4) { parent.append(document.createTextNode(text)); return; }
    const pattern = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/g;
    let offset = 0;
    for (const match of text.matchAll(pattern)) {
      parent.append(document.createTextNode(text.slice(offset, match.index)));
      const token = match[0];
      if (token.startsWith("`")) parent.append(element("code", "inline-code", token.slice(1, -1)));
      else if (token.startsWith("**")) {
        const strong = element("strong");
        inline(strong, token.slice(2, -2), depth + 1);
        parent.append(strong);
      } else {
        const parts = /^\[([^\]]+)\]\((.+)\)$/.exec(token);
        const link = element("a", null, parts[1]);
        link.href = parts[2]; link.target = "_blank"; link.rel = "noopener noreferrer";
        parent.append(link);
      }
      offset = match.index + token.length;
    }
    parent.append(document.createTextNode(text.slice(offset)));
  }

  async function copyText(text, button) {
    const previous = button.textContent;
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "已复制";
      setTimeout(() => { button.textContent = previous; }, 1800);
    } catch { notice("无法访问剪贴板，请选中文本后手动复制。", "info"); }
  }

  function markdown(text) {
    const root = element("div", "prose");
    const lines = String(text || "").replace(/\r\n?/g, "\n").split("\n");
    let paragraph = [];
    let list = null;
    const flush = () => {
      if (paragraph.length) {
        const p = element("p"); inline(p, paragraph.join("\n")); root.append(p); paragraph = [];
      }
      list = null;
    };
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const fence = /^\s*(`{3,}|~{3,})(.*)$/.exec(line);
      if (fence) {
        flush();
        const content = [];
        const closing = new RegExp(`^\\s*${fence[1][0]}{${fence[1].length},}\\s*$`);
        while (++i < lines.length && !closing.test(lines[i])) content.push(lines[i]);
        const block = element("div", "code-block");
        const header = element("div", "code-header");
        header.append(element("span", null, fence[2].trim() || "代码"));
        const copy = element("button", null, "复制代码"); copy.type = "button";
        copy.addEventListener("click", () => copyText(content.join("\n"), copy));
        header.append(copy);
        const pre = element("pre"); pre.append(element("code", null, content.join("\n")));
        block.append(header, pre); root.append(block); continue;
      }
      if (!line.trim()) { flush(); continue; }
      const heading = /^(#{1,6})\s+(.+)$/.exec(line);
      if (heading) {
        flush(); const h = element(`h${Math.min(heading[1].length + 1, 6)}`);
        inline(h, heading[2]); root.append(h); continue;
      }
      if (/^\s*([-*_])(?:\s*\1){2,}\s*$/.test(line)) { flush(); root.append(element("hr")); continue; }
      // 简单管道表格，适用于概念比较。内容仍逐格安全渲染。
      if (line.includes("|") && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}:?\s*\|(?:\s*:?-{3,}:?\s*\|?)+\s*$/.test(lines[i + 1])) {
        flush();
        const cells = (row) => row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
        const wrapper = element("div", "table-scroll");
        const table = element("table"); const head = element("thead"); const row = element("tr");
        for (const cell of cells(line)) { const th = element("th"); inline(th, cell); row.append(th); }
        head.append(row); table.append(head);
        const body = element("tbody"); i++;
        while (i + 1 < lines.length && lines[i + 1].includes("|") && lines[i + 1].trim()) {
          const tr = element("tr");
          for (const cell of cells(lines[++i])) { const td = element("td"); inline(td, cell); tr.append(td); }
          body.append(tr);
        }
        table.append(body); wrapper.append(table); root.append(wrapper); continue;
      }
      const item = /^\s*(?:([-*+])|\d+[.)])\s+(.+)$/.exec(line);
      if (item) {
        if (paragraph.length) flush();
        const type = item[1] ? "UL" : "OL";
        if (!list || list.tagName !== type) { list = element(type.toLowerCase()); root.append(list); }
        const li = element("li"); inline(li, item[2]); list.append(li); continue;
      }
      if (/^>\s?/.test(line)) {
        flush(); const quote = element("blockquote"); inline(quote, line.replace(/^>\s?/, "")); root.append(quote); continue;
      }
      list = null;
      paragraph.push(line);
    }
    flush(); return root;
  }

  function renderSources(sources) {
    const details = element("details", "sources");
    details.append(element("summary", null, `参考来源 · ${sources.length} 处`));
    // sources 是实际引用的片段；API 没有保留 S 编号，不能重新编造对应关系。
    for (const source of sources) {
      const card = element("article", "source-card");
      card.append(element("div", "source-path", source.source_path));
      if (source.heading_path?.length) card.append(element("div", "source-heading", source.heading_path.join(" › ")));
      card.append(element("div", "source-detail", `原文第 ${source.start_line}–${source.end_line} 行`));
      details.append(card);
    }
    return details;
  }

  function message(role, text, result = null) {
    const article = element("article", `message message--${role}`);
    article.append(element("div", "message-avatar", role === "user" ? "你" : "A"));
    const content = element("div", "message-content");
    content.append(element("div", "message-label", role === "user" ? "你" : "Person A"));
    const body = element("div", "message-body");
    if (role === "user") body.textContent = text;
    else body.append(markdown(text));
    content.append(body);
    if (result) {
      if (result.sources?.length) content.append(renderSources(result.sources));
      const meta = element("div", "message-meta");
      const badge = element("span", "status-badge", statusLabels[result.status] || "回答结果");
      badge.dataset.status = result.status;
      meta.append(badge);
      if (result.unsaved || ["service_error", "limit_reached"].includes(result.status)) {
        meta.append(element("span", "unsaved-label", "本轮未保存"));
      }
      const copy = element("button", "copy-answer", "复制回答"); copy.type = "button";
      copy.addEventListener("click", () => copyText(text, copy)); meta.append(copy);
      content.append(meta);
    }
    article.append(content); return article;
  }

  function renderConversation() {
    const fragment = document.createDocumentFragment();
    for (const turn of state.turns) {
      fragment.append(message("user", turn.question), message("assistant", turn.answer, turn));
    }
    if (state.pendingQuestion) {
      fragment.append(message("user", state.pendingQuestion));
      if (state.operation === "send") {
        const pending = message("assistant", "正在检索文档、整理回答，请稍候…");
        pending.classList.add("loading-message");
        pending.setAttribute("role", "status");
        fragment.append(pending);
      }
    }
    ui.conversation.replaceChildren(fragment);
    ui.welcome.hidden = state.turns.length > 0 || Boolean(state.pendingQuestion);
    ui.conversation.hidden = !ui.welcome.hidden;
    requestAnimationFrame(() => { ui["conversation-scroll"].scrollTop = ui["conversation-scroll"].scrollHeight; });
  }

  function displayDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric" }).format(date);
  }

  function renderSessions() {
    const search = ui["session-search"].value.trim().toLocaleLowerCase();
    const sessions = state.sessions.filter((session) => (session.title || "未命名会话").toLocaleLowerCase().includes(search));
    const fragment = document.createDocumentFragment();
    for (const session of sessions) {
      const item = element("li", "session-item");
      const button = element("button"); button.type = "button";
      button.dataset.sessionId = session.session_id;
      button.title = session.title || "未命名会话";
      button.setAttribute("aria-current", session.session_id === state.current?.session_id ? "true" : "false");
      button.disabled = Boolean(state.operation);
      button.append(element("span", "session-item-title", session.title || "未命名会话"));
      button.append(element("span", "session-item-date", displayDate(session.updated_at)));
      button.addEventListener("click", () => selectSession(session.session_id));
      item.append(button); fragment.append(item);
    }
    if (!sessions.length) fragment.append(element("li", "empty-sessions", search ? "已加载会话中没有匹配项" : "从一个问题，开始你的学习记录。"));
    ui["session-list"].replaceChildren(fragment);
    ui["load-more-sessions"].hidden = !state.hasMore;
    ui["load-more-sessions"].disabled = Boolean(state.operation);
  }

  function updateControls() {
    const busy = Boolean(state.operation);
    for (const id of ["new-session", "refresh-sessions", "load-more-sessions"]) ui[id].disabled = busy;
    for (const id of ["refresh-history", "clear-session", "delete-session"]) ui[id].disabled = busy || !state.current;
    ui.question.disabled = busy;
    ui["skill-select"].disabled = busy;
    ui["send-button"].disabled = busy || state.needsRefresh || !ui.question.value.trim();
    ui["send-button"].setAttribute("aria-label", state.operation === "send" ? "正在生成回答" : "发送问题");
    ui["composer-form"].setAttribute("aria-busy", String(state.operation === "send"));
    ui["current-title"].textContent = state.current?.title || (state.current ? "未命名会话" : "新的学习会话");
    ui["session-subtitle"].textContent = state.current ? "历史保存在本机 · 可继续追问" : "Python / LangChain · 本地文档知识库";
    ui["composer-hint"].textContent = state.operation === "send"
      ? "正在等待完整回答。离开页面不会取消后台任务。"
      : state.needsRefresh ? "请先点击上方「刷新历史」确认本轮结果，再继续提问。"
      : state.operation ? "正在同步会话…" : "Enter 发送 · Shift + Enter 换行";
    ui["char-count"].textContent = `${ui.question.value.length.toLocaleString("zh-CN")} / 10,000`;
    for (const button of document.querySelectorAll("[data-prompt]")) button.disabled = busy;
    renderSessions();
  }

  function setOperation(operation) { state.operation = operation; updateControls(); }
  function fitInput() { ui.question.style.height = "auto"; ui.question.style.height = `${Math.min(ui.question.scrollHeight, 180)}px`; }
  function closeSidebar() { document.body.classList.remove("sidebar-open"); ui["menu-toggle"].setAttribute("aria-expanded", "false"); }

  async function loadSessions(append = false) {
    const offset = append ? state.sessionOffset : 0;
    const data = await request(`/sessions?limit=${PAGE_SIZE}&offset=${offset}`);
    const previous = append ? state.sessions : [];
    state.sessions = Array.from(new Map([...previous, ...data.items].map((session) => [session.session_id, session])).values());
    state.sessionOffset = offset + data.items.length;
    state.hasMore = data.items.length === PAGE_SIZE;
    renderSessions();
  }

  async function loadHistory(id) {
    const turns = [];
    // 历史按正序分页；必须读完各页，才能把最近一轮显示在底部。
    for (let offset = 0; ; offset += PAGE_SIZE) {
      const page = await request(`/sessions/${encodeURIComponent(id)}/messages?limit=${PAGE_SIZE}&offset=${offset}`);
      turns.push(...page.items);
      if (page.items.length < PAGE_SIZE) return turns;
    }
  }

  function resetConversation() {
    state.current = null; state.turns = []; state.pendingQuestion = ""; state.needsRefresh = false;
    rememberSession(null); ui.question.value = ""; fitInput();
    renderConversation(); updateControls();
  }

  async function selectSession(id) {
    if (state.operation) return;
    closeSidebar(); notice(""); setOperation("history");
    try {
      const record = await request(`/sessions/${encodeURIComponent(id)}`);
      const turns = await loadHistory(id);
      state.current = record; state.turns = turns; state.pendingQuestion = ""; state.needsRefresh = false;
      rememberSession(id); ui.question.value = ""; fitInput(); renderConversation();
    } catch (error) {
      if (error.status === 404 && state.current?.session_id === id) resetConversation();
      notice(error.message);
    } finally { setOperation(null); }
  }

  async function refreshHistory() {
    if (state.operation || !state.current) return;
    notice(""); setOperation("history");
    try {
      state.turns = await loadHistory(state.current.session_id);
      state.pendingQuestion = ""; state.needsRefresh = false;
      renderConversation();
      await loadSessions();
      notice("历史已同步。", "info");
    } catch (error) {
      if (error.status === 404) resetConversation();
      notice(error.message);
    } finally { setOperation(null); }
  }

  async function sendQuestion(event) {
    event.preventDefault();
    const question = ui.question.value.trim();
    if (!question || state.operation || state.needsRefresh) return;
    if (question.length > 10000) { notice(errorMessages.validation_error); return; }
    notice(""); closeSidebar(); setOperation("send");
    let submitted = false;
    try {
      if (!state.current) {
        state.current = await request("/sessions", { method: "POST", body: { title: question.slice(0, 80) }, timeout: 0 });
        rememberSession(state.current.session_id);
        state.sessions.unshift(state.current);
        updateControls();
      }
      state.pendingQuestion = question; ui.question.value = ""; fitInput(); renderConversation();
      submitted = true;
      // 不设置前端生成超时、不自动重发；后端负责模型超时，断连后人工刷新确认结果。
      const skillId = ui["skill-select"].value || null;
      const result = await request(`/sessions/${encodeURIComponent(state.current.session_id)}/messages`, {
        method: "POST", body: { question, skill_id: skillId }, timeout: 0,
      });
      state.turns.push(result); state.pendingQuestion = ""; renderConversation();
      try { await loadSessions(); }
      catch { notice("回答已返回，会话列表暂未刷新，可稍后手动刷新。", "info"); }
    } catch (error) {
      if (error.body?.result) {
        state.turns.push({ ...error.body.result, unsaved: error.body.persisted === false });
        state.pendingQuestion = "";
      }
      state.needsRefresh = submitted && ["network_error", "request_timeout", "invalid_response", "internal_error"].includes(error.code);
      if (!error.body?.result) ui.question.value = question;
      const recovery = state.needsRefresh ? " 后台可能仍在处理，请稍后刷新历史，避免重复提交。" : "";
      notice(error.message + recovery);
      if (error.status === 404 && state.current) {
        state.sessions = state.sessions.filter((item) => item.session_id !== state.current.session_id);
        resetConversation(); ui.question.value = question;
      }
    } finally {
      setOperation(null); renderConversation(); fitInput(); ui.question.focus();
    }
  }

  function confirmAction(kind) {
    const deleting = kind === "delete";
    ui["confirm-title"].textContent = deleting ? "删除这个会话？" : "清空会话历史？";
    ui["confirm-description"].textContent = deleting
      ? "会话及其中的全部问答将从本机永久删除，此操作无法撤销。"
      : "全部问答将从本机永久删除。会话会保留，可以重新开始提问。";
    ui["confirm-accept"].textContent = deleting ? "删除会话" : "清空历史";
    ui["confirm-dialog"].returnValue = "cancel";
    ui["confirm-dialog"].showModal();
    ui["confirm-cancel"].focus();
    return new Promise((resolve) => ui["confirm-dialog"].addEventListener("close", () => {
      resolve(ui["confirm-dialog"].returnValue === "confirm");
    }, { once: true }));
  }

  async function manageSession(kind) {
    if (state.operation || !state.current) return;
    const id = state.current.session_id;
    if (!await confirmAction(kind)) return;
    notice(""); setOperation(kind);
    try {
      await request(`/sessions/${encodeURIComponent(id)}${kind === "clear" ? "/messages" : ""}`, { method: "DELETE", timeout: 0 });
      if (kind === "delete") {
        state.sessions = state.sessions.filter((item) => item.session_id !== id);
        resetConversation();
      } else {
        state.turns = []; state.pendingQuestion = ""; state.needsRefresh = false; renderConversation();
      }
      notice(kind === "delete" ? "会话已删除。" : "历史已清空，可以开始新的提问。", "info");
      try { await loadSessions(); }
      catch { notice("操作已完成，会话列表暂未刷新。", "info"); }
    } catch (error) { notice(error.message); }
    finally { setOperation(null); }
  }

  async function checkHealth() {
    try {
      await request("/health", { timeout: 6000 });
      ui["service-status"].textContent = "API 已连接";
      ui["service-status"].dataset.state = "online";
      ui["service-status"].title = "HTTP 服务可访问；模型及知识库状态以问答结果为准。";
    } catch {
      ui["service-status"].textContent = "API 未连接";
      ui["service-status"].dataset.state = "offline";
      ui["service-status"].title = "请确认本机 FastAPI 服务正在运行。";
    }
  }

  async function refreshSessions(append = false) {
    if (state.operation) return;
    notice(""); setOperation("sessions");
    try { await loadSessions(append); }
    catch (error) { notice(error.message); }
    finally { setOperation(null); }
    void checkHealth();
  }

  ui["composer-form"].addEventListener("submit", sendQuestion);
  ui.question.addEventListener("input", () => { fitInput(); updateControls(); });
  ui.question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
      event.preventDefault(); ui["composer-form"].requestSubmit();
    }
  });
  ui["new-session"].addEventListener("click", () => {
    if (state.operation) return;
    resetConversation(); notice(""); closeSidebar(); ui.question.focus();
  });
  ui["session-search"].addEventListener("input", renderSessions);
  ui["load-more-sessions"].addEventListener("click", () => refreshSessions(true));
  ui["refresh-sessions"].addEventListener("click", () => refreshSessions());
  ui["refresh-history"].addEventListener("click", refreshHistory);
  ui["clear-session"].addEventListener("click", () => manageSession("clear"));
  ui["delete-session"].addEventListener("click", () => manageSession("delete"));
  ui["notice-dismiss"].addEventListener("click", () => notice(""));
  ui["confirm-cancel"].addEventListener("click", () => ui["confirm-dialog"].close("cancel"));
  ui["confirm-accept"].addEventListener("click", () => ui["confirm-dialog"].close("confirm"));
  ui["menu-toggle"].addEventListener("click", () => {
    document.body.classList.add("sidebar-open"); ui["menu-toggle"].setAttribute("aria-expanded", "true");
    ui["sidebar-close"].focus();
  });
  ui["sidebar-backdrop"].addEventListener("click", closeSidebar);
  ui["sidebar-close"].addEventListener("click", closeSidebar);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && document.body.classList.contains("sidebar-open")) {
      closeSidebar(); ui["menu-toggle"].focus();
    }
  });
  for (const button of document.querySelectorAll("[data-prompt]")) {
    button.addEventListener("click", () => {
      if (state.operation) return;
      ui.question.value = button.dataset.prompt; fitInput(); updateControls(); ui.question.focus();
    });
  }
  window.addEventListener("beforeunload", (event) => {
    if (state.operation === "send") { event.preventDefault(); event.returnValue = ""; }
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) void checkHealth(); });
  window.addEventListener("online", checkHealth);
  window.addEventListener("offline", checkHealth);

  async function initialize() {
    renderConversation(); setOperation("sessions"); void checkHealth();
    try {
      const [skills] = await Promise.all([request("/skills"), loadSessions()]);
      for (const skill of skills.items) {
        const option = element("option", null, skill.description);
        option.value = skill.skill_id;
        ui["skill-select"].append(option);
      }
    }
    catch (error) { notice(error.message); }
    finally { setOperation(null); }
    let id = null;
    try { id = localStorage.getItem(STORAGE_KEY); } catch { /* 可选功能 */ }
    if (id && /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(id)) await selectSession(id);
    fitInput();
  }
  void initialize();
})();
