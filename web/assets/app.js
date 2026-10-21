/* 页面只保存当前会话 ID；问答正文由 API 写入本地 SQLite。 */
(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const ui = Object.fromEntries([
    "sidebar", "sidebar-backdrop", "menu-toggle", "sidebar-close", "new-session",
    "session-search", "session-list", "load-more-sessions", "refresh-sessions",
    "service-status", "current-title", "session-subtitle", "refresh-history",
    "clear-session", "delete-session", "welcome", "conversation", "conversation-scroll",
    "composer-form", "question", "response-mode", "skill-select", "send-button", "composer-hint", "char-count", "mbti-start",
    "notice", "notice-text", "notice-dismiss", "confirm-dialog", "confirm-title",
    "confirm-description", "confirm-cancel", "confirm-accept", "knowledge-open",
    "knowledge-dialog", "knowledge-close", "knowledge-message", "knowledge-refresh",
    "knowledge-status-cards", "knowledge-build", "knowledge-update",
    "knowledge-upload-form", "knowledge-file", "knowledge-document-count",
    "knowledge-document-list", "knowledge-preview-title", "knowledge-preview-meta",
    "knowledge-preview-content", "operation-status", "knowledge-version-note",
    "knowledge-preview-warning", "knowledge-preview-prev", "knowledge-preview-next",
    "knowledge-preview-page", "knowledge-preview-jump-form", "knowledge-preview-line",
    "knowledge-preview-jump-button", "knowledge-preview-reload",
    "session-summary", "session-summary-meta", "session-summary-content", "session-summary-refresh",
    "memory-open", "memory-dialog", "memory-text", "memory-message",
    "memory-reload", "memory-forget", "memory-close", "memory-save",
  ].map((id) => [id, byId(id)]));
  const PAGE_SIZE = 100;
  const PREVIEW_PAGE_SIZE = 200;
  const STORAGE_KEY = "person-a.current-session";
  const KNOWLEDGE_PENDING_KEY = "person-a.pending-knowledge-operation";
  const state = {
    sessions: [], sessionOffset: 0, hasMore: false, current: null, turns: [],
    operation: null, pendingQuestion: "", liveProgress: [], needsRefresh: false, assessmentModeHint: false,
    knowledgeBusy: false, knowledgeOperation: null, knowledgeNeedsRefresh: false,
    knowledgeDocuments: [], knowledgeDocumentsLoaded: false, knowledgeDocumentsStale: true,
    knowledgeStatus: null, knowledgeStatusStale: true,
    knowledgePreviewPath: null, knowledgePreview: null, knowledgePreviewRequest: null,
    knowledgePreviewReferenceHash: null, knowledgePreviewChanged: false,
    summaryRequestVersion: 0,
    memorySnapshot: null, memoryBusy: false, memorySessionId: null,
    businessState: null,
    readOperation: false, activeSend: null, answerRevision: 0,
    drafts: new Map(), uncertainSessions: new Set(), unsavedAnswers: new Map(),
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
    request_queue_full: "服务等待队列已满，本次请求未接收，请稍后手动重试。",
    request_queue_timeout: "请求排队超时，尚未执行，请稍后手动重试。",
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
    document_changed: "原文已变化或被移除，无法按原来的内容核对。",
    document_not_found: "文档已不存在，请刷新文档列表。",
    document_too_large: "文档超过原文预览的大小限制。",
    invalid_document_path: "文档路径无效，无法读取原文。",
    document_exists: "同名文档已存在，请改名后上传。",
    invalid_filename: "文件名无效，只能上传单个 Markdown 文件。",
    invalid_encoding: "文档必须使用 UTF-8 编码。",
    upload_too_large: "文档超过上传大小限制。",
    knowledge_busy: "知识库正在处理其他操作，请稍后刷新状态。",
    service_busy: "服务正在处理其他操作，请稍后刷新状态。",
    knowledge_storage_error: "无法读取或写入知识库目录，请检查目录及权限。",
    index_incomplete: "已发布索引不完整，请检查索引文件。",
    index_corrupt: "索引清单无法读取，请检查索引文件。",
    index_ingest_failed: "索引更新失败，请检查文档和模型服务。",
    index_refresh_failed: "索引已发布，但运行中的服务未能加载新版本。",
  };

  class ApiError extends Error {
    constructor(code, status = 0, body = {}) {
      super(errorMessages[code] || `请求未完成（HTTP ${status || "未知"}），请稍后重试。`);
      this.code = code;
      this.status = status;
      this.body = body;
    }
  }

  async function request(path, { method = "GET", body, rawBody, contentType, timeout = 20000 } = {}) {
    const controller = new AbortController();
    const timer = timeout ? setTimeout(() => controller.abort(), timeout) : null;
    try {
      const response = await fetch(path, {
        method, signal: controller.signal, cache: "no-store",
        headers: rawBody === undefined
          ? (body === undefined ? {} : { "Content-Type": "application/json" })
          : { "Content-Type": contentType || "application/octet-stream" },
        body: rawBody === undefined ? (body === undefined ? undefined : JSON.stringify(body)) : rawBody,
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

  async function streamAnswer(path, body, onProgress) {
    // 一次 POST 对应一轮问答。流中只有公开阶段和已校验的最终结果，不自动重连或重发。
    let reader;
    try {
      const response = await fetch(path, {
        method: "POST", cache: "no-store",
        headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        let data;
        try { data = await response.json(); }
        catch { throw new ApiError("invalid_response", response.status); }
        throw new ApiError(data.code || "http_error", response.status, data);
      }
      if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
        throw new ApiError("invalid_response", response.status);
      }
      reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
        if (buffer.length > 4_000_000) throw new ApiError("invalid_response");
        let delimiter;
        while ((delimiter = /\r?\n\r?\n/.exec(buffer))) {
          const frame = buffer.slice(0, delimiter.index);
          buffer = buffer.slice(delimiter.index + delimiter[0].length);
          let type = "message";
          const dataLines = [];
          for (const line of frame.split(/\r?\n/)) {
            if (line.startsWith("event:")) type = line.slice(6).trim();
            else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
          }
          if (!dataLines.length) continue; // SSE 注释心跳。
          let data;
          try { data = JSON.parse(dataLines.join("\n")); }
          catch { throw new ApiError("invalid_response"); }
          if (!data || typeof data !== "object" || Array.isArray(data)) throw new ApiError("invalid_response");
          if (type === "progress") onProgress(data);
          else if (type === "error") throw new ApiError(data.code || "internal_error", data.http_status || 0, data);
          else if (type === "result") {
            if (typeof data.answer !== "string" || typeof data.question !== "string" || typeof data.status !== "string") {
              throw new ApiError("invalid_response");
            }
            return data;
          }
        }
        if (done) throw new ApiError("network_error"); // 没有最终帧，不能将已有进度当作答案。
      }
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw new ApiError("network_error");
    } finally {
      if (reader) {
        try { await reader.cancel(); } catch { /* 保留原结果或错误。 */ }
        reader.releaseLock();
      }
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

  function knowledgeMessage(message, kind = "info") {
    ui["knowledge-message"].hidden = !message;
    ui["knowledge-message"].dataset.kind = kind;
    ui["knowledge-message"].textContent = message;
  }

  function isBusy() { return Boolean(state.operation || state.knowledgeBusy); }
  function chatControlsLocked() { return isBusy() || state.readOperation || state.knowledgeNeedsRefresh; }
  function readingLocked() {
    return state.readOperation || state.knowledgeBusy ||
      Boolean(state.operation && (state.operation !== "send" || !state.activeSend));
  }
  function setReadOperation(value) { state.readOperation = value; updateControls(); }
  function saveDraft() { state.drafts.set(state.current?.session_id || "", ui.question.value); }
  function pendingFor(id) { return state.activeSend?.sessionId === id && !state.activeSend.completed ? state.activeSend.question : ""; }

  function setKnowledgeOperation(operation) {
    state.knowledgeOperation = operation;
    state.knowledgeBusy = Boolean(operation);
    updateControls();
  }

  function setKnowledgeNeedsRefresh(value) {
    state.knowledgeNeedsRefresh = value;
    // 只记录是否存在未确认操作；刷新页面后仍需先查询状态，不能自动重发。
    try {
      if (value) sessionStorage.setItem(KNOWLEDGE_PENDING_KEY, "pending");
      else sessionStorage.removeItem(KNOWLEDGE_PENDING_KEY);
    } catch { /* 浏览器禁用存储时，本页仍保留恢复状态。 */ }
  }

  function knowledgeErrorMessage(error) {
    return error.code === "validation_error"
      ? "知识库请求参数无效，请检查文件名、文件内容或预览行号。"
      : error.message;
  }

  function mutationOutcomeUnknown(error) {
    return ["network_error", "request_timeout", "invalid_response", "internal_error"].includes(error.code)
      || error.status >= 500;
  }

  function invalidateKnowledgeData() {
    state.knowledgeStatusStale = true;
    state.knowledgeDocumentsStale = true;
    renderKnowledgeStatus(); renderKnowledgeDocuments();
  }

  function renderOperationStatus() {
    const labels = {
      build: "正在完整重建索引，暂不能提问。关闭知识库窗口不会取消操作。",
      update: "正在更新索引，暂不能提问。关闭知识库窗口不会取消操作。",
      upload: "正在上传文档，完成后可继续提问。",
      preview: "正在读取原文，请稍候。",
      refresh: "正在读取文档与索引状态，请稍候。",
    };
    const text = state.knowledgeOperation ? labels[state.knowledgeOperation]
      : state.knowledgeNeedsRefresh ? "知识库操作结果尚未确认。请打开「知识库管理」刷新状态，再继续提问或更新。"
      : state.operation === "send" ? "正在生成回答，完成后可操作知识库。"
      : state.needsRefresh ? "上一轮问答结果尚未确认。请先刷新会话历史，再上传资料或更新索引。" : "";
    ui["operation-status"].textContent = text;
    ui["operation-status"].hidden = !text;
    ui["operation-status"].dataset.kind = (state.knowledgeNeedsRefresh || state.needsRefresh) && !isBusy() ? "warning" : "info";
  }

  const knowledgeStateLabels = {
    pending: "待加入索引", synced: "已同步", modified: "原文已修改",
    removed: "原文已移除", unreadable: "无法读取", unknown: "状态未知",
  };
  const indexStateLabels = {
    not_built: "尚未建立", published: "已发布", incomplete: "发布不完整",
    corrupt: "索引损坏", unknown: "状态未知",
  };

  function renderKnowledgeStatus() {
    const status = state.knowledgeStatus;
    const pending = status?.pending_changes?.total;
    const cards = [
      ["发布状态", status ? (indexStateLabels[status.disk_status] || status.disk_status) : "—"],
      ["待同步变化", pending == null ? "—" : `${pending} 项`],
      ["文档 / 片段", status ? `${status.indexed_document_count ?? "—"} / ${status.chunk_count ?? "—"}` : "—"],
      ["已发布版本", status?.published_version || "未确认"],
      ["问答使用版本", status?.active_version || "未确认"],
    ];
    ui["knowledge-status-cards"].replaceChildren(...cards.map(([label, value]) => {
      const card = element("div", "knowledge-status-card");
      card.append(element("span", null, label), element("strong", null, value));
      return card;
    }));
    ui["knowledge-status-cards"].dataset.stale = String(state.knowledgeStatusStale);
    const notes = [];
    let kind = "info";
    if (state.knowledgeStatusStale) {
      notes.push(status ? "以下为上次读取的状态，当前状态尚未确认。" : "尚未取得索引状态。");
      kind = "warning";
    }
    if (status) {
      if (["incomplete", "corrupt", "unknown"].includes(status.disk_status) || status.error) {
        notes.push("索引状态异常，需检查知识库目录或索引文件。"); kind = "error";
      }
      if (status.published_version && status.active_version) {
        if (status.published_version === status.active_version) {
          notes.push("此次读取时，已发布版本与问答使用版本一致。");
        } else {
          notes.push("已发布版本与问答使用版本不同，当前问答尚未使用所示的已发布版本。");
          if (kind !== "error") kind = "warning";
        }
      } else if (status.published_version) {
        notes.push("问答使用版本未确认，可能尚未加载；不能据此认定已切换到新索引。");
        if (kind !== "error") kind = "warning";
      } else if (status.active_version) {
        notes.push("磁盘发布版本未确认，问答仍持有已加载的索引快照。");
        if (kind !== "error") kind = "warning";
      } else if (status.disk_status === "not_built") {
        notes.push("尚未建库，上传资料后需要建立索引。");
      }
    }
    ui["knowledge-version-note"].textContent = notes.join(" ");
    ui["knowledge-version-note"].hidden = !notes.length;
    ui["knowledge-version-note"].dataset.kind = kind;
  }

  function renderKnowledgeDocuments() {
    ui["knowledge-document-count"].textContent = state.knowledgeDocumentsLoaded
      ? `${state.knowledgeDocuments.length} 条记录${state.knowledgeDocumentsStale ? " · 上次读取" : ""}` : "尚未读取";
    const fragment = document.createDocumentFragment();
    for (const documentRecord of state.knowledgeDocuments) {
      const item = element("li");
      const button = element("button"); button.type = "button";
      button.disabled = isBusy() || !documentRecord.exists || documentRecord.sync_status === "unreadable";
      button.setAttribute("aria-current", String(documentRecord.path === state.knowledgePreviewPath));
      button.append(element("span", "knowledge-document-path", documentRecord.path));
      const detail = element("span", "knowledge-document-state", knowledgeStateLabels[documentRecord.sync_status] || documentRecord.sync_status);
      detail.dataset.state = documentRecord.sync_status;
      button.append(detail);
      button.addEventListener("click", () => openKnowledgeDocument(documentRecord.path, documentRecord.content_hash));
      item.append(button); fragment.append(item);
    }
    if (!state.knowledgeDocuments.length) fragment.append(element("li", "empty-sessions",
      state.knowledgeDocumentsLoaded ? "还没有 Markdown 文档。" : "刷新状态后查看文档。"));
    ui["knowledge-document-list"].replaceChildren(fragment);
  }

  function updateKnowledgeControls() {
    const busy = isBusy();
    ui["knowledge-refresh"].disabled = busy;
    for (const id of ["knowledge-build", "knowledge-update", "knowledge-file"]) {
      ui[id].disabled = busy || state.knowledgeNeedsRefresh || state.needsRefresh;
    }
    ui["knowledge-upload-form"].querySelector("button").disabled = busy || state.knowledgeNeedsRefresh || state.needsRefresh;
    // 知识库操作过程中仍能重新打开弹窗查看提示，但不能发起第二个请求。
    ui["knowledge-open"].disabled = Boolean(state.operation);
    ui["knowledge-dialog"].setAttribute("aria-busy", String(state.knowledgeBusy));
    const preview = state.knowledgePreview;
    ui["knowledge-preview-prev"].disabled = busy || !preview || preview.start_line <= 1;
    ui["knowledge-preview-next"].disabled = busy || !preview || preview.end_line >= preview.total_lines;
    const noLines = !preview || preview.total_lines === 0;
    ui["knowledge-preview-line"].disabled = busy || noLines;
    ui["knowledge-preview-jump-button"].disabled = busy || noLines;
    ui["knowledge-preview-reload"].disabled = busy || !state.knowledgePreviewRequest;
    renderKnowledgeDocuments();
  }

  async function loadKnowledgeData() {
    invalidateKnowledgeData();
    // 所有业务请求串行发送；成功响应才更新显示，不把旧数据当作当前状态。
    const status = await request("/knowledge/status");
    state.knowledgeStatus = status; state.knowledgeStatusStale = false;
    renderKnowledgeStatus();
    const documents = [];
    for (let offset = 0; ; offset += PAGE_SIZE) {
      const page = await request(`/knowledge/documents?limit=${PAGE_SIZE}&offset=${offset}`);
      documents.push(...page.items);
      if (page.items.length < PAGE_SIZE) break;
    }
    state.knowledgeDocuments = documents;
    state.knowledgeDocumentsLoaded = true; state.knowledgeDocumentsStale = false;
    setKnowledgeNeedsRefresh(false);
    renderKnowledgeDocuments(); renderKnowledgeStatus();
  }

  function knowledgeReadFailure(error) {
    const prefix = state.knowledgeStatusStale
      ? "当前状态未能确认。" : "索引状态已读取，但文档列表未能刷新。";
    return `${prefix} ${knowledgeErrorMessage(error)} 请稍后点击「刷新状态」。`;
  }

  async function openKnowledgeManager() {
    if (state.operation) { notice("当前会话操作尚未完成，请稍后打开知识库管理。", "info"); return false; }
    closeSidebar();
    if (!ui["knowledge-dialog"].open) ui["knowledge-dialog"].showModal();
    if (state.knowledgeBusy) return false;
    const recovering = state.knowledgeNeedsRefresh;
    setKnowledgeOperation("refresh"); knowledgeMessage("正在读取文档与索引状态…");
    try {
      await loadKnowledgeData();
      knowledgeMessage(state.needsRefresh
        ? "上一轮问答结果尚未确认，请关闭此窗口并刷新会话历史，再上传资料或更新索引。"
        : recovering ? "已读取当前状态。上次操作的最终结果未收到，请核对版本和文档后再决定是否重试。" : "");
      return true;
    } catch (error) { knowledgeMessage(knowledgeReadFailure(error), "error"); return false; }
    finally { setKnowledgeOperation(null); }
  }

  function previewWarning(message) {
    ui["knowledge-preview-warning"].textContent = message;
    ui["knowledge-preview-warning"].hidden = !message;
    ui["knowledge-preview-warning"].dataset.kind = "warning";
  }

  async function openKnowledgeDocument(path, contentHash = null, startLine = 1, options = {}) {
    if (isBusy()) return;
    if (!Number.isInteger(startLine) || startLine < 1) {
      previewWarning("请输入有效的正整数行号。"); return;
    }
    const continuing = options.continuing && path === state.knowledgePreviewPath;
    if (!continuing) {
      state.knowledgePreviewReferenceHash = options.referenceHash || null;
      state.knowledgePreviewChanged = false;
    }
    state.knowledgePreviewPath = path; state.knowledgePreview = null;
    state.knowledgePreviewRequest = { path, contentHash, startLine };
    setKnowledgeOperation("preview");
    ui["knowledge-preview-title"].textContent = path;
    ui["knowledge-preview-meta"].textContent = "正在读取原文…";
    ui["knowledge-preview-content"].textContent = "";
    ui["knowledge-preview-page"].textContent = "正在读取…";
    previewWarning(state.knowledgePreviewChanged ? "原文曾发生变化，正在读取当前内容。" : "");
    try {
      let preview;
      let requestedLine = startLine;
      let requestedHash = contentHash;
      for (let attempt = 0; attempt < 2; attempt++) {
        const params = new URLSearchParams({ path, start_line: String(requestedLine), limit: String(PREVIEW_PAGE_SIZE) });
        if (requestedHash) params.set("content_hash", requestedHash);
        try {
          preview = await request(`/knowledge/documents/content?${params}`);
          break;
        } catch (error) {
          if (attempt === 0 && error.code === "document_changed" && requestedHash) {
            // 始终持有同一把页面操作锁；变化后从第 1 行读取，防止新文件变短导致越界。
            state.knowledgePreviewChanged = true;
            requestedHash = null; requestedLine = 1;
            invalidateKnowledgeData();
          } else { throw error; }
        }
      }
      state.knowledgePreview = preview;
      state.knowledgePreviewRequest = { path, contentHash: preview.content_hash, startLine: Math.max(1, preview.start_line) };
      const empty = preview.total_lines === 0;
      ui["knowledge-preview-meta"].textContent = empty ? "这是一个空文档。"
        : `第 ${preview.start_line}–${preview.end_line} 行，共 ${preview.total_lines} 行 · 每页最多 ${PREVIEW_PAGE_SIZE} 行`;
      ui["knowledge-preview-page"].textContent = empty ? "无可预览内容"
        : `${preview.start_line}–${preview.end_line} / ${preview.total_lines} 行`;
      ui["knowledge-preview-content"].textContent = preview.content;
      ui["knowledge-preview-content"].scrollTop = 0;
      ui["knowledge-preview-line"].max = String(Math.max(1, preview.total_lines));
      ui["knowledge-preview-line"].value = String(Math.max(1, preview.start_line));
      if (state.knowledgePreviewReferenceHash && preview.content_hash !== state.knowledgePreviewReferenceHash) {
        previewWarning("当前原文与回答引用时不同，下面显示最新内容；旧引用的行号不能用于核对当前文本。");
      } else if (state.knowledgePreviewChanged) {
        previewWarning("原文在预览期间发生了变化，已重新读取最新内容；此前看到的分页内容属于旧版本。");
      } else { previewWarning(""); }
    } catch (error) {
      previewWarning(knowledgeErrorMessage(error));
      ui["knowledge-preview-meta"].textContent = "无法读取原文。";
      ui["knowledge-preview-page"].textContent = "可点击「重新读取」重试";
    } finally { setKnowledgeOperation(null); }
  }

  function changePreviewPage(direction) {
    if (isBusy() || !state.knowledgePreview) return;
    const preview = state.knowledgePreview;
    const line = direction > 0 ? preview.end_line + 1 : Math.max(1, preview.start_line - PREVIEW_PAGE_SIZE);
    if (preview.total_lines === 0 || line > preview.total_lines) return;
    void openKnowledgeDocument(preview.path, preview.content_hash, line, { continuing: true });
  }

  function jumpToPreviewLine(event) {
    event.preventDefault();
    if (isBusy() || !state.knowledgePreview) return;
    const preview = state.knowledgePreview;
    const line = Number(ui["knowledge-preview-line"].value);
    if (!Number.isInteger(line) || line < 1 || line > preview.total_lines) {
      previewWarning(`请输入 1 到 ${preview.total_lines} 之间的整数行号。`); return;
    }
    void openKnowledgeDocument(preview.path, preview.content_hash, line, { continuing: true });
  }

  async function uploadKnowledgeDocument(event) {
    event.preventDefault();
    const file = ui["knowledge-file"].files[0];
    if (!file || isBusy() || state.knowledgeNeedsRefresh || state.needsRefresh) return;
    setKnowledgeNeedsRefresh(true);
    setKnowledgeOperation("upload"); invalidateKnowledgeData(); knowledgeMessage("正在上传文档…");
    let uploaded = false;
    try {
      await request(`/knowledge/documents?filename=${encodeURIComponent(file.name)}`, {
        method: "POST", rawBody: file, contentType: "text/markdown; charset=utf-8", timeout: 0,
      });
      uploaded = true; setKnowledgeNeedsRefresh(false);
      ui["knowledge-upload-form"].reset();
      await loadKnowledgeData();
      knowledgeMessage("上传成功，待更新索引。", "success");
    } catch (error) {
      if (uploaded) {
        knowledgeMessage(`上传已成功，尚未更新索引。${knowledgeReadFailure(error)} 无需重新上传。`, "warning");
      } else {
        setKnowledgeNeedsRefresh(mutationOutcomeUnknown(error));
        knowledgeMessage(`${knowledgeErrorMessage(error)}${state.knowledgeNeedsRefresh
          ? " 上传结果尚未确认，请刷新状态并核对文件是否已存在，不要重复上传。" : ""}`, "error");
      }
    } finally { setKnowledgeOperation(null); }
  }

  function describeIndexOperation(result) {
    const version = result.published_version ? `（${result.published_version}）` : "（版本未确认）";
    if (result.published !== true) {
      return { text: result.published === false ? "本次操作未能完成索引发布。" : "本次索引发布结果尚未确认。", kind: "error" };
    }
    const published = `索引已发布${version}。`;
    if (result.refreshed === false || result.error_category === "refresh") {
      return { text: `${published}运行时刷新失败，当前问答可能仍使用旧索引。`, kind: "warning" };
    }
    if (result.published_version && result.active_version && result.published_version !== result.active_version) {
      return { text: `${published}问答使用版本与已发布版本不一致，请核对版本状态。`, kind: "warning" };
    }
    if (result.succeeded !== true) {
      return { text: `${published}操作未全部成功，请查看当前状态。`, kind: "warning" };
    }
    if (result.refreshed === true) {
      return { text: `${published}服务端已完成运行时刷新。`, kind: "success" };
    }
    return { text: `${published}本次未返回运行时刷新结果，问答使用版本以状态栏为准。`, kind: "info" };
  }

  async function runKnowledgeIndex(operation) {
    if (isBusy() || state.knowledgeNeedsRefresh || state.needsRefresh || !["build", "update"].includes(operation)) return;
    setKnowledgeNeedsRefresh(true);
    setKnowledgeOperation(operation); invalidateKnowledgeData();
    knowledgeMessage(operation === "build" ? "正在完整重建索引，请勿重复提交…" : "正在更新索引，请勿重复提交…");
    let result = null;
    let operationError = null;
    try {
      result = await request(`/knowledge/index/${operation}`, { method: "POST", timeout: 0 });
      if (!result || result.operation !== operation || typeof result.succeeded !== "boolean") {
        throw new ApiError("invalid_response", 200);
      }
      setKnowledgeNeedsRefresh(false);
    } catch (error) {
      operationError = error;
      // HTTP 失败也可能附带“已发布但刷新失败”的真实结果。
      const returnedOperation = error.body?.operation;
      result = returnedOperation?.operation === operation && typeof returnedOperation.succeeded === "boolean"
        ? returnedOperation : null;
      setKnowledgeNeedsRefresh(!result && mutationOutcomeUnknown(error));
    }
    try {
      let refreshError = null;
      try { await loadKnowledgeData(); }
      catch (error) { refreshError = error; }
      const summary = result ? describeIndexOperation(result)
        : { text: knowledgeErrorMessage(operationError), kind: "error" };
      if (refreshError) {
        knowledgeMessage(`${summary.text} ${knowledgeReadFailure(refreshError)}`, result ? "warning" : "error");
      } else {
        const tail = operationError
          ? " 已读取当前索引状态和文档列表，请核对版本后再决定是否重试；状态读取不代表上次操作成功。"
          : " 文档列表与索引状态已刷新。";
        knowledgeMessage(summary.text + tail, summary.kind);
      }
    } finally { setKnowledgeOperation(null); }
  }

  // 模型输出始终作为文本构建 DOM，不将 Markdown/HTML 交给 innerHTML 执行。
  function inline(parent, text, depth = 0) {
    if (depth > 4) { parent.append(document.createTextNode(text)); return; }
    const pattern = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\[(?:\\.|[^\]\\\n])+\]\(https?:\/\/[^\s)]+\))/g;
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
        const parts = /^\[((?:\\.|[^\]\\])+)\]\((.+)\)$/.exec(token);
        const link = element("a", null, parts[1].replace(/\\([\\`*_{}\[\]()#!|>~<])/g, "$1"));
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
      const card = element("button", "source-card"); card.type = "button";
      card.title = "打开引用原文";
      card.disabled = isBusy();
      card.append(element("div", "source-path", source.source_path));
      if (source.heading_path?.length) card.append(element("div", "source-heading", source.heading_path.join(" › ")));
      card.append(element("div", "source-detail", `原文第 ${source.start_line}–${source.end_line} 行`));
      card.addEventListener("click", async () => {
        if (isBusy()) return;
        const opened = await openKnowledgeManager();
        if (opened) await openKnowledgeDocument(source.source_path, source.content_hash, source.start_line, {
          referenceHash: source.content_hash,
        });
      });
      details.append(card);
    }
    return details;
  }

  const traceStageLabels = {
    memory: "组装会话记忆",
    profile: "加载学习画像",
    classify: "理解问题", intent: "识别意图", route: "选择执行路线", initial_retrieval: "初始检索", skill_prepare: "准备 Skill",
    model: "模型调用", tool: "工具调用", finalize: "整理回答与核验引用",
    planning: "任务规划", plan_step: "执行计划步骤",
    skill_select: "选择 Skill", skill_retrieval: "执行技能检索策略",
  };
  const traceStatusLabels = { running: "进行中", success: "完成", empty: "无结果", error: "失败", skipped: "已跳过" };
  const traceOutcomeLabels = { completed: "流程完成", failed: "执行失败", interrupted: "执行中断" };
  const traceDecisionSourceLabels = { rules: "规则", model: "模型", fallback: "回退", manual: "手动指定", disabled: "未启用" };
  const skillModeLabels = { off: "普通问答", auto: "自动选择", manual: "手动指定" };
  const strategyLabels = { default: "普通检索", concept: "概念讲解检索", comparison: "比较对象检索", code: "代码 API 检索" };
  const traceIntentKindLabels = {
    chat: "普通聊天",
    knowledge: "知识问答", greeting: "问候", scope: "能力范围", needs_clarification: "需要澄清",
    out_of_scope: "超出范围", mcp_tool: "MCP 工具请求",
  };
  const traceRouteKindLabels = { chat: "普通聊天（未检索知识库）", knowledge: "知识问答", mcp: "MCP 工具", direct: "直接回复" };
  const traceSkillLabels = {
    "concept-explanation": "概念讲解", "concept-comparison": "概念比较", "code-explanation": "代码解读",
  };
  const traceReasonLabels = {
    chat_completed: "聊天完成（未检索知识库）", chat_manual: "手动选择普通聊天",
    chat_rule_match: "规则识别为聊天", chat_semantic_match: "模型识别为聊天",
    chat_memory_statement: "用户提供会话信息，直接进入聊天",
    chat_memory_recall: "回忆用户已提供的信息，使用会话上下文",
    chat_classifier_fallback: "聊天分类不可用，保守进入知识流程",
    chat_budget_exhausted: "聊天预算不足", chat_invalid_response: "聊天响应无效",
    knowledge_manual: "手动选择知识库",
    memory_ready: "记忆已按上下文预算准备",
    memory_trimmed: "上下文超限，已裁剪可选历史或摘要",
    profile_injected: "已注入相关学习偏好",
    profile_disabled: "已关闭学习画像",
    profile_empty: "没有相关画像信息",
    profile_load_failed: "画像读取失败，继续普通回答",
    profile_budget_skipped: "上下文空间不足，已省略画像",
    video_results_read: "已读取视频标题和链接",
    browser_result_limit_invalid: "视频数量应为1至10个",
    browser_dependency_missing: "未安装浏览器读取依赖",
    browser_search_blocked: "B站搜索页拒绝访问",
    browser_results_unavailable: "未读取到有效视频列表",
    browser_search_timeout: "读取搜索页超时",
    browser_reader_failed: "浏览器读取失败",
    answer_completed: "回答完成", greeting_handled: "已回复问候", scope_explained: "已说明助手范围",
    question_is_ambiguous: "问题需要澄清", question_out_of_scope: "问题超出当前范围",
    index_not_built: "索引尚未建立", initial_retrieval_failed: "初始检索失败",
    initial_retrieval_empty: "初始检索没有结果", evidence_context_limit: "资料超出上下文限制",
    initial_retrieval_irrelevant: "检索候选未达到相关性要求",
    retrieval_relevant: "候选通过相关性筛选", retrieval_below_relevance: "候选相关性不足",
    retrieval_no_candidates: "索引没有可用候选",
    dependency_evidence_reused: "复用前置步骤已引用的原始资料，无需新检索",
    dependency_evidence_unavailable: "前置步骤缺少可复用的已引用原文",
    dependency_evidence_after_empty_retrieval: "新检索为空，继续使用前置原始资料",
    plan_dependency_transform: "根据明确前置步骤整理已有内容",
    model_call_budget_exhausted: "达到模型调用上限", tool_budget_exhausted: "达到工具调用上限",
    followup_search_budget_exhausted: "达到补充检索上限", model_empty_response: "模型未返回有效内容",
    model_did_not_finish: "模型未生成最终回答", model_requested_clarification: "模型请求澄清",
    model_reported_insufficient_evidence: "模型报告资料不足", partial_answer: "资料只能支持部分回答",
    model_reported_out_of_scope: "模型明确说明问题超出服务范围",
    answer_contract_invalid: "回答状态格式不符合约定", answer_status_conflict: "回答状态与正文不一致",
    refusal_format_nonconforming: "模型声明资料不足，回复格式异常；已用程序说明替代正文",
    answer_status_body_mismatch: "模型声明与正文不一致，已按实际拒答类型处理",
    model_authentication_failed: "模型鉴权失败", model_timeout: "模型调用超时",
    model_not_found: "模型不存在", model_request_invalid: "模型请求无效",
    model_output_invalid: "模型返回的内容或工具参数格式无效",
    model_service_unavailable: "模型服务不可用", model_invocation_failed: "模型调用失败",
    missing_citations: "回答缺少可核验引用", unknown_citation: "回答含有未知来源标记",
    unverified_citation_metadata: "回答中的来源信息无法核验", invalid_agent_parameters: "执行参数无效",
    invalid_tool_call: "工具调用格式无效", invalid_tool_call_id: "工具调用标识无效",
    duplicate_tool_call_id: "工具调用标识重复", unknown_tool: "请求的工具不可用",
    invalid_tool_arguments: "工具参数无效", tool_execution_failed: "工具执行失败",
    index_version_changed: "本轮索引版本发生变化", section_read_failed: "章节读取失败",
    skill_prepare_failed: "Skill 准备失败", answer_finalization_failed: "回答整理失败",
    trace_recording_failed: "执行记录未能完整生成", unfinished_event: "阶段结束状态未完整记录",
    turn_failed: "本轮执行失败", turn_interrupted: "本轮执行中断",
    greeting_input: "识别为问候", scope_input: "询问助手范围", vague_input: "问题缺少必要信息",
    out_of_scope_marker: "问题超出当前范围", default_knowledge: "按知识问题处理",
    explicit_mcp_tool_request: "明确请求 MCP 工具", mcp_tools_unavailable: "没有可用的 MCP 工具",
    mcp_tool_not_matched: "未匹配到唯一的 MCP 工具", unsupported_stateful_followup: "没有可继续的任务状态",
    explicit_topic_switch: "明确切换话题", followup_without_context: "追问缺少上下文",
    dependency_retrieval_required: "前置资料不足以完成此步骤，继续检索",
    dependency_evidence_reuse: "复用前置证据，无需新检索",
    dependency_retrieval_budget_fallback: "判断预算不足，保守检索",
    dependency_retrieval_timeout_fallback: "资料需求判断超时，保守检索",
    dependency_retrieval_invalid_fallback: "资料需求判断格式无效，保守检索",
    dependency_retrieval_error_fallback: "资料需求判断不可用，保守检索",
    knowledge_followup_semantic: "模型补全知识追问", followup_semantic_ambiguous: "追问指代需要澄清",
    followup_semantic_budget_exhausted: "追问补充预算不足，使用原检索问题",
    followup_semantic_timeout: "追问改写超时，使用原检索问题",
    followup_semantic_invalid: "追问改写未通过校验，使用原检索问题",
    followup_semantic_error: "追问改写不可用，使用原检索问题",
    followup_semantic_model: "知识追问语义补充", semantic_plan_candidate: "待确认的多任务请求",
    plan_semantic_assisted: "模型辅助确认任务或主题", plan_semantic_model: "规划语义补充",
    plan_semantic_timeout: "规划语义补充超时", plan_semantic_invalid: "规划语义提议未通过校验",
    plan_semantic_unavailable: "规划语义模型不可用",
    knowledge_followup: "知识问题追问", hybrid_mcp_tool: "模型识别为 MCP 工具请求",
    hybrid_needs_clarification: "模型判断需要澄清", hybrid_knowledge: "模型识别为知识问题",
    intent_model_empty_response: "意图模型未返回有效内容", intent_model_response_too_long: "意图模型响应过长",
    intent_model_invalid_json: "意图模型返回的 JSON 无效", intent_model_invalid_structure: "意图模型响应结构无效",
    intent_model_invalid_fields: "意图模型响应字段无效", intent_model_invalid_kind: "意图模型返回未知类别",
    intent_model_capability_mismatch: "意图模型选择了不可用的工具",
    intent_model_unconfirmed_tool_request: "工具请求未经用户明确确认", intent_model_ambiguous_tool: "工具选择存在歧义",
    intent_model_timeout: "意图模型调用超时", intent_model_error: "意图模型调用失败",
    intent_identification_failed: "意图识别失败", route_resolution_failed: "路由选择失败",
    manual_skill_mcp_conflict: "所选 Skill 与 MCP 工具请求冲突",
    mcp_capability_unavailable: "请求的 MCP 工具能力不可用",
    multi_step_plan: "按多步骤任务执行",
    plan_invalid_response: "规划模型未返回有效内容", plan_unexpected_tool_call: "规划模型返回了不允许的工具调用",
    plan_invalid_json: "规划模型返回的 JSON 无效", plan_invalid_structure: "规划模型返回的字段或结构不符合约定",
    plan_response_step_count: "模型计划的步骤数量与请求不一致", plan_invalid_step_id: "模型计划的步骤编号无效",
    plan_invalid_goal: "计划目标格式无效", plan_invalid_kind: "模型计划的步骤类别无效",
    plan_invalid_dependencies: "模型计划的步骤依赖无效", plan_ungrounded_step_goal: "模型改写了原始任务目标",
    plan_unauthorized_tool: "模型计划包含未授权工具", plan_invalid_step_tools: "模型计划的步骤类别与工具不匹配",
    plan_ungrounded_tool_goal: "模型计划的工具与原始任务权限不一致", plan_missing_requested_tool: "模型计划遗漏了请求的工具",
    plan_invalid_question: "缺少有效任务描述", plan_not_requested: "未识别到明确的多个任务",
    plan_invalid_request_goal: "原始任务目标为空、过长或包含非法字符",
    plan_invalid_numbering: "任务编号需从 1 开始且连续",
    plan_manual_skill_mcp_conflict: "手动知识 Skill 与计划中的工具操作冲突",
    plan_completed: "全部任务已完成", plan_partial: "任务部分完成",
    plan_partial_execution: "部分任务发生执行异常或达到限制",
    plan_partial_validation: "部分任务的回答未通过校验",
    plan_answer_validation_failed: "回答未通过校验",
    plan_explicit_clauses: "按当前明确子句构造计划，无模型规划调用",
    plan_invalid_dependency_reference: "引用了不存在、当前或后续的步骤",
    plan_ambiguous_dependency: "步骤中的指代不明确",
    plan_tool_dependency_unsupported: "工具参数不能从前置回答中推测",
    plan_knowledge_tool_dependency_unsupported: "工具结果暂不支持作为知识步骤的前置资料",
    plan_topic_reference_missing: "当前请求没有可补全的明确搜索主题",
    plan_topic_reference_ambiguous: "当前请求包含多个可能的搜索主题，请明确选择",
    plan_topic_reference_invalid: "搜索主题引用不符合当前请求的约束",
    plan_clause_intent_failed: "步骤意图识别服务失败",
    browser_bilibili_topic_reference: "从同一条请求的前置用户描述补全 B 站搜索主题",
    browser_search_topic_mismatch: "模型生成的搜索参数与已确定的主题不一致，未执行搜索",
    browser_bilibili_semantic_timeout: "浏览器意图识别超时",
    browser_bilibili_semantic_budget_exhausted: "浏览器意图识别预算不足",
    browser_bilibili_semantic_error: "浏览器意图识别服务失败",
    browser_bilibili_unavailable: "B 站浏览器工具当前不可用",
    plan_dependency_unfinished: "依赖的前置任务未完成", plan_tool_budget_exhausted: "该步骤的工具调用预算不足",
    plan_context_budget_exhausted: "该步骤的资料与背景超过上下文限制",
    plan_budget_exhausted: "任务执行预算不足", plan_out_of_scope: "请求的任务超出知识范围",
    plan_step_failed: "该步骤执行失败", plan_failed: "任务未能完成",
    plan_needs_clarification: "任务需要补充信息", plan_insufficient_evidence: "任务缺少可支持回答的资料",
    plan_invalid_step_count: "请求需包含 2～3 个不同任务", plan_stateful_assessment: "MBTI 问卷操作需要单独执行",
    plan_repeated_tool: "同一轮计划不能重复调用同一工具", plan_unavailable_tool: "请求的工具尚未接入",
    plan_clause_needs_clarification: "某一步的操作目标需要补充", plan_unconfirmed_clause_action: "某一步尚不能确认为可执行操作",
    plan_requested_clarification: "规划模型需要进一步明确任务目标", plan_context_too_long: "规划内容超出上下文限制",
    plan_model_budget_exhausted: "规划或执行的模型调用预算不足", plan_preparation_failed: "任务规划准备失败",
    plan_invalid_capabilities: "规划工具能力配置无效", plan_invalid_context: "规划历史上下文格式无效",
    plan_invalid_step_limit: "规划步骤上限配置无效", plan_invalid_context_limit: "规划上下文上限配置无效",
    plan_invalid_invoker: "规划模型调用器配置无效", plan_invalid_clause_resolver: "规划步骤意图识别器配置无效",
    plan_clause_intent_unavailable: "规划步骤的意图识别器不可用", plan_invalid_clause_intent: "规划步骤的意图识别结果无效",
    plan_unauthorized_clause_tool: "规划步骤的工具权限校验失败",
    skill_disabled: "本轮使用普通问答", skill_manual: "使用手动指定的技能",
    skill_rule_match: "规则匹配到技能", skill_model_match: "模型选择了技能",
    skill_no_match: "没有匹配的技能，使用普通问答",
    skill_catalog_invalid: "技能目录不可用，使用普通问答", skill_catalog_empty: "没有可用技能，使用普通问答",
    skill_selector_budget_exhausted: "为回答保留预算，使用普通问答",
    skill_selector_unavailable: "选择模型不可用，使用普通问答",
    skill_selector_timeout: "技能选择超时，使用普通问答",
    skill_selector_invalid_output: "选择结果无效，使用普通问答",
    skill_selector_error: "技能选择失败，使用普通问答",
    skill_load_failed: "技能加载失败，使用普通问答",
    skill_non_knowledge_route: "当前路线不使用知识技能",
  };

  function traceText(value, fallback = "未提供") {
    return typeof value === "string" && value.trim() ? value : fallback;
  }

  function traceLabel(labels, value, fallback = "未记录") {
    const text = traceText(value, "");
    return Object.prototype.hasOwnProperty.call(labels, text) ? labels[text] : (text || fallback);
  }

  function traceReason(value) {
    const code = traceText(value, "");
    if (!code) return "未提供";
    const label = traceLabel(traceReasonLabels, code);
    return label === code ? code : `${label}（${code}）`;
  }

  function traceNumber(value, suffix = "") {
    return Number.isInteger(value) && value >= 0 ? `${value.toLocaleString("zh-CN")}${suffix}` : "未记录";
  }

  function traceDuration(value) {
    if (!Number.isInteger(value) || value < 0) return "未记录";
    return value < 1000
      ? `${value.toLocaleString("zh-CN")} ms`
      : `${(value / 1000).toLocaleString("zh-CN", { maximumFractionDigits: 2 })} 秒`;
  }

  function traceStartedAt(value) {
    if (typeof value !== "string" || !value.trim()) return "未记录";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "未记录";
    return new Intl.DateTimeFormat("zh-CN", {
      year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
    }).format(date);
  }

  function traceSkill(value, fallback) {
    const id = traceText(value, "");
    if (!id) return fallback;
    const label = traceLabel(traceSkillLabels, id);
    return label === id ? id : `${label}（${id}）`;
  }

  function traceFields(entries, className = "trace-fields") {
    const list = element("dl", className);
    for (const [label, value] of entries) {
      const item = element("div", "trace-field");
      item.append(element("dt", null, label), element("dd", null, value));
      list.append(item);
    }
    return list;
  }

  const declaredAnswerLabels = {
    supported: "资料支持回答", partial: "资料仅支持部分回答",
    insufficient_evidence: "模型报告资料不足", needs_clarification: "模型请求澄清",
    out_of_scope: "模型报告超出范围",
  };
  const validationLabels = {
    passed: "通过", warning: "有提示（见校验说明）", rejected: "未通过，原正文未发布",
  };

  function answerDiagnosticFields(diagnostic) {
    if (!diagnostic || typeof diagnostic !== "object" || Array.isArray(diagnostic)) return [];
    const fields = [];
    if (traceText(diagnostic.declared_status, "")) {
      fields.push(["模型声明", traceLabel(declaredAnswerLabels, diagnostic.declared_status)]);
    }
    if (traceText(diagnostic.validation_status, "")) {
      fields.push(["正文校验", traceLabel(validationLabels, diagnostic.validation_status)]);
    }
    if (Number.isInteger(diagnostic.supplied_source_count)) {
      fields.push(["提供给模型的来源", traceNumber(diagnostic.supplied_source_count, " 个")]);
    }
    if (Number.isInteger(diagnostic.accepted_source_count)) {
      fields.push(["最终接受的引用", traceNumber(diagnostic.accepted_source_count, " 个")]);
    }
    if (traceText(diagnostic.validation_reason, "")) {
      fields.push(["校验说明", traceReason(diagnostic.validation_reason)]);
    }
    return fields;
  }

  function renderTraceEvent(event) {
    const item = element("li", "trace-event");
    if (Number.isInteger(event.sequence) && event.sequence >= 0) item.value = event.sequence + 1;
    const heading = element("div", "trace-event-heading");
    const title = event.stage === "tool" && event.initiator === "model"
      ? "模型追加工具调用" : traceLabel(traceStageLabels, event.stage, "未命名阶段");
    heading.append(element("strong", null, title));
    const status = element("span", "trace-status", traceLabel(traceStatusLabels, event.status, "状态未记录"));
    status.dataset.status = traceText(event.status, "unknown");
    heading.append(status); item.append(heading);
    const fields = [
      ["发起方", traceLabel({ program: "程序", model: "模型" }, event.initiator)],
      ["开始于本轮", traceDuration(event.started_offset_ms)],
      ["阶段耗时", traceDuration(event.duration_ms)],
    ];
    if (traceText(event.tool_name, "")) fields.push(["工具", event.tool_name]);
    if (traceText(event.step_id, "")) fields.push(["计划步骤", event.step_id]);
    if (traceText(event.skill_mode, "")) fields.push(["技能模式", traceLabel(skillModeLabels, event.skill_mode)]);
    if (traceText(event.selected_skill_id, "")) fields.push(["选中技能", traceSkill(event.selected_skill_id, "普通问答")]);
    if (traceText(event.strategy_id, "")) fields.push(["检索策略", traceLabel(strategyLabels, event.strategy_id)]);
    if (Number.isInteger(event.result_count) && event.result_count >= 0) {
      const label = event.stage === "profile" ? "注入字段数量" : ["finalize", "plan_step"].includes(event.stage) ? "最终接受的引用数量" : "返回结果数量";
      fields.push([label, traceNumber(event.result_count, " 个")]);
    }
    if (traceText(event.index_version, "")) fields.push(["阶段索引版本", event.index_version]);
    for (const [key, label, unit] of [
      ["history_turns", "注入历史", " 轮"],
      ["dropped_turns", "预算裁剪历史", " 轮"],
      ["summary_chars", "注入摘要（含包装）", " 字符"],
      ["memory_chars", "历史背景总量（含包装，不含画像）", " 字符"],
    ]) {
      if (Number.isInteger(event[key]) && event[key] >= 0) fields.push([label, traceNumber(event[key], unit)]);
    }
    if (traceText(event.tool_call_id, "")) fields.push(["工具调用 ID", event.tool_call_id]);
    if (traceText(event.decision_source, "")) {
      fields.push(["决策来源", traceLabel(traceDecisionSourceLabels, event.decision_source)]);
    }
    if (traceText(event.intent_kind, "")) fields.push(["意图类别", traceLabel(traceIntentKindLabels, event.intent_kind)]);
    if (traceText(event.route_kind, "")) fields.push(["执行路线", traceLabel(traceRouteKindLabels, event.route_kind)]);
    if (traceText(event.reason_code, "") && event.reason_code !== event.error_code) {
      fields.push(["决策原因", traceReason(event.reason_code)]);
    }
    if (traceText(event.error_code, "")) fields.push(["原因 / 错误码", traceReason(event.error_code)]);
    fields.push(...answerDiagnosticFields(event));
    item.append(traceFields(fields, "trace-fields trace-event-fields"));
    return item;
  }

  function renderExecutionDetails(result) {
    const details = element("details", "execution-details");
    const summary = element("summary", "execution-summary");
    summary.append(element("span", "execution-summary-title", "执行详情"));
    // 摘要只读取这轮 API 数据，不使用当前 Skill 下拉框或内部对话消息补全历史。
    const trace = result.trace && typeof result.trace === "object" && !Array.isArray(result.trace)
      && result.trace.schema_version === 1 ? result.trace : null;
    const overview = trace
      ? `${traceLabel(traceOutcomeLabels, trace.outcome)} · ${traceDuration(trace.duration_ms) === "未记录" ? "耗时未记录" : traceDuration(trace.duration_ms)}`
      : "调用统计";
    summary.append(element("span", "execution-summary-meta", overview));
    details.append(summary);
    const body = element("div", "execution-body");
    body.append(traceFields([
      ["模型调用", traceNumber(result.model_calls, " 次")],
      ["工具调用", traceNumber(result.tool_calls, " 次")],
      ["结束原因", traceReason(result.stop_reason)],
      ["回答引用的索引版本", traceText(result.index_version)],
      ...answerDiagnosticFields(result.answer_diagnostics),
    ]));
    body.append(element("p", "trace-note", "工具次数包含程序发起的初始检索；被拒绝的请求可能留有阶段记录，但不计为已执行工具。模型次数按 Agent 调用轮数统计。"));
    if (!trace) {
      const explanation = result.trace != null
        ? "此执行记录的格式暂不支持，以上为可用的调用统计。"
        : result.turn_id ? "历史记录未保存详细过程。" : "本轮暂未提供详细过程。";
      body.append(element("p", "trace-empty", explanation));
      details.append(body);
      return details;
    }
    body.append(traceFields([
      ["流程状态", traceLabel(traceOutcomeLabels, trace.outcome)],
      ["Agent 处理耗时", traceDuration(trace.duration_ms)],
      ["开始时间（本地）", traceStartedAt(trace.started_at)],
      ["执行记录 ID", traceText(trace.trace_id)],
    ]));
    body.append(element("p", "trace-note", "处理耗时截至 Agent 生成本轮结果，不包含服务排队、会话写入和 HTTP 传输。模型阶段包含内部重试的等待，但重试不单独增加模型调用次数。"));
    const routing = result.routing && typeof result.routing === "object" && !Array.isArray(result.routing)
      ? result.routing : null;
    if (routing) {
      body.append(element("h4", "trace-section-title", "意图与路由"));
      body.append(traceFields([
        ["意图类别", traceLabel(traceIntentKindLabels, routing.intent)],
        ["执行路线", traceLabel(traceRouteKindLabels, routing.route)],
        ["决策来源", traceLabel(traceDecisionSourceLabels, routing.decision_source)],
        ["决策原因", traceReason(routing.reason_code)],
      ]));
    }
    body.append(element("h4", "trace-section-title", "本轮 Skill"));
    body.append(traceFields([
      ["手动指定", traceSkill(trace.requested_skill_id, "未指定")],
      ["已准备", traceSkill(trace.prepared_skill_id, "未准备")],
      ["实际应用", traceSkill(trace.applied_skill_id, "未应用")],
    ], "trace-fields trace-skill-fields"));
    body.append(element("p", "trace-note", "“实际应用”表示已携带技能说明开始调用模型，不代表调用成功。多步骤任务请查看各步骤的选择与准备记录；上方显示最近一次准备和应用的技能。"));
    if (traceText(trace.error_code, "")) {
      const warning = element("p", "trace-alert", `本轮原因 / 错误码：${traceReason(trace.error_code)}`);
      warning.dataset.kind = "error"; body.append(warning);
    }
    body.append(element("h4", "trace-section-title", "执行阶段"));
    const events = Array.isArray(trace.events)
      ? trace.events.filter((event) => event && typeof event === "object" && !Array.isArray(event))
        .slice().sort((left, right) => left.sequence - right.sequence)
      : [];
    if (events.length) {
      const list = element("ol", "trace-events");
      for (const event of events) list.append(renderTraceEvent(event));
      body.append(list);
    } else {
      body.append(element("p", "trace-empty", "本轮没有可展示的阶段记录，不能据此判断没有执行任何操作。"));
    }
    if (trace.events_truncated === true) {
      const warning = element("p", "trace-alert", "阶段记录已达到数量上限，部分步骤未保留；以上仅展示已记录的步骤，调用总数仍以本轮统计为准。");
      warning.dataset.kind = "warning"; body.append(warning);
    }
    details.append(body);
    return details;
  }

  const planStatusLabels = {
    completed: "全部完成", partial: "部分完成", failed: "未能完成", needs_clarification: "需要补充信息",
  };
  const planStepStatusLabels = {
    pending: "尚未执行", success: "已完成", insufficient_evidence: "未完整完成",
    needs_clarification: "需要补充信息", service_error: "执行异常", limit_reached: "达到调用上限",
    out_of_scope: "超出范围", skipped: "已跳过",
  };

  const answerValidationReasons = new Set([
    "missing_citations", "unknown_citation", "unverified_citation_metadata",
    "answer_contract_invalid", "answer_status_conflict",
  ]);

  function planStepLabel(step) {
    if (step.status === "insufficient_evidence"
      && step.answer_diagnostics?.declared_status === "insufficient_evidence"
      && step.answer_diagnostics?.validation_reason === "refusal_format_nonconforming") {
      return "模型报告资料不足（回复格式异常）";
    }
    if (answerValidationReasons.has(step.stop_reason)) return "回答校验未通过";
    if (step.stop_reason === "partial_tool_failure") return "部分工具执行失败";
    if (step.status === "insufficient_evidence") return "资料覆盖不足";
    return traceLabel(planStepStatusLabels, step.status, "状态未记录");
  }

  function renderPlan(result) {
    const plan = result.plan;
    if (!plan || typeof plan !== "object" || Array.isArray(plan) || plan.version !== 1
      || !Array.isArray(plan.steps)) return null;
    const steps = plan.steps.filter((step) => step && typeof step === "object" && !Array.isArray(step));
    const details = element("details", "execution-details plan-details");
    details.open = plan.status !== "completed";
    const summary = element("summary", "execution-summary");
    summary.append(element("span", "execution-summary-title", "任务执行步骤"));
    summary.append(element("span", "execution-summary-meta",
      `${steps.length} 步 · ${traceLabel(planStatusLabels, plan.status, "状态未记录")}`));
    details.append(summary);
    const body = element("div", "execution-body");
    if (!steps.length) {
      body.append(element("p", null, "计划尚未进入执行阶段。"));
      body.append(element("p", "trace-detail", traceReason(result.stop_reason)));
    } else {
      const groups = new Map();
      for (const step of steps) {
        if (step.status === "success") continue;
        const label = planStepLabel(step);
        const ids = groups.get(label) || [];
        ids.push(traceText(step.step_id, "步骤"));
        groups.set(label, ids);
      }
      if (groups.size) {
        body.append(element("p", "trace-detail",
          Array.from(groups, ([label, ids]) => `${label}：${ids.join("、")}`).join("；")));
      }
    }
    const list = element("ol", "plan-steps");
    for (const step of steps) {
      const item = element("li", "plan-step");
      const heading = element("div", "trace-event-heading");
      heading.append(element("strong", "plan-step-id", traceText(step.step_id, "步骤")));
      const badge = element("span", "plan-step-status", planStepLabel(step));
      badge.dataset.status = traceText(step.status, "unknown");
      heading.append(badge); item.append(heading);
      item.append(element("p", "plan-step-goal", traceText(step.goal, "未记录任务目标")));
      const dependencies = Array.isArray(step.depends_on)
        ? step.depends_on.filter((value) => typeof value === "string" && value.trim()) : [];
      const binding = step.topic_binding;
      const topicBinding = binding && typeof binding === "object" && !Array.isArray(binding)
        && typeof binding.topic === "string" && binding.topic.trim()
        && typeof binding.source_step_id === "string"
        && /^P[12]$/.test(binding.source_step_id) ? binding : null;
      const fields = [
        ["执行方式", traceLabel({ knowledge: "知识问答", mcp: "MCP 工具" }, step.kind)],
        ["依赖步骤", dependencies.length ? dependencies.join("、") : "无"],
        ...(topicBinding ? [
          ["搜索主题", topicBinding.topic],
          ["主题来源", `${topicBinding.source_step_id} 的用户描述（同一条请求）`],
        ] : []),
        ["模型调用", traceNumber(step.model_calls, " 次")],
        ["工具调用", traceNumber(step.tool_calls, " 次")],
        ...answerDiagnosticFields(step.answer_diagnostics),
      ];
      const stepEvents = Array.isArray(result.trace?.events)
        ? result.trace.events.filter((event) => event?.step_id === step.step_id) : [];
      const selection = stepEvents.find((event) => event.stage === "skill_select");
      if (selection) {
        fields.push(["技能模式", traceLabel(skillModeLabels, selection.skill_mode)]);
        fields.push(["选中技能", traceSkill(selection.selected_skill_id, "普通问答")]);
        fields.push(["检索策略", traceLabel(strategyLabels, selection.strategy_id)]);
        fields.push(["选择原因", traceReason(selection.reason_code)]);
        const applied = stepEvents.some((event) => event.stage === "model" && event.selected_skill_id === selection.selected_skill_id);
        fields.push(["技能进入回答模型", selection.selected_skill_id && applied ? "是" : "否"]);
      }
      if (traceText(step.stop_reason, "")) fields.push(["结束原因", traceReason(step.stop_reason)]);
      item.append(traceFields(fields, "trace-fields plan-step-fields"));
      list.append(item);
    }
    body.append(list);
    body.append(element("p", "trace-note plan-note",
      "资料覆盖不足表示当前检索证据不够，不等于服务出错，也不能证明整个知识库没有相关内容。各步骤共享本轮调用预算。"));
    details.append(body);
    return details;
  }

  const assessmentStatusLabels = {
    active: "进行中", paused: "已暂停", ready: "等待评分", completed: "已完成", cancelled: "已退出",
  };

  function assessmentData(result) {
    const data = result?.assessment;
    return data && typeof data === "object" && !Array.isArray(data)
      && typeof data.test_id === "string" && data.test_id.length > 0
      && Object.hasOwn(assessmentStatusLabels, data.status) ? data : null;
  }

  function assessmentCanSubmit(result, sessionId) {
    return Boolean(sessionId && state.current?.session_id === sessionId
      && state.turns[state.turns.length - 1] === result && !result.unsaved
      && !["service_error", "limit_reached"].includes(result.status)
      && !chatControlsLocked() && !state.needsRefresh);
  }

  function sendAssessmentCommand(command) {
    if (chatControlsLocked() || state.needsRefresh) return;
    ui["skill-select"].value = "";
    state.assessmentModeHint = true;
    updateControls();
    // 仍经由单次问答接口提交，沿用同一会话锁和断连恢复机制。
    void sendQuestion(null, command);
  }

  function assessmentButton(label, command, result, sessionId, className = "assessment-action") {
    const button = element("button", className, label);
    button.type = "button";
    button.dataset.assessmentControl = "true";
    button.dataset.assessmentCurrent = String(state.turns[state.turns.length - 1] === result
      && !result.unsaved && !["service_error", "limit_reached"].includes(result.status));
    button.disabled = !assessmentCanSubmit(result, sessionId);
    button.addEventListener("click", () => {
      // 捕获的会话和轮次必须仍有效，防止旧卡片、重复点击或切换会话后的事件提交。
      if (assessmentCanSubmit(result, sessionId)) sendAssessmentCommand(command);
    });
    return button;
  }

  function assessmentPercent(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100
      ? value : null;
  }

  function renderAssessmentResult(result) {
    const section = element("div", "assessment-result");
    if (!result || typeof result !== "object" || Array.isArray(result)) {
      section.append(element("p", "assessment-note", "结果详情暂不可用，请查看本轮回答。"));
      return section;
    }
    const heading = element("div", "assessment-type");
    heading.append(element("strong", null, typeof result.type === "string" ? result.type : "测试结果"));
    if (typeof result.type_name === "string") heading.append(element("span", null, result.type_name));
    section.append(heading, element("h4", "assessment-section-title", "维度得分比例"));
    section.append(element("p", "assessment-note", "以下比例来自本次答题计分，不代表人格判定的置信度。"));
    const dimensionLabels = {
      EI: ["外向 E", "内向 I"], SN: ["实感 S", "直觉 N"],
      TF: ["思考 T", "情感 F"], JP: ["判断 J", "知觉 P"],
    };
    for (const [key, labels] of Object.entries(dimensionLabels)) {
      const score = result.dimension_scores?.[key];
      if (!score || typeof score !== "object" || Array.isArray(score)) continue;
      const left = assessmentPercent(score[`${key[0]}_pct`]);
      const right = assessmentPercent(score[`${key[1]}_pct`]);
      const dimension = element("div", "assessment-dimension");
      const values = element("div", "assessment-dimension-labels");
      const format = (value) => value === null ? "未提供" : `${value.toLocaleString("zh-CN", { maximumFractionDigits: 1 })}%`;
      values.append(element("span", null, `${labels[0]} · ${format(left)}`), element("span", null, `${labels[1]} · ${format(right)}`));
      dimension.append(values);
      if (left !== null && right !== null) {
        const bar = element("div", "assessment-dimension-bar");
        bar.setAttribute("aria-hidden", "true");
        const fill = element("span"); fill.style.width = `${left}%`; bar.append(fill); dimension.append(bar);
      }
      if (result.ties?.[key] === true) {
        const tie = typeof score.result === "string" ? `，按评分规则归为 ${score.result}` : "，以评分规则为准";
        dimension.append(element("p", "assessment-tie", `本维度得分相同${tie}；本次回答未显示明确倾向。`));
      }
      section.append(dimension);
    }
    if (typeof result.notice === "string" && result.notice.trim()) {
      section.append(element("p", "assessment-notice", result.notice));
    }
    return section;
  }

  function renderAssessment(result) {
    const data = assessmentData(result);
    if (!data) return null;
    const sessionId = state.current?.session_id;
    const card = element("section", "assessment-card");
    card.setAttribute("aria-label", "MBTI 测试");
    const header = element("div", "assessment-header");
    header.append(element("h3", null, "MBTI 人格倾向测试"));
    const status = element("span", "assessment-status", assessmentStatusLabels[data.status]);
    status.dataset.status = data.status; header.append(status); card.append(header);
    if (Number.isInteger(data.total) && data.total > 0 && Number.isInteger(data.answered_count)
      && data.answered_count >= 0 && data.answered_count <= data.total) {
      const progress = element("progress", "assessment-progress");
      progress.max = data.total; progress.value = data.answered_count;
      progress.setAttribute("aria-label", `已完成 ${data.answered_count} / ${data.total} 题`);
      card.append(element("p", "assessment-progress-label", `已完成 ${data.answered_count} / ${data.total} 题`), progress);
    }
    const current = data.current_question;
    if (["active", "paused"].includes(data.status) && current && Number.isInteger(current.id)
      && typeof current.text === "string") {
      const number = Number.isInteger(current.number) ? `第 ${current.number} 题` : "当前题目";
      card.append(element("p", "assessment-question-number", number), element("p", "assessment-question", current.text));
      if (data.status === "active" && Array.isArray(data.options)) {
        const options = element("div", "assessment-options");
        options.setAttribute("role", "group"); options.setAttribute("aria-label", `${number}的回答选项，点选即提交`);
        for (const option of data.options) {
          if (!option || !Number.isInteger(option.value) || option.value < 1 || option.value > 5
            || typeof option.label !== "string") continue;
          const button = assessmentButton("", `MBTI答题 ${data.test_id} ${current.id} ${option.value}`,
            result, sessionId, "assessment-option");
          button.append(element("span", "assessment-option-value", String(option.value)), element("span", null, option.label));
          options.append(button);
        }
        card.append(options);
      }
    }
    if (data.status === "active" && !current) {
      card.append(element("p", "assessment-note", "已提交的答案已保留，下一题暂未取得，可重试获取。"));
    }
    if (data.status === "paused") card.append(element("p", "assessment-note", "进度已保留，继续测试后可提交下一题。"));
    if (data.status === "ready") card.append(element("p", "assessment-note", "题目已答完，可以查看计分结果。"));
    if (data.status === "cancelled") card.append(element("p", "assessment-note", "本次测试已退出。如需重新作答，可使用输入框上方的测试入口。"));
    if (data.status === "completed") card.append(renderAssessmentResult(data.result));
    const actions = element("div", "assessment-actions");
    if (data.status === "active" && !current) {
      actions.append(assessmentButton("重试获取题目", "继续 MBTI 测试", result, sessionId));
    }
    if (data.status === "active") actions.append(assessmentButton("暂停测试", "暂停 MBTI 测试", result, sessionId));
    if (data.status === "paused") actions.append(assessmentButton("继续测试", "继续 MBTI 测试", result, sessionId));
    if (data.status === "ready") actions.append(assessmentButton("查看结果", "查看 MBTI 结果", result, sessionId));
    if (["active", "paused", "ready"].includes(data.status)) {
      actions.append(assessmentButton("退出测试", "退出 MBTI 测试", result, sessionId));
    }
    if (actions.childNodes.length) card.append(actions);
    const latest = state.turns[state.turns.length - 1] === result;
    if (result.unsaved || ["service_error", "limit_reached"].includes(result.status)) {
      card.append(element("p", "assessment-readonly", "本轮未保存，卡片仅供查看。请刷新历史确认当前测试进度。"));
    } else if (!latest) {
      card.append(element("p", "assessment-readonly", "历史快照，仅供查看。继续作答请使用最新一轮卡片，或发送“继续 MBTI 测试”。"));
    } else if (data.status === "active" && current) {
      card.append(element("p", "assessment-note", "点击选项即提交。本次测试操作使用普通问答模式。"));
    }
    return card;
  }

  function displayQuestion(text) {
    const answer = typeof text === "string" ? /^MBTI答题\s+\S+\s+\d+\s+([1-5])$/.exec(text) : null;
    return answer ? `MBTI 答题：选择 ${answer[1]}` : text;
  }

  function message(role, text, result = null) {
    const article = element("article", `message message--${role}`);
    article.append(element("div", "message-avatar", role === "user" ? "你" : "A"));
    const content = element("div", "message-content");
    content.append(element("div", "message-label", role === "user" ? "你" : "Person A"));
    const body = element("div", "message-body");
    if (role === "user") body.textContent = displayQuestion(text);
    else body.append(markdown(text));
    content.append(body);
    if (result) {
      const assessment = renderAssessment(result);
      if (assessment) content.append(assessment);
      if (result.sources?.length) content.append(renderSources(result.sources));
      const meta = element("div", "message-meta");
      const label = result.plan?.status === "partial" ? ({
        plan_insufficient_evidence: "部分任务资料不足",
        plan_partial_execution: "部分任务执行异常",
        plan_partial_validation: "部分答案校验未通过",
        plan_answer_validation_failed: "答案校验未通过",
        plan_needs_clarification: "部分任务需要澄清",
      }[result.stop_reason] || "任务部分完成")
        : assessment && result.status === "success" ? "MBTI 测试" : statusLabels[result.status] || "回答结果";
      const badge = element("span", "status-badge", label);
      badge.dataset.status = result.status;
      meta.append(badge);
      if (result.unsaved || ["service_error", "limit_reached"].includes(result.status)) {
        meta.append(element("span", "unsaved-label", "本轮未保存"));
      }
      const copy = element("button", "copy-answer", "复制回答"); copy.type = "button";
      copy.addEventListener("click", () => copyText(text, copy)); meta.append(copy);
      content.append(meta);
      const plan = renderPlan(result);
      if (plan) content.append(plan);
      content.append(renderExecutionDetails(result));
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
        const pending = message("assistant", "正在处理请求，请稍候…");
        pending.classList.add("loading-message");
        const progress = element("div", "live-progress");
        const label = element("p", "live-progress-title", "正在接收执行进度…");
        label.setAttribute("role", "status");
        label.setAttribute("aria-live", "polite");
        progress.append(label, element("ol", "live-progress-events"));
        pending.querySelector(".message-body").replaceChildren(progress);
        fragment.append(pending);
      }
    }
    ui.conversation.replaceChildren(fragment);
    renderLiveProgress();
    ui.welcome.hidden = state.turns.length > 0 || Boolean(state.pendingQuestion);
    ui.conversation.hidden = !ui.welcome.hidden;
    requestAnimationFrame(() => { ui["conversation-scroll"].scrollTop = ui["conversation-scroll"].scrollHeight; });
  }

  function receiveProgress(event) {
    if (!Number.isInteger(event.sequence) || event.sequence < 0
      || !Object.hasOwn(traceStageLabels, event.stage)
      || !Object.hasOwn(traceStatusLabels, event.status)) return;
    const position = state.liveProgress.findIndex((item) => item.sequence === event.sequence);
    if (position >= 0) state.liveProgress[position] = event;
    else if (state.liveProgress.length < 128) state.liveProgress.push(event);
    renderLiveProgress();
  }

  function renderLiveProgress() {
    if (state.activeSend && state.activeSend.sessionId !== state.current?.session_id) return;
    const panel = ui.conversation.querySelector(".live-progress");
    if (!panel) return;
    const events = state.liveProgress.slice().sort((a, b) => a.sequence - b.sequence);
    const list = document.createDocumentFragment();
    for (const event of events.slice(-8)) {
      const item = element("li", "live-progress-event");
      item.dataset.status = event.status;
      const prefix = event.step_id ? `${traceText(event.step_id)} · ` : "";
      item.append(element("span", "", `${prefix}${traceStageLabels[event.stage]}`));
      item.append(element("span", "live-progress-state", traceStatusLabels[event.status]));
      if (event.stage === "skill_select" && event.status !== "running") {
        item.append(element("small", "live-progress-detail", event.selected_skill_id
          ? `选中：${traceSkill(event.selected_skill_id, "普通问答")}`
          : traceReasonLabels[event.reason_code] || "使用普通问答"));
      } else if (event.strategy_id) {
        item.append(element("small", "live-progress-detail", traceLabel(strategyLabels, event.strategy_id)));
      }
      list.append(item);
    }
    panel.querySelector("ol").replaceChildren(list);
    const running = events.filter((event) => event.status === "running").at(-1);
    const title = panel.querySelector(".live-progress-title");
    const next = running ? `正在${traceStageLabels[running.stage]}…`
      : events.length ? "等待后续步骤或最终结果…" : "正在接收执行进度…";
    if (title.textContent !== next) title.textContent = next;
  }

  function displayDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric" }).format(date);
  }

  function resetSummaryPanel() {
    state.memorySnapshot = null;
    state.summaryRequestVersion += 1;
    ui["session-summary"].open = false;
    ui["session-summary-meta"].textContent = "展开后读取当前会话已保存的摘要。";
    ui["session-summary-content"].textContent = "";
  }

  async function refreshSummary() {
    const id = state.current?.session_id;
    if (!id || readingLocked()) return;
    const version = ++state.summaryRequestVersion;
    setReadOperation(true);
    ui["session-summary-meta"].textContent = "正在读取摘要…";
    try {
      const result = await request(`/sessions/${encodeURIComponent(id)}/summary`);
      if (state.current?.session_id !== id || version !== state.summaryRequestVersion) return;
      ui["session-summary-content"].textContent = result.summary || "当前没有会话摘要（可能尚未生成，或已被清空）。";
      ui["session-summary-meta"].textContent = result.covered_turn_no
        ? `记忆处理至第 ${result.covered_turn_no} 轮（包含已归纳或主动遗忘的轮次）；完整历史仍保留。`
        : "较早的知识问答和聊天会在超出最近历史窗口后归纳；也可能尚未启用或更新未完成。";
    } catch (error) {
      if (state.current?.session_id !== id || version !== state.summaryRequestVersion) return;
      ui["session-summary-meta"].textContent = `摘要暂未刷新：${error.message}`;
    } finally {
      setReadOperation(false);
    }
  }

  function memoryControls(busy) {
    state.memoryBusy = busy;
    for (const id of ["memory-reload", "memory-close", "memory-text"]) ui[id].disabled = busy;
    for (const id of ["memory-save", "memory-forget"]) ui[id].disabled = busy || !state.memorySnapshot;
  }

  async function loadEditableMemory() {
    const id = state.memorySessionId;
    if (!id || chatControlsLocked() || state.memoryBusy) return;
    state.memorySnapshot = null;
    memoryControls(true); setReadOperation(true);
    ui["memory-message"].textContent = "正在读取…";
    try {
      const value = await request(`/sessions/${encodeURIComponent(id)}/memory`);
      if (state.current?.session_id !== id) return;
      state.memorySnapshot = value;
      ui["memory-text"].value = value.summary;
      ui["memory-message"].textContent = `当前共 ${value.expected_turn_no} 轮；前 ${value.excluded_through} 轮已排除出记忆。保存会替换背景并排除截至当前的全部旧轮次。`;
    } catch (error) {
      ui["memory-message"].textContent = `读取失败：${error.message}`;
    } finally { setReadOperation(false); memoryControls(false); }
  }

  async function saveEditableMemory(forget) {
    const snapshot = state.memorySnapshot;
    if (!snapshot || state.memoryBusy || chatControlsLocked() || state.current?.session_id !== snapshot.session_id) return;
    if (!window.confirm(forget ? "清空当前会话记忆，并停止注入截至现在的全部旧对话？聊天记录与学习画像会保留。" : "用编辑后的内容替换会话记忆，并停止注入截至现在的旧对话？")) return;
    memoryControls(true); setOperation("memory");
    ui["memory-message"].textContent = "正在保存…";
    try {
      await request(`/sessions/${encodeURIComponent(snapshot.session_id)}/memory`, {
        method: "POST", body: { summary: forget ? "" : ui["memory-text"].value,
          expected_turn_no: snapshot.expected_turn_no, expected_updated_at: snapshot.expected_updated_at },
      });
      state.summaryRequestVersion += 1;
      ui["session-summary-content"].textContent = "记忆已更新，请刷新查看。";
      ui["session-summary-meta"].textContent = "此前轮次已排除出自动记忆；新对话仍会积累记忆。";
      ui["memory-message"].textContent = "已保存。旧聊天保留供查看；后续回答使用新的记忆边界。可重新读取确认内容。";
      if (forget) ui["memory-text"].value = "";
    } catch (error) {
      ui["memory-message"].textContent = `保存未确认：${error.message} 请重新读取确认状态后再编辑，不要重复提交。`;
    } finally {
      state.memorySnapshot = null;
      setOperation(null); memoryControls(false);
    }
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
      button.disabled = readingLocked();
      button.append(element("span", "session-item-title", session.title || "未命名会话"));
      button.append(element("span", "session-item-date", displayDate(session.updated_at)));
      button.addEventListener("click", () => selectSession(session.session_id));
      item.append(button); fragment.append(item);
    }
    if (!sessions.length) fragment.append(element("li", "empty-sessions", search ? "已加载会话中没有匹配项" : "从一个问题，开始你的学习记录。"));
    ui["session-list"].replaceChildren(fragment);
    ui["load-more-sessions"].hidden = !state.hasMore;
    ui["load-more-sessions"].disabled = readingLocked();
  }

  function updateControls() {
    const busy = chatControlsLocked();
    ui["session-summary"].hidden = !state.current;
    ui["session-summary-refresh"].disabled = readingLocked() || !state.current;
    ui["memory-open"].disabled = chatControlsLocked() || !state.current;
    ui["new-session"].disabled = busy;
    for (const id of ["clear-session", "delete-session"]) ui[id].disabled = busy || !state.current;
    for (const id of ["refresh-sessions", "load-more-sessions"]) ui[id].disabled = readingLocked();
    ui["refresh-history"].disabled = readingLocked() || !state.current;
    ui.question.disabled = state.readOperation || Boolean(state.operation && state.operation !== "send");
    ui["response-mode"].disabled = ui.question.disabled;
    ui["skill-select"].disabled = ui.question.disabled || ui["response-mode"].value === "chat";
    ui["mbti-start"].disabled = busy || state.needsRefresh;
    ui["send-button"].disabled = busy || state.needsRefresh || !ui.question.value.trim();
    ui["send-button"].setAttribute("aria-label", state.operation === "send" ? "正在生成回答" : "发送问题");
    ui["composer-form"].setAttribute("aria-busy", String(isBusy()));
    ui["current-title"].textContent = state.current?.title || (state.current ? "未命名会话" : "新的学习会话");
    ui["session-subtitle"].textContent = state.current ? "历史保存在本机 · 可继续追问" : "Python / LangChain · 本地文档知识库";
    const businessHint = state.businessState?.waiting_for_input
      ? (state.businessState.clarification_question || "当前业务需要补充信息。")
      : state.businessState?.phase === "execution_unknown"
        ? "上次操作结果尚未确认；不会自动重放，请明确发起新的操作。"
        : "";
    ui["composer-hint"].textContent = state.operation === "send"
      ? "正在生成回答；可输入草稿、切换会话或查看历史。完成后可继续发送，离开页面不会取消任务。"
      : state.knowledgeBusy ? "知识库操作进行中，完成后可继续提问。"
      : state.knowledgeNeedsRefresh ? "请先打开「知识库管理」刷新状态，确认上次操作结果。"
      : state.needsRefresh ? "请先点击上方「刷新历史」确认本轮结果，再继续提问。"
      : state.operation ? "正在同步会话…"
      : businessHint || (state.assessmentModeHint ? "MBTI 操作已使用普通问答模式；可点选题目作答。" : "Enter 发送 · Shift + Enter 换行");
    ui["char-count"].textContent = `${ui.question.value.length.toLocaleString("zh-CN")} / 10,000`;
    for (const button of document.querySelectorAll("[data-prompt]")) button.disabled = busy;
    for (const button of document.querySelectorAll("button.source-card")) button.disabled = isBusy();
    for (const button of document.querySelectorAll("[data-assessment-control]")) {
      button.disabled = busy || state.needsRefresh || button.dataset.assessmentCurrent !== "true";
    }
    renderSessions();
    updateKnowledgeControls();
    renderOperationStatus();
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

  async function loadBusinessState(id) {
    const result = await request(`/sessions/${encodeURIComponent(id)}/business-state`);
    return result?.state && typeof result.state === "object" ? result.state : null;
  }

  function resetConversation() {
    resetSummaryPanel();
    state.current = null; state.turns = []; state.pendingQuestion = ""; state.needsRefresh = false; state.assessmentModeHint = false; state.businessState = null;
    rememberSession(null); ui.question.value = ""; fitInput();
    renderConversation(); updateControls();
  }

  async function selectSession(id, confirmRecovery = false) {
    if (readingLocked()) return;
    saveDraft();
    closeSidebar(); notice(""); setReadOperation(true);
    try {
      let idle = false;
      if (confirmRecovery) {
        try { idle = (await request("/health")).busy === false; }
        catch { /* 无法确认空闲时不解除恢复保护。 */ }
      }
      const revision = state.answerRevision;
      const record = await request(`/sessions/${encodeURIComponent(id)}`);
      let turns = await loadHistory(id);
      let businessState = await loadBusinessState(id);
      // 读取期间原请求可能刚完成，重新读取已提交结果，避免旧快照覆盖它。
      if (revision !== state.answerRevision) {
        turns = await loadHistory(id);
        businessState = await loadBusinessState(id);
      }
      state.current = record;
      state.turns = [...turns, ...(state.unsavedAnswers.get(id) || [])];
      state.pendingQuestion = pendingFor(id);
      if (confirmRecovery && idle && !state.activeSend) state.uncertainSessions.delete(id);
      state.needsRefresh = state.uncertainSessions.has(id);
      state.businessState = businessState;
      resetSummaryPanel(); state.assessmentModeHint = false;
      rememberSession(id); ui.question.value = state.drafts.get(id) || "";
      fitInput(); renderConversation();
      return true;
    } catch (error) {
      notice(error.message);
    } finally { setReadOperation(false); }
  }

  async function refreshHistory() {
    if (readingLocked() || !state.current) return;
    await selectSession(state.current.session_id, true);
  }

  async function sendQuestion(event, command = null) {
    event?.preventDefault();
    const question = command === null ? ui.question.value.trim() : command;
    if (!question || chatControlsLocked() || state.needsRefresh) return;
    if (question.length > 10000) { notice(errorMessages.validation_error); return; }
    const responseMode = command === null ? ui["response-mode"].value : "auto";
    const choice = command === null && responseMode !== "chat" ? ui["skill-select"].value : "";
    const skillMode = choice === "auto" ? "auto" : choice ? "manual" : "off";
    const skillId = skillMode === "manual" ? choice : null;
    // 先消费已提交草稿，此后的输入属于下一条消息，不在请求结束时清空。
    if (command === null) ui.question.value = "";
    saveDraft();
    notice(""); closeSidebar(); setOperation("send");
    let submitted = false;
    let sessionId = state.current?.session_id;
    try {
      if (!sessionId) {
        state.current = await request("/sessions", { method: "POST", body: { title: question.slice(0, 80) }, timeout: 0 });
        sessionId = state.current.session_id;
        rememberSession(sessionId); state.sessions.unshift(state.current);
      }
      state.activeSend = { sessionId, question, initialTurnCount: state.turns.length };
      state.pendingQuestion = question; state.liveProgress = [];
      updateControls(); fitInput(); renderConversation();
      submitted = true;
      const result = await streamAnswer(`/sessions/${encodeURIComponent(sessionId)}/messages/stream`,
        { question, skill_mode: skillMode, skill_id: skillId, response_mode: responseMode }, receiveProgress);
      if (state.activeSend) state.activeSend.completed = true;
      state.answerRevision += 1;
      state.uncertainSessions.delete(sessionId);
      if (state.current?.session_id === sessionId) {
        // 历史刷新可能已读到本轮落库结果，避免 SSE 到达后再次追加。
        const last = state.turns.at(-1);
        const alreadyLoaded = state.turns.length > state.activeSend.initialTurnCount &&
          last?.question === result.question && last?.answer === result.answer;
        if (!alreadyLoaded) state.turns.push(result);
        state.pendingQuestion = ""; state.needsRefresh = false;
        if (result.business_state && typeof result.business_state === "object") state.businessState = result.business_state;
        renderConversation();
      } else {
        notice("原会话的回答已完成，切回该会话可查看。", "info");
      }
      try { await loadSessions(); }
      catch { notice("回答已返回，会话列表暂未刷新，可稍后手动刷新。", "info"); }
    } catch (error) {
      if (state.activeSend) state.activeSend.completed = true;
      state.answerRevision += 1;
      const uncertain = submitted && ["network_error", "request_timeout", "invalid_response", "internal_error"].includes(error.code);
      if (uncertain) state.uncertainSessions.add(sessionId);
      if (error.body?.result) {
        const result = { ...error.body.result, unsaved: error.body.persisted !== true };
        if (result.unsaved) state.unsavedAnswers.set(sessionId, [...(state.unsavedAnswers.get(sessionId) || []), result]);
        if (state.current?.session_id === sessionId) {
          state.turns.push(result);
          if (error.body.persisted === true && result.business_state) state.businessState = result.business_state;
        }
      }
      if (state.current?.session_id === sessionId || !sessionId) {
        state.pendingQuestion = ""; state.needsRefresh = uncertain;
        if (!error.body?.result && command === null && !ui.question.value) {
          ui.question.value = question;
          state.drafts.set(sessionId || "", question);
        }
      } else if (!error.body?.result && command === null && !state.drafts.get(sessionId)) {
        state.drafts.set(sessionId, question);
      }
      notice(error.message + (uncertain ? " 后台可能仍在处理，请稍后刷新原会话历史，避免重复提交。" : ""));
    } finally {
      state.activeSend = null; state.liveProgress = []; state.pendingQuestion = "";
      setOperation(null); renderConversation(); fitInput();
      if (!state.readOperation) ui.question.focus();
      if (!state.needsRefresh && ui["session-summary"].open) void refreshSummary();
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
    if (chatControlsLocked() || !state.current) return;
    const id = state.current.session_id;
    if (!await confirmAction(kind)) return;
    if (chatControlsLocked() || state.current?.session_id !== id) return;
    notice(""); setOperation(kind);
    try {
      await request(`/sessions/${encodeURIComponent(id)}${kind === "clear" ? "/messages" : ""}`, { method: "DELETE", timeout: 0 });
      state.unsavedAnswers.delete(id); state.uncertainSessions.delete(id); state.drafts.delete(id);
      if (kind === "delete") {
        state.sessions = state.sessions.filter((item) => item.session_id !== id);
        resetConversation();
      } else {
        resetSummaryPanel();
        state.turns = []; state.pendingQuestion = ""; state.needsRefresh = false; state.businessState = null; renderConversation();
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
    if (readingLocked()) return;
    notice(""); setReadOperation(true);
    try { await loadSessions(append); }
    catch (error) { notice(error.message); }
    finally { setReadOperation(false); }
    void checkHealth();
  }

  ui["composer-form"].addEventListener("submit", sendQuestion);
  ui["mbti-start"].addEventListener("click", () => sendAssessmentCommand("开始 MBTI 测试"));
  ui["response-mode"].addEventListener("change", updateControls);
  ui["skill-select"].addEventListener("change", () => { state.assessmentModeHint = false; updateControls(); });
  ui.question.addEventListener("input", () => { fitInput(); updateControls(); });
  ui.question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
      event.preventDefault(); ui["composer-form"].requestSubmit();
    }
  });
  ui["new-session"].addEventListener("click", () => {
    if (chatControlsLocked()) return;
    resetConversation(); notice(""); closeSidebar(); ui.question.focus();
  });
  ui["session-search"].addEventListener("input", renderSessions);
  ui["load-more-sessions"].addEventListener("click", () => refreshSessions(true));
  ui["refresh-sessions"].addEventListener("click", () => refreshSessions());
  ui["refresh-history"].addEventListener("click", refreshHistory);
  ui["memory-open"].addEventListener("click", () => {
    if (chatControlsLocked() || !state.current) return;
    state.memorySessionId = state.current.session_id;
    ui["memory-dialog"].showModal();
    void loadEditableMemory();
  });
  ui["memory-reload"].addEventListener("click", loadEditableMemory);
  ui["memory-save"].addEventListener("click", () => void saveEditableMemory(false));
  ui["memory-forget"].addEventListener("click", () => void saveEditableMemory(true));
  ui["memory-close"].addEventListener("click", () => { if (!state.memoryBusy) ui["memory-dialog"].close(); });
  ui["memory-dialog"].addEventListener("cancel", (event) => { if (state.memoryBusy) event.preventDefault(); });
  ui["session-summary-refresh"].addEventListener("click", refreshSummary);
  ui["session-summary"].addEventListener("toggle", () => {
    if (ui["session-summary"].open) void refreshSummary();
  });
  ui["clear-session"].addEventListener("click", () => manageSession("clear"));
  ui["delete-session"].addEventListener("click", () => manageSession("delete"));
  ui["notice-dismiss"].addEventListener("click", () => notice(""));
  ui["confirm-cancel"].addEventListener("click", () => ui["confirm-dialog"].close("cancel"));
  ui["confirm-accept"].addEventListener("click", () => ui["confirm-dialog"].close("confirm"));
  ui["knowledge-open"].addEventListener("click", openKnowledgeManager);
  ui["knowledge-close"].addEventListener("click", () => ui["knowledge-dialog"].close());
  ui["knowledge-refresh"].addEventListener("click", openKnowledgeManager);
  ui["knowledge-upload-form"].addEventListener("submit", uploadKnowledgeDocument);
  ui["knowledge-build"].addEventListener("click", () => runKnowledgeIndex("build"));
  ui["knowledge-update"].addEventListener("click", () => runKnowledgeIndex("update"));
  ui["knowledge-preview-prev"].addEventListener("click", () => changePreviewPage(-1));
  ui["knowledge-preview-next"].addEventListener("click", () => changePreviewPage(1));
  ui["knowledge-preview-jump-form"].addEventListener("submit", jumpToPreviewLine);
  ui["knowledge-preview-reload"].addEventListener("click", () => {
    if (isBusy() || !state.knowledgePreviewRequest) return;
    const { path, contentHash, startLine } = state.knowledgePreviewRequest;
    void openKnowledgeDocument(path, contentHash, startLine, { continuing: true });
  });
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
      if (chatControlsLocked()) return;
      ui.question.value = button.dataset.prompt; fitInput(); updateControls(); ui.question.focus();
    });
  }
  window.addEventListener("beforeunload", (event) => {
    if (state.operation === "send" || ["upload", "build", "update"].includes(state.knowledgeOperation)) {
      event.preventDefault(); event.returnValue = "";
    }
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) void checkHealth(); });
  window.addEventListener("online", checkHealth);
  window.addEventListener("offline", checkHealth);

  async function initialize() {
    try { state.knowledgeNeedsRefresh = sessionStorage.getItem(KNOWLEDGE_PENDING_KEY) === "pending"; }
    catch { /* 可选的断连恢复标记。 */ }
    renderKnowledgeStatus();
    renderConversation(); setOperation("sessions"); void checkHealth();
    try {
      // 一个请求失败时，仍等另一个请求结束后再释放页面业务锁。
      const [skillsResult, sessionsResult] = await Promise.allSettled([request("/skills"), loadSessions()]);
      if (skillsResult.status === "fulfilled") {
        for (const skill of skillsResult.value.items) {
          const option = element("option", null, skill.description);
          option.value = skill.skill_id;
          ui["skill-select"].append(option);
        }
      }
      const errors = [skillsResult, sessionsResult]
        .filter((result) => result.status === "rejected").map((result) => result.reason.message);
      if (errors.length) notice([...new Set(errors)].join(" "));
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
