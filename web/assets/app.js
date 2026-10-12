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
    "confirm-description", "confirm-cancel", "confirm-accept", "knowledge-open",
    "knowledge-dialog", "knowledge-close", "knowledge-message", "knowledge-refresh",
    "knowledge-status-cards", "knowledge-build", "knowledge-update",
    "knowledge-upload-form", "knowledge-file", "knowledge-document-count",
    "knowledge-document-list", "knowledge-preview-title", "knowledge-preview-meta",
    "knowledge-preview-content", "operation-status", "knowledge-version-note",
    "knowledge-preview-warning", "knowledge-preview-prev", "knowledge-preview-next",
    "knowledge-preview-page", "knowledge-preview-jump-form", "knowledge-preview-line",
    "knowledge-preview-jump-button", "knowledge-preview-reload",
  ].map((id) => [id, byId(id)]));
  const PAGE_SIZE = 100;
  const PREVIEW_PAGE_SIZE = 200;
  const STORAGE_KEY = "person-a.current-session";
  const KNOWLEDGE_PENDING_KEY = "person-a.pending-knowledge-operation";
  const state = {
    sessions: [], sessionOffset: 0, hasMore: false, current: null, turns: [],
    operation: null, pendingQuestion: "", needsRefresh: false,
    knowledgeBusy: false, knowledgeOperation: null, knowledgeNeedsRefresh: false,
    knowledgeDocuments: [], knowledgeDocumentsLoaded: false, knowledgeDocumentsStale: true,
    knowledgeStatus: null, knowledgeStatusStale: true,
    knowledgePreviewPath: null, knowledgePreview: null, knowledgePreviewRequest: null,
    knowledgePreviewReferenceHash: null, knowledgePreviewChanged: false,
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
  function chatControlsLocked() { return isBusy() || state.knowledgeNeedsRefresh; }

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
    classify: "理解问题", intent: "识别意图", route: "选择执行路线", initial_retrieval: "初始检索", skill_prepare: "准备 Skill",
    model: "模型调用", tool: "工具调用", finalize: "整理回答与核验引用",
  };
  const traceStatusLabels = { success: "完成", empty: "无结果", error: "失败", skipped: "已跳过" };
  const traceOutcomeLabels = { completed: "流程完成", failed: "执行失败", interrupted: "执行中断" };
  const traceDecisionSourceLabels = { rules: "规则", model: "模型", fallback: "回退" };
  const traceIntentKindLabels = {
    knowledge: "知识问答", greeting: "问候", scope: "能力范围", needs_clarification: "需要澄清",
    out_of_scope: "超出范围", mcp_tool: "MCP 工具请求",
  };
  const traceRouteKindLabels = { knowledge: "知识问答", mcp: "MCP 工具", direct: "直接回复" };
  const traceSkillLabels = {
    "concept-explanation": "概念讲解", "concept-comparison": "概念比较", "code-explanation": "代码解读",
  };
  const traceReasonLabels = {
    answer_completed: "回答完成", greeting_handled: "已回复问候", scope_explained: "已说明助手范围",
    question_is_ambiguous: "问题需要澄清", question_out_of_scope: "问题超出当前范围",
    index_not_built: "索引尚未建立", initial_retrieval_failed: "初始检索失败",
    initial_retrieval_empty: "初始检索没有结果", evidence_context_limit: "资料超出上下文限制",
    model_call_budget_exhausted: "达到模型调用上限", tool_budget_exhausted: "达到工具调用上限",
    followup_search_budget_exhausted: "达到补充检索上限", model_empty_response: "模型未返回有效内容",
    model_did_not_finish: "模型未生成最终回答", model_requested_clarification: "模型请求澄清",
    model_reported_insufficient_evidence: "模型报告资料不足", partial_answer: "资料只能支持部分回答",
    model_authentication_failed: "模型鉴权失败", model_timeout: "模型调用超时",
    model_not_found: "模型不存在", model_request_invalid: "模型请求无效",
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
    if (Number.isInteger(event.result_count) && event.result_count >= 0) {
      const label = event.stage === "finalize" ? "引用来源数量" : "返回结果数量";
      fields.push([label, traceNumber(event.result_count, " 个")]);
    }
    if (traceText(event.index_version, "")) fields.push(["阶段索引版本", event.index_version]);
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
      ["已选择", traceSkill(trace.requested_skill_id, "未选择")],
      ["已准备", traceSkill(trace.prepared_skill_id, "未准备")],
      ["实际应用", traceSkill(trace.applied_skill_id, "未应用")],
    ], "trace-fields trace-skill-fields"));
    body.append(element("p", "trace-note", "“实际应用”表示已携带技能说明开始调用模型，不代表调用成功或模型遵循全部步骤。"));
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
      button.disabled = chatControlsLocked();
      button.append(element("span", "session-item-title", session.title || "未命名会话"));
      button.append(element("span", "session-item-date", displayDate(session.updated_at)));
      button.addEventListener("click", () => selectSession(session.session_id));
      item.append(button); fragment.append(item);
    }
    if (!sessions.length) fragment.append(element("li", "empty-sessions", search ? "已加载会话中没有匹配项" : "从一个问题，开始你的学习记录。"));
    ui["session-list"].replaceChildren(fragment);
    ui["load-more-sessions"].hidden = !state.hasMore;
    ui["load-more-sessions"].disabled = chatControlsLocked();
  }

  function updateControls() {
    const busy = chatControlsLocked();
    for (const id of ["new-session", "refresh-sessions", "load-more-sessions"]) ui[id].disabled = busy;
    for (const id of ["refresh-history", "clear-session", "delete-session"]) ui[id].disabled = busy || !state.current;
    ui.question.disabled = busy;
    ui["skill-select"].disabled = busy;
    ui["send-button"].disabled = busy || state.needsRefresh || !ui.question.value.trim();
    ui["send-button"].setAttribute("aria-label", state.operation === "send" ? "正在生成回答" : "发送问题");
    ui["composer-form"].setAttribute("aria-busy", String(isBusy()));
    ui["current-title"].textContent = state.current?.title || (state.current ? "未命名会话" : "新的学习会话");
    ui["session-subtitle"].textContent = state.current ? "历史保存在本机 · 可继续追问" : "Python / LangChain · 本地文档知识库";
    ui["composer-hint"].textContent = state.operation === "send"
      ? "正在等待完整回答。离开页面不会取消后台任务。"
      : state.knowledgeBusy ? "知识库操作进行中，完成后可继续提问。"
      : state.knowledgeNeedsRefresh ? "请先打开「知识库管理」刷新状态，确认上次操作结果。"
      : state.needsRefresh ? "请先点击上方「刷新历史」确认本轮结果，再继续提问。"
      : state.operation ? "正在同步会话…" : "Enter 发送 · Shift + Enter 换行";
    ui["char-count"].textContent = `${ui.question.value.length.toLocaleString("zh-CN")} / 10,000`;
    for (const button of document.querySelectorAll("[data-prompt]")) button.disabled = busy;
    for (const button of document.querySelectorAll("button.source-card")) button.disabled = isBusy();
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

  function resetConversation() {
    state.current = null; state.turns = []; state.pendingQuestion = ""; state.needsRefresh = false;
    rememberSession(null); ui.question.value = ""; fitInput();
    renderConversation(); updateControls();
  }

  async function selectSession(id) {
    if (chatControlsLocked()) return;
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
    if (chatControlsLocked() || !state.current) return;
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
    if (!question || chatControlsLocked() || state.needsRefresh) return;
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
    if (chatControlsLocked() || !state.current) return;
    const id = state.current.session_id;
    if (!await confirmAction(kind)) return;
    if (chatControlsLocked() || state.current?.session_id !== id) return;
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
    if (chatControlsLocked()) return;
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
    if (chatControlsLocked()) return;
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
