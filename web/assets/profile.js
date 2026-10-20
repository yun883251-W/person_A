/* 手动学习画像只由API保存；浏览器不持久化画像正文。 */
(() => {
  "use strict";
  const dialog = document.getElementById("profile-dialog");
  const form = document.getElementById("profile-form");
  const message = document.getElementById("profile-message");
  const enabled = document.getElementById("profile-enabled");
  const statement = document.getElementById("profile-statement");
  const candidateBox = document.getElementById("profile-candidates");
  const fieldLabels = {python_level: "Python 基础", learning_goal: "学习目标",
    preferred_style: "回答方式", development_environment: "开发环境",
    learning_progress: "学习进度", preferred_language: "语言偏好"};
  const fields = ["python_level", "learning_goal", "preferred_style", "development_environment", "learning_progress", "preferred_language"];
  let revision = null;
  let busy = false;
  const confirmed = new Set();
  const metadataViews = new Map();
  for (const field of fields) {
    const row = document.createElement("div");
    row.className = "profile-field-meta";
    const detail = document.createElement("small");
    const button = document.createElement("button");
    button.type = "button";
    button.id = `profile-confirm-${field}`;
    button.textContent = "确认仍有效";
    button.addEventListener("click", () => {
      if (busy || revision === null) return;
      if (!form.elements.namedItem(field).value.trim()) {
        message.textContent = "请先填写该字段，再确认其有效性。";
        return;
      }
      confirmed.add(field);
      button.textContent = "待保存确认";
      message.textContent = "已标记，点击保存后更新该字段的确认时间。";
    });
    row.append(detail, button);
    form.elements.namedItem(field).closest("label").after(row);
    metadataViews.set(field, {detail, button});
  }

  function controls() {
    for (const control of form.querySelectorAll("input, textarea, button")) {
      control.disabled = busy || (revision === null && !["profile-close", "profile-reload"].includes(control.id));
    }
  }

  async function request(path, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(path, {
        method: body === undefined ? "GET" : "POST",
        headers: body === undefined ? {} : {"Content-Type": "application/json"},
        body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal,
        cache: "no-store",
      });
      if (!response.ok) {
        const error = new Error(response.status === 409 ? "画像已在其他页面修改，请重新加载后编辑。当前输入尚未保存。"
          : response.status === 422 ? "字段格式或长度不正确，请检查输入。"
          : response.status === 502 ? "模型没有返回可核验的候选，本次未修改画像。"
          : "画像服务暂时不可用，请稍后重新加载。");
        error.status = response.status;
        throw error;
      }
      const data = await response.json();
      if (path === "/profile/propose") {
        if (!Number.isInteger(data.revision) || !Array.isArray(data.candidates) || data.candidates.length > 6
          || data.candidates.some((item) => !item || !fields.includes(item.field)
            || typeof item.value !== "string" || typeof item.quote !== "string")) {
          throw new Error("候选响应异常，本次未修改画像。");
        }
        return data;
      }
      if (!Number.isInteger(data.revision) || data.revision < 0 || typeof data.enabled !== "boolean" || !data.fields || typeof data.fields !== "object") {
        throw new Error("画像响应格式异常，请重新加载。");
      }
      return data;
    } finally { clearTimeout(timer); }
  }

  function display(record) {
    candidateBox.replaceChildren();
    statement.value = "";
    revision = record.revision;
    confirmed.clear();
    enabled.checked = record.enabled;
    for (const field of fields) {
      form.elements.namedItem(field).value = record.fields[field] || "";
      const {detail, button} = metadataViews.get(field);
      button.textContent = "确认仍有效";
      const meta = record.field_metadata?.[field];
      const timestamp = meta?.updated_at ? Date.parse(meta.updated_at) : NaN;
      const outdated = Number.isFinite(timestamp) && Date.now() - timestamp > (
        field === "learning_progress" ? 30 : 90
      ) * 86400000;
      detail.textContent = !record.fields[field] ? "未设置"
        : !Number.isFinite(timestamp) ? "历史手动填写 · 字段时间未知，建议重新确认"
        : `手动填写／确认 · ${new Date(timestamp).toLocaleString()}${outdated ? " · 较久未确认，建议核对" : ""}`;
    }
  }

  async function perform(operation, success) {
    if (busy) return;
    busy = true; controls(); message.textContent = "处理中…";
    try {
      display(await operation()); message.textContent = success;
    } catch (error) {
      // 写入超时可能已提交；要求重新读取，不自动重放写入。
      if (error.status !== 422) revision = null;
      message.textContent = error.name === "AbortError" ? "请求超时，保存状态未知。请重新加载确认，不会自动重试。" : error.message;
    } finally { busy = false; controls(); }
  }

  function load() {
    return perform(() => request("/profile"), "已加载本机学习偏好。");
  }
  document.getElementById("profile-propose").addEventListener("click", async () => {
    if (busy || revision === null) return;
    if (!statement.value.trim()) { message.textContent = "请先填写你的描述。"; return; }
    const requestRevision = revision;
    busy = true; controls(); candidateBox.replaceChildren();
    message.textContent = "正在提取候选，不会自动保存…";
    try {
      const data = await request("/profile/propose", {text: statement.value, expected_revision: requestRevision});
      if (data.revision !== requestRevision) throw new Error("候选版本不一致，请重新加载。");
      for (const candidate of data.candidates) {
        const row = document.createElement("section");
        const heading = document.createElement("strong");
        heading.textContent = fieldLabels[candidate.field];
        const value = document.createElement("p");
        value.textContent = `候选：${candidate.value}`;
        const previous = document.createElement("p");
        previous.textContent = `当前编辑值：${form.elements.namedItem(candidate.field).value || "未设置"}`;
        const quote = document.createElement("p");
        quote.textContent = `依据原话：${candidate.quote}`;
        const apply = document.createElement("button");
        apply.type = "button"; apply.textContent = "采用到表单";
        apply.addEventListener("click", () => {
          if (busy || revision !== requestRevision) return;
          const input = form.elements.namedItem(candidate.field);
          if (input.value.trim() && input.value !== candidate.value
            && !window.confirm(`用候选内容替换“${fieldLabels[candidate.field]}”当前编辑值？`)) return;
          input.value = candidate.value;
          message.textContent = "已填入表单，尚未保存。请核对后点击保存；其他字段保持不变。";
          row.remove();
        });
        const dismiss = document.createElement("button");
        dismiss.type = "button"; dismiss.textContent = "忽略";
        dismiss.addEventListener("click", () => row.remove());
        row.append(heading, value, previous, quote, apply, dismiss);
        candidateBox.append(row);
      }
      message.textContent = data.candidates.length
        ? `提取到 ${data.candidates.length} 项候选，请逐项核对并采用，最后点击保存。`
        : "未发现明确的新画像信息，或内容与已保存画像相同。";
    } catch (error) {
      if (error.status === 409) revision = null;
      message.textContent = error.name === "AbortError"
        ? "提取等待超时，本次不会保存画像；后台请求可能仍在完成，请稍后再试。"
        : error.message;
    } finally { busy = false; controls(); }
  });
  document.getElementById("profile-open").addEventListener("click", () => {
    if (!dialog.open) dialog.showModal();
    load();
  });
  document.getElementById("profile-close").addEventListener("click", () => dialog.close());
  document.getElementById("profile-reload").addEventListener("click", load);
  dialog.addEventListener("cancel", (event) => { if (busy) event.preventDefault(); });
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (revision === null || busy) return;
    const values = Object.fromEntries(fields.map((name) => [name, form.elements.namedItem(name).value]));
    const body = {fields: values, enabled: enabled.checked, expected_revision: revision,
      confirmed_fields: [...confirmed].filter((name) => values[name].trim())};
    perform(() => request("/profile", body), "已保存，将用于后续知识问答和聊天；正在生成的回答可能仍使用原来的偏好。");
  });
  document.getElementById("profile-clear").addEventListener("click", () => {
    if (revision === null || busy || !window.confirm("清空全部学习偏好并关闭画像注入？聊天记录和历史摘要会保留。")) return;
    const body = {expected_revision: revision};
    perform(() => request("/profile/clear", body), "画像已清空并关闭。聊天历史与摘要仍保留。");
  });
})();
