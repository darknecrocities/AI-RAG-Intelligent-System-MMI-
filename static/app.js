// MMI Knowledge Assistant Frontend Logic

// Set this to your backend Render URL (e.g. "https://mmi-rag-backend.onrender.com") if hosting the frontend on Vercel.
// Otherwise, keep it empty "" to use the same host.
const API_BASE_URL = "https://ai-rag-intelligent-system-mmi.onrender.com";

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const badgeOllama = document.getElementById("badge-ollama");
  const badgeRedis = document.getElementById("badge-redis");
  const badgeVectors = document.getElementById("badge-vectors");
  
  const checkForceCrawl = document.getElementById("check-force-crawl");
  const checkClearIndex = document.getElementById("check-clear-index");
  const btnIngest = document.getElementById("btn-ingest");
  const ingestStatusText = document.getElementById("ingest-status-text");
  const ingestProgressContainer = document.getElementById("ingest-progress-container");
  const ingestProgressFill = document.getElementById("ingest-progress-fill");
  const statPages = document.getElementById("stat-pages");
  const statChunks = document.getElementById("stat-chunks");
  const statLastRun = document.getElementById("stat-last-run");
  
  const checkStream = document.getElementById("check-stream");
  const btnClearChat = document.getElementById("btn-clear-chat");
  const chatMessagesContainer = document.getElementById("chat-messages-container");
  const chatForm = document.getElementById("chat-form");
  const chatInput = document.getElementById("chat-input");
  const btnSend = document.getElementById("btn-send");
  
  const citationsContainer = document.getElementById("citations-container");
  const queryChips = document.querySelectorAll(".query-chip");

  // Ingestion polling state
  let isPollingIngestion = false;

  // Conversation History
  const chatHistory = [];

  // Simple Markdown Parser
  function parseMarkdown(text) {
    if (!text) return "";
    
    let html = text;
    
    // Protect raw backticks / code blocks
    const codeBlocks = [];
    html = html.replace(/```([\s\S]*?)```/g, (match, code) => {
      codeBlocks.push(code.trim());
      return `__CODE_BLOCK_${codeBlocks.length - 1}__`;
    });

    // Replace headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Replace bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Replace inline code
    html = html.replace(/`(.*?)`/g, '<code>$1</code>');

    // Replace bullet points
    html = html.replace(/^\s*[\-\*]\s+(.*)$/gim, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gim, '<ul>$1<\/ul>');
    // Clean nested ul tags
    html = html.replace(/<\/ul>\s*<ul>/g, '');

    // Replace paragraphs (newlines)
    html = html.replace(/\n\n/g, '<br><br>');

    // Restore code blocks
    codeBlocks.forEach((code, idx) => {
      html = html.replace(`__CODE_BLOCK_${idx}__`, `<pre><code>${escapeHtml(code)}</code></pre>`);
    });

    return html;
  }

  function escapeHtml(text) {
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // System Health Checker
  async function checkSystemHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) throw new Error("Health check returned error");
      const health = await response.json();
      
      // Update HF Space badge
      if (health.ollama_connected) {
        badgeOllama.className = "health-badge ok";
        badgeOllama.querySelector(".badge-label").textContent = "HF Space: Connected";
      } else {
        badgeOllama.className = "health-badge error";
        badgeOllama.querySelector(".badge-label").textContent = "HF Space: Disconnected";
      }
      
      // Update Redis badge
      if (health.redis_connected) {
        badgeRedis.className = "health-badge ok";
        badgeRedis.querySelector(".badge-label").textContent = "Cache: Redis (Online)";
      } else {
        badgeRedis.className = "health-badge warning";
        badgeRedis.querySelector(".badge-label").textContent = "Cache: In-Memory";
      }
      
      // Update Chunks count badge
      badgeVectors.className = "health-badge ok";
      badgeVectors.querySelector(".badge-label").textContent = `Chunks: ${health.vector_store_chunks}`;
      
    } catch (error) {
      console.error("System health check failed:", error);
      badgeOllama.className = "health-badge error";
      badgeOllama.querySelector(".badge-label").textContent = "HF Space: Offline";
      badgeRedis.className = "health-badge error";
      badgeRedis.querySelector(".badge-label").textContent = "Cache: Error";
    }
  }

  // Poll Ingestion Status
  async function pollIngestionStatus() {
    try {
      const response = await fetch(`${API_BASE_URL}/ingest/status`);
      const status = await response.json();
      
      // Update stats and progress bar
      statPages.textContent = status.pages_processed;
      statChunks.textContent = status.chunks_added;
      
      // Set status label classes
      ingestStatusText.textContent = status.status.toUpperCase();
      ingestStatusText.className = `status-val ${status.status}`;
      
      if (status.status === "ingesting") {
        isPollingIngestion = true;
        btnIngest.disabled = true;
        btnIngest.querySelector(".btn-text").textContent = "Ingesting Knowledge...";
        btnIngest.querySelector(".btn-spinner").classList.remove("hidden");
        ingestProgressContainer.classList.remove("hidden");
        
        // Loop animated progress bar fill
        ingestProgressFill.style.width = "40%";
        // Fallback progress bar indicator updates based on pages count
        const calculatedProgress = Math.min((status.pages_processed / 15) * 100, 95);
        if (calculatedProgress > 0) {
          ingestProgressFill.style.width = `${calculatedProgress}%`;
        }
        
        // Schedule next poll
        setTimeout(pollIngestionStatus, 2000);
      } else {
        // Complete or idle
        isPollingIngestion = false;
        btnIngest.disabled = false;
        btnIngest.querySelector(".btn-text").textContent = "Start Crawl & Ingestion";
        btnIngest.querySelector(".btn-spinner").classList.add("hidden");
        ingestProgressContainer.classList.add("hidden");
        
        if (status.status === "completed") {
          ingestStatusText.textContent = "COMPLETED";
          statLastRun.textContent = `Last ingestion: ${status.last_ingested}`;
          checkSystemHealth(); // Refresh chunks badge
        } else if (status.status === "failed") {
          ingestStatusText.textContent = "FAILED";
          alert(`Ingestion job failed: ${status.last_error}`);
        }
      }
    } catch (error) {
      console.error("Failed to fetch ingestion status:", error);
    }
  }

  // Handle Ingest Button Click
  btnIngest.addEventListener("click", async () => {
    if (isPollingIngestion) return;
    
    const body = {
      force_recrawl: checkForceCrawl.checked,
      clear_index: checkClearIndex.checked
    };
    
    try {
      const response = await fetch(`${API_BASE_URL}/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const data = await response.json();
      
      if (data.status === "started") {
        pollIngestionStatus();
      } else {
        alert(`Failed to start ingestion: ${data.message}`);
      }
    } catch (error) {
      console.error("Error triggering ingestion:", error);
      alert("Error contacting server to trigger ingestion.");
    }
  });

  // Render citation cards in the right sidebar panel
  function renderCitations(sources) {
    if (!sources || sources.length === 0) {
      citationsContainer.innerHTML = `
        <div class="empty-citations-state">
          <div class="empty-citations-icon">🔍</div>
          <p>Not found in knowledge base or no grounded sources available.</p>
        </div>
      `;
      return;
    }

    citationsContainer.innerHTML = "";
    sources.forEach(src => {
      const card = document.createElement("div");
      card.className = "citation-card";
      
      // Calculate display percentage score (e.g. 0.85 -> 85%)
      const scorePercent = Math.round(src.score * 100);
      
      card.innerHTML = `
        <div class="citation-header">
          <div class="citation-title-wrap">
            <a href="${src.url}" target="_blank" class="citation-title">${src.title || "Page Context"}</a>
            <span class="citation-section">${src.section || "General Information"}</span>
          </div>
          <span class="citation-score">${scorePercent}% Match</span>
        </div>
        <div class="citation-snippet">
          "${src.snippet}"
        </div>
        <a href="${src.url}" target="_blank" class="citation-link">
          Open Source page ↗
        </a>
      `;
      citationsContainer.appendChild(card);
    });
  }

  // Auto scroll chat list
  function scrollChatToBottom() {
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
  }

  // Handle Query Submission (Chat)
  async function submitQuery(query) {
    if (!query.trim()) return;

    // Add User Message Card
    const userMsg = document.createElement("div");
    userMsg.className = "message user";
    userMsg.innerHTML = `
      <div class="message-avatar">👤</div>
      <div class="message-content">
        <p>${escapeHtml(query)}</p>
      </div>
    `;
    chatMessagesContainer.appendChild(userMsg);
    scrollChatToBottom();

    // Add Assistant Card with Loader
    const assistantMsg = document.createElement("div");
    assistantMsg.className = "message assistant";
    assistantMsg.innerHTML = `
      <div class="message-avatar">🤖</div>
      <div class="message-content">
        <div class="typing-loader">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
      </div>
    `;
    chatMessagesContainer.appendChild(assistantMsg);
    scrollChatToBottom();

    const assistantContent = assistantMsg.querySelector(".message-content");
    const streamResponse = checkStream.checked;

    if (streamResponse) {
      // SSE Streaming Mode
      try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            user_query: query, 
            history: chatHistory.slice(-8), 
            stream: true 
          })
        });

        if (!response.ok) throw new Error("Chat call failed");
        
        // Remove loader and prepare token collector
        assistantContent.innerHTML = "";
        let collectedText = "";
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n\n");
          buffer = lines.pop(); // Keep partial line in buffer

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const dataStr = line.slice(6).strip || line.slice(6);
              if (dataStr === "[DONE]") continue;
              
              try {
                const data = JSON.parse(dataStr);
                if (data.token) {
                  collectedText += data.token;
                  
                  // Use requestAnimationFrame for smoother DOM updates and parsing
                  if (!window.pendingRender) {
                    window.pendingRender = true;
                    requestAnimationFrame(() => {
                      assistantContent.innerHTML = parseMarkdown(collectedText);
                      scrollChatToBottom();
                      window.pendingRender = false;
                    });
                  }
                } else if (data.sources) {
                  renderCitations(data.sources);
                } else if (data.error) {
                  assistantContent.innerHTML = `<span style="color: var(--error)">Error: ${escapeHtml(data.error)}</span>`;
                }
              } catch (e) {
                // Ignore chunk split parsing errors
              }
            }
          }
        }
        
        // Save to chat history
        chatHistory.push({ role: "user", content: query });
        chatHistory.push({ role: "assistant", content: collectedText });

      } catch (error) {
        console.error("Streaming error:", error);
        assistantContent.innerHTML = `<span style="color: var(--error)">Failed to stream response from the server. Verify Ollama status.</span>`;
      }
    } else {
      // Standard Synchronous Mode
      try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            user_query: query, 
            history: chatHistory.slice(-8), 
            stream: false 
          })
        });
        
        if (!response.ok) throw new Error("Chat call failed");
        const data = await response.json();
        
        assistantContent.innerHTML = parseMarkdown(data.answer);
        renderCitations(data.sources);
        scrollChatToBottom();
        
        // Save to chat history
        chatHistory.push({ role: "user", content: query });
        chatHistory.push({ role: "assistant", content: data.answer });

      } catch (error) {
        console.error("Chat error:", error);
        assistantContent.innerHTML = `<span style="color: var(--error)">Failed to fetch response. Make sure the local LLM is responsive.</span>`;
      }
    }
  }

  // Chat Form Submission
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = chatInput.value;
    chatInput.value = "";
    submitQuery(query);
  });

  // Handle Textarea height resizing and Shift+Enter key binding
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // Suggestion query chips click
  queryChips.forEach(chip => {
    chip.addEventListener("click", () => {
      submitQuery(chip.textContent);
    });
  });

  // Clear Chat button click
  btnClearChat.addEventListener("click", () => {
    chatHistory.length = 0; // Clear conversation history
    chatMessagesContainer.innerHTML = `
      <div class="message assistant welcome">
        <div class="message-avatar">🤖</div>
        <div class="message-content">
          <h3>Welcome to the MMI Knowledge Assistant</h3>
          <p>I can answer questions regarding the business, philosophy, technology divisions, and history of <strong>株式会社マン・マシンインターフェース (Man Machine Interface)</strong>.</p>
          <p>To begin, ensure you have ran the crawler to ingest content from <a href="https://www.mmi-sc.co.jp/" target="_blank">mmi-sc.co.jp</a>, or simply type your questions below!</p>
        </div>
      </div>
    `;
    citationsContainer.innerHTML = `
      <div class="empty-citations-state">
        <div class="empty-citations-icon">🔍</div>
        <p>Ask a question to see the retrieved ground-truth sources and relevance scores here.</p>
      </div>
    `;
  });

  // Initialize and run health checks
  checkSystemHealth();
  pollIngestionStatus();
  setInterval(checkSystemHealth, 8000);
});
