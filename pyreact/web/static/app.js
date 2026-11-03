(function(){
  const proto = (location.protocol === 'https:') ? 'wss://' : 'ws://';
  const cli = document.getElementById('cli');
  const chronologicalOutput = document.getElementById('chronological-output');
  const chatContainer = document.getElementById('chat-container');
  const dbgTreeBtn = document.getElementById('dbg-tree');
  const dbgTraceBtn = document.getElementById('dbg-trace');

  let ws;
  let retries = 0;
  const maxDelay = 10000; // 10s cap
  let heartbeat;

  // Session id: per tab (sessionStorage), survives reloads so the server can
  // resume this tab's session; a new tab gets a new session.
  const SESSION_KEY = 'pyreact.session';
  function loadSessionId() { try { return sessionStorage.getItem(SESSION_KEY); } catch { return null; } }
  function saveSessionId(id) { try { sessionStorage.setItem(SESSION_KEY, id); } catch {} }

  // Newest event id this page has shown. Sent in `hello` so a reconnect only
  // replays what we missed; a reload starts from null (full replay of the
  // session's recent events, the page is empty again).
  let lastEventId = null;

  // The same chat message may arrive both as a legacy `chat` message and as
  // an EventBus `message` event; render it once (only cross-channel copies
  // within 2s are treated as duplicates).
  const recentMessages = new Map(); // key -> [{t, src}]
  function isDuplicate(data, src) {
    const key = [data.sender, data.message_type, data.text].join('\u0000');
    const now = Date.now();
    for (const [k, list] of recentMessages) {
      const kept = list.filter((e) => now - e.t <= 2000);
      if (kept.length) recentMessages.set(k, kept); else recentMessages.delete(k);
    }
    const list = recentMessages.get(key) || [];
    const i = list.findIndex((e) => e.src !== src);
    if (i >= 0) { list.splice(i, 1); return true; }
    list.push({ t: now, src });
    recentMessages.set(key, list);
    return false;
  }
  function appendChatMessage(data, src) {
    if (!data || typeof data.text !== 'string') return;
    data = Object.assign({ sender: 'assistant', message_type: 'chat' }, data);
    if (isDuplicate(data, src)) return;
    chronologicalOutput.appendChild(createChatMessage(data));
    scrollToBottom(chatContainer || chronologicalOutput);
  }

  function scrollToBottom(el){ try{ el.scrollTop = el.scrollHeight; }catch{} }
  function isOpen() { return ws && ws.readyState === WebSocket.OPEN; }
  
  function createChatMessage(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${data.sender}`;
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = `message-bubble ${data.message_type}`;
    
    const senderDiv = document.createElement('div');
    senderDiv.className = 'message-sender';
    senderDiv.textContent = data.sender.charAt(0).toUpperCase() + data.sender.slice(1);
    
    const textDiv = document.createElement('div');
    textDiv.textContent = data.text;
    
    bubbleDiv.appendChild(senderDiv);
    bubbleDiv.appendChild(textDiv);
    messageDiv.appendChild(bubbleDiv);
    
    return messageDiv;
  }
  
  function createLogEntry(html) {
    // Skip empty log entries
    if (!html || html.trim() === '') {
      return null;
    }
    const logDiv = document.createElement('div');
    logDiv.className = 'log-entry';
    
    // Preserve whitespace and line breaks
    logDiv.style.whiteSpace = 'pre-wrap';
    logDiv.style.fontFamily = 'monospace';
    logDiv.innerHTML = html;
    return logDiv;
  }

  function connect() {
    ws = new WebSocket(proto + location.host + '/ws');

    ws.onopen = () => {
      retries = 0;
      try {
        ws.send(
          JSON.stringify({
            t: 'hello',
            session: loadSessionId(),
            last_event: lastEventId,
            path: location.pathname,
            query: location.search || '',
            fragment: location.hash || ''
          })
        );
      } catch {}
      clearInterval(heartbeat);
      heartbeat = setInterval(() => {
        try { if (isOpen()) ws.send(JSON.stringify({t:'ping', ts: Date.now()})); } catch {}
      }, 25025);
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.channel === 'session' && msg.type === 'session') {
          if (msg.data && msg.data.id) saveSessionId(msg.data.id);
          return;
        }
        if (msg.channel === 'events' && msg.event) {
          const e = msg.event;
          if (typeof e.id === 'number' && (lastEventId === null || e.id > lastEventId)) lastEventId = e.id;
          if (e.kind === 'message') appendChatMessage(e.payload || {}, 'events');
          return;
        }
        if (msg.channel === 'ui' && msg.type === 'nav') {
          if (location.pathname !== msg.data) history.pushState({}, '', msg.data);
          return;
        }
        if (msg.channel === 'logs' && msg.type === 'stdout') {
          const logElement = createLogEntry(msg.data);
          if (logElement) {
            chronologicalOutput.appendChild(logElement);
            scrollToBottom(chatContainer || chronologicalOutput);
          }
          return;
        }
        if (msg.channel === 'chat' && msg.type === 'message') {
          appendChatMessage(msg.data, 'chat');
          return;
        }
      } catch {
        document.getElementById('root').innerHTML = ev.data;
      }
    };

    ws.onclose = () => {
      clearInterval(heartbeat);
      retries += 1;
      const delay = Math.min(1000 * Math.pow(2, retries - 1), maxDelay) + Math.random() * 500;
      setTimeout(connect, delay);
    };

    ws.onerror = () => {
      try { ws.close(); } catch {}
    };
  }

  connect();

  // Back/forward → server
  window.addEventListener('popstate', () => {
    try {
      if (isOpen())
        ws.send(
          JSON.stringify({
            t: 'nav',
            path: location.pathname,
            query: location.search || '',
            fragment: location.hash || ''
          })
        );
    } catch {}
  });

  cli.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const userInput = cli.value.trim();
      if (userInput) {
        // Send to server - the server will create the user message bubble
        try { if (isOpen()) ws.send(JSON.stringify({t:'submit', v: userInput})); } catch {}
      }
      cli.value = '';
      cli.focus();
      scrollToBottom(chatContainer || chronologicalOutput);
    }
  });

  // Debug: print VNode tree
  if (dbgTreeBtn) {
    dbgTreeBtn.addEventListener('click', () => {
      try { if (isOpen()) ws.send(JSON.stringify({t:'debug', what:'tree'})); } catch {}
    });
  }
  if (dbgTraceBtn) {
    dbgTraceBtn.addEventListener('click', () => {
      try { if (isOpen()) ws.send(JSON.stringify({t:'debug', what:'trace'})); } catch {}
    });
  }
})();


