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
            path: location.pathname,
            query: location.search || '',
            fragment: location.hash || ''
          })
        );
      } catch {}
      try { ws.send(JSON.stringify({t:'subscribe', channels:['ui','logs','chat']})); } catch {}
      clearInterval(heartbeat);
      heartbeat = setInterval(() => {
        try { if (isOpen()) ws.send(JSON.stringify({t:'ping', ts: Date.now()})); } catch {}
      }, 25025);
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
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
          const messageElement = createChatMessage(msg.data);
          chronologicalOutput.appendChild(messageElement);
          scrollToBottom(chatContainer || chronologicalOutput);
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


