<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JuprisMini-NCM AI Chat Dashboard</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        dark: {
                            900: '#0f172a',
                            800: '#1e293b',
                            700: '#334155',
                        }
                    }
                }
            }
        }
    </script>
</head>
<body class="bg-dark-900 text-slate-100 h-screen flex flex-col justify-between overflow-hidden">

    <!-- HEADER -->
    <header class="bg-dark-800 border-b border-dark-700 px-6 py-4 flex items-center justify-between shadow-md">
        <div class="flex items-center space-x-3">
            <div class="w-3 h-3 bg-emerald-500 rounded-full animate-pulse"></div>
            <h1 class="font-bold text-lg tracking-wide">JuprisMini-NCM <span class="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full border border-emerald-500/30">Prod Ready</span></h1>
        </div>
        <div id="health-status" class="text-xs text-slate-400 bg-dark-900 px-3 py-1.5 rounded-lg border border-dark-700">
            Checking backend node (:8893)...
        </div>
    </header>

    <!-- CHAT CONTAINER -->
    <main id="chat-container" class="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6 max-w-4xl w-full mx-auto">
        <!-- Welcome Message -->
        <div class="flex items-start space-x-4">
            <div class="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center font-bold text-white shadow-lg shrink-0">AI</div>
            <div class="bg-dark-800 border border-dark-700 rounded-2xl p-4 max-w-xl text-slate-200 shadow-sm">
                <p class="leading-relaxed">Halo bosku! JuprisMini-NCM V6 active di port <code class="bg-dark-900 px-1.5 py-0.5 rounded text-indigo-400">8893</code>. Ada yang bisa dibantu hari ini?</p>
                <div class="mt-3 flex flex-wrap gap-2">
                    <button onclick="quickPrompt('kopi rokok wis')" class="text-xs bg-dark-900 hover:bg-dark-700 text-slate-300 px-3 py-1.5 rounded-lg border border-dark-700 transition">kopi rokok wis</button>
                    <button onclick="quickPrompt('buka browser')" class="text-xs bg-dark-900 hover:bg-dark-700 text-slate-300 px-3 py-1.5 rounded-lg border border-dark-700 transition">buka browser</button>
                    <button onclick="quickPrompt('nyalakan mesin')" class="text-xs bg-dark-900 hover:bg-dark-700 text-slate-300 px-3 py-1.5 rounded-lg border border-dark-700 transition">nyalakan mesin</button>
                </div>
            </div>
        </div>
    </main>

    <!-- INPUT AREA -->
    <footer class="bg-dark-800 border-t border-dark-700 p-4 sm:p-6">
        <form id="chat-form" onsubmit="sendMessage(event)" class="max-w-4xl mx-auto flex items-center gap-3">
            <input 
                type="text" 
                id="user-input" 
                placeholder="Ketik pesan atau perintah (misal: piye kabare, food in indonesian)..." 
                autocomplete="off"
                class="flex-1 bg-dark-900 border border-dark-700 rounded-xl px-4 py-3.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition shadow-inner"
                required
            >
            <button 
                type="submit" 
                id="send-btn"
                class="bg-indigo-600 hover:bg-indigo-500 text-white font-medium px-6 py-3.5 rounded-xl transition flex items-center justify-center shrink-0 shadow-lg shadow-indigo-600/20">
                Kirim
            </button>
        </form>
    </footer>

    <!-- SCRIPT CHAT LOGIC -->
    <script>
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const healthStatus = document.getElementById('health-status');

        // Cek health backend saat load
       async function checkHealth() {
            try {
                // Lewatkan melalui proxy PHP agar aman dari Mixed Content HTTPS
                let res = await fetch('api_proxy.php?action=health');
                let data = await res.json();
                healthStatus.innerHTML = `Port: <span class="text-emerald-400 font-semibold">${data.port || 8893}</span> | Vocab: <span class="text-indigo-400 font-semibold">${data.vocab_size}</span>`;
            } catch (e) {
                healthStatus.innerHTML = `<span class="text-rose-400 font-semibold">Backend Offline</span>`;
            }
        }
        checkHealth();

        function quickPrompt(text) {
            userInput.value = text;
            document.getElementById('chat-form').dispatchEvent(new Event('submit'));
        }

        async function sendMessage(e) {
            e.preventDefault();
            const text = userInput.value.trim();
            if (!text) return;

            // Append User Message
            appendMessage(text, 'user');
            userInput.value = '';
            userInput.disabled = true;
            sendBtn.disabled = true;

            // Append Typing Indicator
            const typingId = appendTypingIndicator();

            try {
                let res = await fetch('api_proxy.php', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: text })
                });
                let data = await res.json();
                
                removeTypingIndicator(typingId);

                if (data.status && data.status === 'error') {
                    appendMessage('Error: ' + data.message, 'ai', null, true);
                } else {
                    appendMessage(data.reply, 'ai', data);
                }
            } catch (err) {
                removeTypingIndicator(typingId);
                appendMessage('Gagal terhubung ke server proxy PHP.', 'ai', null, true);
            } finally {
                userInput.disabled = false;
                sendBtn.disabled = false;
                userInput.focus();
            }
        }

        function appendMessage(text, sender, meta = null, isError = false) {
            const isUser = sender === 'user';
            const wrapper = document.createElement('div');
            wrapper.className = `flex items-start space-x-4 ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`;

            let metaBadge = '';
            if (meta && !isUser) {
                const badgeColor = meta.route_type === 'TOOL_CALL' ? 'bg-amber-500/20 text-amber-400 border-amber-500/30' : 'bg-indigo-500/20 text-indigo-400 border-indigo-500/30';
                metaBadge = `
                    <div class="mt-2.5 pt-2 border-t border-dark-700 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
                        <span class="px-2 py-0.5 rounded border ${badgeColor}">${meta.route_type}</span>
                        <span>CID: <b>${meta.concept_id}</b></span>
                        ${meta.tool_name ? `<span>Tool: <b class="text-slate-200">${meta.tool_name}</b></span>` : ''}
                        <span>Conf: <b>${meta.confidence}</b></span>
                        <span class="ml-auto text-emerald-400">${meta.latency_ms} ms</span>
                    </div>
                `;
            }

            wrapper.innerHTML = `
                <div class="w-9 h-9 rounded-xl ${isUser ? 'bg-slate-700' : 'bg-indigo-600'} flex items-center justify-center font-bold text-white shadow-lg shrink-0">${isUser ? 'U' : 'AI'}</div>
                <div class="bg-dark-800 border ${isError ? 'border-rose-500/50 text-rose-300' : 'border-dark-700 text-slate-200'} rounded-2xl p-4 max-w-xl shadow-sm">
                    <p class="leading-relaxed whitespace-pre-wrap">${escapeHtml(text)}</p>
                    ${metaBadge}
                </div>
            `;
            chatContainer.appendChild(wrapper);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function appendTypingIndicator() {
            const id = 'typing-' + Date.now();
            const wrapper = document.createElement('div');
            wrapper.id = id;
            wrapper.className = `flex items-start space-x-4`;
            wrapper.innerHTML = `
                <div class="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center font-bold text-white shadow-lg shrink-0">AI</div>
                <div class="bg-dark-800 border border-dark-700 rounded-2xl p-4 shadow-sm flex items-center space-x-2">
                    <div class="w-2 h-2 bg-indigo-500 rounded-full animate-bounce"></div>
                    <div class="w-2 h-2 bg-indigo-500 rounded-full animate-bounce [animation-delay:0.2s]"></div>
                    <div class="w-2 h-2 bg-indigo-500 rounded-full animate-bounce [animation-delay:0.4s]"></div>
                </div>
            `;
            chatContainer.appendChild(wrapper);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return id;
        }

        function removeTypingIndicator(id) {
            const el = document.getElementById(id);
            if (el) el.remove();
        }

        function escapeHtml(text) {
            const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
            return text.replace(/[&<>"']/g, m => map[m]);
        }
    </script>
</body>
</html>