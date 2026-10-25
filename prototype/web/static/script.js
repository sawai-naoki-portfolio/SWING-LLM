/* =========================================================================
   SWI Web UI — application script

   バックエンドの API 契約は一切変更していない（モバイルアプリと同じものを話す）:
     POST /api/generate_stream   {prompt, use_cot, images, persona, client, history} -> text/plain 逐次
     POST /api/generate          {..., best_of_n} -> {response}
     POST /api/feedback          {query, response, rating}
     POST /api/generate_image    {prompt} -> PNG
     POST /api/tts | /api/tts_stream  {text, lang, ref_audio?, ref_speaker_id?}
     GET  /api/voice/status | /api/image/status | /api/llm/status

   ストリームの読み方も従来どおり（docs/03）:
     - text/plain をそのまま連結して描画する（SSE ではない）
     - "Error: " 接頭辞 / "[ERROR]" はインライン失敗
     - <tool_call> / <tool_result> / <think> と "grounded": true|false は表示用マークアップ
     - サーバ注入の <div class="tool-box success"> HTML と
       "⚙️/⚠️ [システム: ...]" プレーンテキストのツール通知は同じ見た目のチップに畳む
   ========================================================================= */
(() => {
'use strict';

/* ═══════════════ constants ═══════════════ */
const MAX_PROMPT = 4000;
const MAX_HISTORY_MESSAGES = 16;   // サーバ側 max_turns とは別に、送信ペイロードの肥大を抑える
const MAX_CHATS = 40;
const LS = {
    theme: 'swi.theme', persona: 'swi.persona', modes: 'swi.modes',
    ttsEngine: 'swi.ttsEngine', bonN: 'swi.bonN',
    chats: 'swi.chats.v1', activeChat: 'swi.activeChat.v1',
};

/* ═══════════════ element refs ═══════════════ */
const $ = (id) => document.getElementById(id);
const root = document.documentElement;
const app = $('app');
const messagesEl = $('messages');
const threadEl = $('thread');
const heroEl = $('hero');
const promptEl = $('prompt');
const sendBtn = $('send');
const stopBtn = $('stop');
const charCount = $('char-count');
const scrollFab = $('scroll-fab');

const chatListEl = $('chat-list');
const chatEmptyEl = $('chat-empty');

const personaSel = $('persona');
const personaTag = $('persona-tag');
const bonNSel = $('bon-n');
const ttsEngineSel = $('tts-engine');

const modelChip = $('model-chip');
const modelName = $('model-name');
const modelTag = $('model-tag');
const backendPill = $('backend-pill');
const backendDesc = $('backend-desc');

const attachBtn = $('attach');
const fileInput = $('file-input');
const attachmentsEl = $('attachments');
const dropzone = $('dropzone');
const dropOverlay = $('drop-overlay');
const micBtn = $('mic');
const toastsEl = $('toasts');

const voiceRefBtn = $('voice-ref');
const voiceRefInput = $('voice-ref-input');
const voiceRefClear = $('voice-ref-clear');
const refAudioName = $('ref-audio-name');
const refSpeakerId = $('ref-speaker-id');
const rowVoiceRef = $('row-voice-ref');

const sheet = $('settings-sheet');

/* ═══════════════ state ═══════════════ */
let active = { id: null, messages: [] };   // messages: [{role, content, images}]
let attachedImages = [];                   // [dataURL]（送信時のみ・履歴には保存しない）
let voiceRefAudioB64 = null;
let controller = null;                     // 進行中ストリームの AbortController
// 送信を開始した時点のチャット ID。ストリーム中に別チャットへ切り替えられたとき、
// 返ってきた応答を「今開いているチャット」へ混入させないための照合キー。
let turnChatId = null;
let currentTtsAudio = null;
let lastTurn = null;                       // 再生成用 {text, images}

const modes = { cot: true, bon: false, imagegen: false, tts: false };
// サーバが「実際に使える」と申告した機能（/api/llm/status）。取得までは楽観的に全部 true。
let caps = { vision: true, cot: true, best_of_n: true, tools: true };
let engineUp = { voice: false, image: false };

/* ═══════════════ localStorage helpers ═══════════════ */
function lsGet(key, fallback) {
    try {
        const v = localStorage.getItem(key);
        return v === null ? fallback : JSON.parse(v);
    } catch { return fallback; }
}
function lsSet(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* quota / privacy mode */ }
}

/* ═══════════════ chat store（ブラウザ内のみ・サーバへは送らない） ═══════════════ */
function loadChats() {
    const chats = lsGet(LS.chats, []);
    return Array.isArray(chats) ? chats : [];
}
function saveChats(chats) { lsSet(LS.chats, chats.slice(0, MAX_CHATS)); }

function persistActive() {
    if (!active.id || active.messages.length === 0) return;
    const chats = loadChats().filter((c) => c.id !== active.id);
    const firstUser = active.messages.find((m) => m.role === 'user');
    chats.unshift({
        id: active.id,
        title: (firstUser ? firstUser.content : '新しいチャット').slice(0, 60) || '画像の質問',
        updatedAt: Date.now(),
        // 画像バイト列は保存しない（1 枚 ~70KB で localStorage(5MB) をすぐ枯らすため）。
        // 枚数だけ残し、復元時は「🖼️ 画像 N 枚」と表示する。
        messages: active.messages.map((m) => ({
            role: m.role, content: m.content, images: m.images || 0,
        })),
    });
    saveChats(chats);
    lsSet(LS.activeChat, active.id);
    renderChatList();
}

function renderChatList() {
    const chats = loadChats();
    chatListEl.innerHTML = '';
    chatEmptyEl.hidden = chats.length > 0;
    chats.forEach((c) => {
        const li = document.createElement('li');
        li.className = 'chat' + (c.id === active.id ? ' is-active' : '');
        const title = document.createElement('span');
        title.className = 'chat__title';
        title.textContent = c.title;
        const del = document.createElement('button');
        del.className = 'chat__del';
        del.title = 'このチャットを削除';
        del.setAttribute('aria-label', 'このチャットを削除');
        del.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
        del.addEventListener('click', (e) => { e.stopPropagation(); deleteChat(c.id); });
        li.append(title, del);
        li.addEventListener('click', () => openChat(c.id));
        chatListEl.appendChild(li);
    });
}

function openChat(id) {
    const chat = loadChats().find((c) => c.id === id);
    if (!chat) return;
    abortStream();
    active = { id: chat.id, messages: chat.messages.map((m) => ({ ...m })) };
    lsSet(LS.activeChat, id);
    messagesEl.innerHTML = '';
    heroEl.hidden = true;
    active.messages.forEach((m, i) => {
        if (m.role === 'user') {
            renderUserMessage(m.content, m.images || 0);
        } else if (m.role === 'assistant') {
            const el = appendBotShell();
            renderBot(el, m.content, true, true);
            const query = (active.messages[i - 1] || {}).content || '';
            addActions(el, query, () => m.content);
        } else {
            appendSystem(m.content);
        }
    });
    // 復元後も ↻ 再生成が効くように、直近のユーザー発話を復元する
    // （添付画像は保存していないため、再生成はテキストのみで行う）。
    const lastUser = [...active.messages].reverse().find((m) => m.role === 'user');
    lastTurn = lastUser ? { text: lastUser.content, images: [] } : null;
    closeSidebar();
    renderChatList();
    scrollToBottom(true);
}

function deleteChat(id) {
    saveChats(loadChats().filter((c) => c.id !== id));
    if (active.id === id) newChat();
    else renderChatList();
}

function newChat() {
    abortStream();
    active = { id: null, messages: [] };
    localStorage.removeItem(LS.activeChat);
    messagesEl.innerHTML = '';
    heroEl.hidden = false;
    attachedImages = [];
    renderAttachments();
    lastTurn = null;
    renderChatList();
    closeSidebar();
    promptEl.focus();
}
$('new-chat').addEventListener('click', newChat);

/* ═══════════════ settings / theme / persona / modes ═══════════════ */
function setTheme(theme) {
    root.setAttribute('data-theme', theme);
    lsSet(LS.theme, theme);
    const meta = $('theme-color');
    if (meta) meta.setAttribute('content', theme === 'light' ? '#faf9f7' : '#0d0d0f');
}
$('theme-toggle').addEventListener('click', () => {
    setTheme(root.getAttribute('data-theme') === 'light' ? 'dark' : 'light');
});

function applyPersona(p) {
    root.setAttribute('data-persona', p);
    personaTag.textContent = p === 'navi' ? 'NAVI' : 'JARVIS';
}
personaSel.addEventListener('change', () => {
    applyPersona(personaSel.value);
    lsSet(LS.persona, personaSel.value);
});
bonNSel.addEventListener('change', () => lsSet(LS.bonN, bonNSel.value));
ttsEngineSel.addEventListener('change', () => {
    lsSet(LS.ttsEngine, ttsEngineSel.value);
    syncVoiceRefRow();
});

/* mode chips（次の送信の振る舞いを変えるトグル） */
const modeButtons = {
    cot: $('mode-cot'), bon: $('mode-bon'), imagegen: $('mode-imagegen'), tts: $('mode-tts'),
};
Object.entries(modeButtons).forEach(([key, btn]) => {
    btn.addEventListener('click', () => {
        if (btn.disabled) return;
        modes[key] = !modes[key];
        // 高品質(Best-of-N) と 画像生成 は同時に成立しないので排他にする
        if (modes[key] && key === 'bon') modes.imagegen = false;
        if (modes[key] && key === 'imagegen') modes.bon = false;
        lsSet(LS.modes, modes);
        syncModeButtons();
    });
});
function syncModeButtons() {
    const enabled = {
        cot: caps.cot, bon: caps.best_of_n, imagegen: engineUp.image, tts: true,
    };
    const why = {
        cot: '思考連鎖（<think>）は自作モデル（ローカル）専用です',
        bon: 'Best-of-N（PRM 報酬モデル）は自作モデル（ローカル）専用です',
        imagegen: 'SWI-Image が未学習です',
        tts: '',
    };
    Object.entries(modeButtons).forEach(([key, btn]) => {
        const ok = enabled[key];
        btn.disabled = !ok;
        const on = ok && modes[key];
        btn.setAttribute('aria-pressed', on ? 'true' : 'false');
        btn.title = ok ? '' : why[key];
    });
    $('row-bon-n').hidden = !caps.best_of_n;
}

/* ═══════════════ sidebar / sheet ═══════════════ */
$('sidebar-open').addEventListener('click', () => { app.classList.add('is-sidebar-open'); $('scrim').hidden = false; });
function closeSidebar() { app.classList.remove('is-sidebar-open'); $('scrim').hidden = true; }
$('sidebar-close').addEventListener('click', closeSidebar);
$('scrim').addEventListener('click', closeSidebar);

function openSheet() { sheet.hidden = false; }
function closeSheet() { sheet.hidden = true; }
$('open-settings').addEventListener('click', openSheet);
$('open-settings-2').addEventListener('click', () => { closeSidebar(); openSheet(); });
$('settings-close').addEventListener('click', closeSheet);
$('settings-scrim').addEventListener('click', closeSheet);
$('clear-chats').addEventListener('click', () => {
    saveChats([]);
    newChat();
    toast('チャット履歴を削除しました', 'success');
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { if (!sheet.hidden) closeSheet(); else closeSidebar(); }
});

/* ═══════════════ status（バックエンド / エンジン） ═══════════════ */
function setEngine(name, online) {
    const el = document.querySelector(`.engine[data-engine="${name}"]`);
    if (!el) return;
    el.classList.remove('is-checking', 'is-online', 'is-offline');
    el.classList.add(online ? 'is-online' : 'is-offline');
}

async function refreshStatus() {
    // LLM: このページが返っている＝サーバは生きている。どのモデルで動いているかを訊く。
    setEngine('llm', true);
    try {
        const s = await (await fetch('/api/llm/status')).json();
        caps = Object.assign({ vision: true, cot: true, best_of_n: true, tools: true }, s.capabilities || {});
        modelChip.classList.add('is-live');
        modelName.textContent = s.model || 'SWI-LLM';
        modelTag.hidden = false;
        modelTag.textContent = s.backend || 'local';
        modelChip.title = `自作モデル（${s.model}）をこのプロセスで推論しています。`;
        backendPill.textContent = s.backend || 'local';
        backendDesc.textContent = s.model || '';
    } catch {
        modelName.textContent = 'SWI-LLM';
        modelChip.title = 'バックエンド情報を取得できませんでした。';
        backendPill.textContent = '不明';
        backendDesc.textContent = '/api/llm/status に到達できませんでした';
    }
    // サーバが画像理解（VLM）を申告しないときはボタン自体を落とし、
    // 「送れるのに失敗する」体験を作らない。
    attachBtn.disabled = !caps.vision;
    attachBtn.title = caps.vision ? '画像を添付' : '画像理解は自作モデル（ローカル）専用です';

    try {
        engineUp.voice = !!(await (await fetch('/api/voice/status')).json()).available;
    } catch { engineUp.voice = false; }
    setEngine('voice', engineUp.voice);
    const swiOpt = ttsEngineSel.querySelector('option[value="swivoice"]');
    if (swiOpt && !engineUp.voice) {
        swiOpt.textContent = 'SWI-Voice（未学習）';
        swiOpt.disabled = true;
        if (ttsEngineSel.value === 'swivoice') ttsEngineSel.value = 'browser';
    }
    syncVoiceRefRow();

    try {
        engineUp.image = !!(await (await fetch('/api/image/status')).json()).available;
    } catch { engineUp.image = false; }
    setEngine('image', engineUp.image);

    syncModeButtons();
}

/* ═══════════════ composer ═══════════════ */
function autosize() {
    promptEl.style.height = 'auto';
    promptEl.style.height = Math.min(promptEl.scrollHeight, 208) + 'px';
}
function syncComposer() {
    const n = promptEl.value.length;
    charCount.hidden = n < MAX_PROMPT * 0.75;
    charCount.textContent = `${n} / ${MAX_PROMPT}`;
    charCount.classList.toggle('is-warn', n > MAX_PROMPT * 0.95);
    sendBtn.disabled = !!controller || (promptEl.value.trim() === '' && attachedImages.length === 0);
}
promptEl.addEventListener('input', () => { autosize(); syncComposer(); });
promptEl.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    // 日本語入力の変換確定 Enter を送信と誤認しない（IME 変換中は isComposing=true）。
    if (e.isComposing || e.keyCode === 229) return;
    if (e.shiftKey) return;                     // Shift+Enter = 改行
    e.preventDefault();
    handleSend();                               // Enter / Ctrl+Enter = 送信
});
sendBtn.addEventListener('click', handleSend);
stopBtn.addEventListener('click', abortStream);

document.querySelectorAll('.sug').forEach((sug) => sug.addEventListener('click', () => {
    promptEl.value = sug.dataset.prompt;
    autosize(); syncComposer(); promptEl.focus();
}));

/* ═══════════════ image attach (VLM) ═══════════════ */
attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

let dragDepth = 0;
window.addEventListener('dragenter', (e) => {
    if (!hasFiles(e) || !caps.vision) return;
    dragDepth++; dropOverlay.hidden = false; dropzone.classList.add('is-dragover');
});
window.addEventListener('dragover', (e) => { if (hasFiles(e) && caps.vision) e.preventDefault(); });
window.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) { dropOverlay.hidden = true; dropzone.classList.remove('is-dragover'); }
});
window.addEventListener('drop', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth = 0; dropOverlay.hidden = true; dropzone.classList.remove('is-dragover');
    if (!caps.vision) { toast('画像理解は自作モデル（ローカル）専用です', 'error'); return; }
    handleFiles(e.dataTransfer.files);
});
function hasFiles(e) { return e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files'); }

function handleFiles(files) {
    const file = files && files[0];
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            const MAX = 512;                    // 送信前に縮小（サーバ側は 224/AnyRes で再処理）
            let { width, height } = img;
            if (width > height && width > MAX) { height *= MAX / width; width = MAX; }
            else if (height > MAX) { width *= MAX / height; height = MAX; }
            const canvas = document.createElement('canvas');
            canvas.width = width; canvas.height = height;
            canvas.getContext('2d').drawImage(img, 0, 0, width, height);
            attachedImages = [canvas.toDataURL('image/jpeg', 0.85)];
            renderAttachments();
            syncComposer();
        };
        img.onerror = () => toast('画像を読み込めませんでした', 'error');
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}
function renderAttachments() {
    attachmentsEl.innerHTML = '';
    attachmentsEl.hidden = attachedImages.length === 0;
    attachedImages.forEach((src, i) => {
        const div = document.createElement('div');
        div.className = 'thumb';
        div.innerHTML = `<img src="${src}" alt="添付画像"><button class="thumb__rm" aria-label="削除">✕</button>`;
        div.querySelector('.thumb__rm').addEventListener('click', () => {
            attachedImages.splice(i, 1); renderAttachments(); syncComposer();
        });
        attachmentsEl.appendChild(div);
    });
}

/* ═══════════════ speech recognition (STT) ═══════════════ */
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SR();
    recognition.lang = 'ja-JP';
    recognition.interimResults = false;
    recognition.onstart = () => micBtn.classList.add('is-recording');
    recognition.onend = () => micBtn.classList.remove('is-recording');
    recognition.onerror = () => micBtn.classList.remove('is-recording');
    recognition.onresult = (ev) => {
        promptEl.value += ev.results[0][0].transcript;
        autosize(); syncComposer();
    };
    micBtn.addEventListener('click', () => {
        if (micBtn.classList.contains('is-recording')) recognition.stop();
        else { try { recognition.start(); } catch { /* already started */ } }
    });
} else {
    micBtn.hidden = true;
}

/* ═══════════════ voice reference (zero-shot cloning) ═══════════════ */
function syncVoiceRefRow() {
    rowVoiceRef.hidden = !(ttsEngineSel.value === 'swivoice' && engineUp.voice);
    const has = !!voiceRefAudioB64;
    refAudioName.hidden = !has;
    refSpeakerId.hidden = !has;
    voiceRefClear.hidden = !has;
}
voiceRefBtn.addEventListener('click', () => voiceRefInput.click());
voiceRefInput.addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
        voiceRefAudioB64 = reader.result;
        refAudioName.textContent = file.name.length > 20 ? file.name.slice(0, 20) + '…' : file.name;
        syncVoiceRefRow();
    };
    reader.readAsDataURL(file);
});
voiceRefClear.addEventListener('click', () => {
    voiceRefAudioB64 = null;
    voiceRefInput.value = '';
    syncVoiceRefRow();
});

// クローニングは話者ごとの voice_cloning 同意が必須（未登録は 403）。
function withRefAudio(payload) {
    if (voiceRefAudioB64) {
        payload.ref_audio = voiceRefAudioB64;
        payload.ref_speaker_id = (refSpeakerId.value || '').trim();
    }
    return payload;
}
async function warnIfConsentDenied(r) {
    if (r.status !== 403) return;
    let msg = '参照音声の話者に voice_cloning の同意が登録されていません';
    try { msg = (await r.clone().json()).error || msg; } catch { /* body 無し */ }
    console.warn('[SWI-Voice] cloning denied:', msg);
    toast(msg, 'error');
}

/* ═══════════════ rendering ═══════════════ */
function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* core/generation/stream.py が出すツール実行通知。サーバが HTML を注入する場合と
   プレーンテキスト（"⚙️/⚠️ [システム: ...]"）で来る場合がある。
   見た目を揃えるため、後者をここで同じ HTML チップへ畳む。 */
const RE_NOTICE = /\n*(⚙️|⚠️)\s*\[システム:[^\]]*\]\n*/g;
const RE_NOTICE_PARTIAL = /\n*(?:⚙️|⚠️)\s*\[システム:[^\]]*$/;   // チャンク境界で切れた通知
const CHIP_OK = '\n\n<div class="tool-box success"><span class="tool-check">✓</span><span>ツールを実行し、結果を反映しました</span></div>\n\n';
const CHIP_FAIL = '\n\n<div class="tool-box error"><span class="tool-check">!</span><span>ツール呼び出しに失敗しました</span></div>\n\n';
const CHIP_RUNNING = '\n\n<div class="tool-box"><span class="tool-spinner"></span><span>ツールを実行中…</span></div>\n\n';

const BADGE = {
    grounded: '<span class="ground-badge grounded">🔍 根拠あり</span>',
    ungrounded: '<span class="ground-badge ungrounded">⚠️ 根拠不十分</span>',
};
function toolDone(payload) {
    let name = 'Tool';
    try { name = JSON.parse(payload.trim()).name || name; } catch { /* 途中まで */ }
    return `\n\n<div class="tool-box success"><span class="tool-check">✓</span><span>実行完了: ${escapeHtml(String(name))}</span></div>\n\n`;
}
function toolRunning(payload) {
    const m = payload.match(/"name"\s*:\s*"([^"]+)"/);
    return `\n\n<div class="tool-box"><span class="tool-spinner"></span><span>実行中: ${escapeHtml(m ? m[1] : 'Tool')}</span></div>\n\n`;
}
function toolResult(inner) { return `\n\n<div class="tool-result">${escapeHtml(inner.trim())}</div>\n\n`; }
function errorCallout(msg) {
    return `\n\n<div class="callout"><span class="callout__ico">⚠️</span><span>${escapeHtml(msg)}</span></div>\n\n`;
}

/* marked/DOMPurify があればそれで、無ければ最小 markdown で描画する。
   どちらの経路でも「自前の信頼済み HTML（tool-box 等）」は素通しする。 */
/* GFM テーブル。marked が使えないとき（CDN 不通・オフライン）用のフォールバック。
   モデルが表を出したとき、ここが無いと生のパイプ記法が露出する。 */
function splitTableRow(line) {
    return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
}
function isTableDivider(line) {
    if (!line || line.indexOf('-') < 0) return false;
    const cells = splitTableRow(line);
    return cells.length >= 2 && cells.every((c) => /^:?-+:?$/.test(c));
}
function extractTables(src, blocks) {
    const lines = src.split('\n');
    const out = [];
    for (let i = 0; i < lines.length;) {
        const head = lines[i];
        const divider = lines[i + 1];
        if (head && head.indexOf('|') >= 0 && isTableDivider(divider) &&
            splitTableRow(head).length === splitTableRow(divider).length) {
            const cols = splitTableRow(head);
            const rows = [];
            let j = i + 2;
            while (j < lines.length && lines[j].indexOf('|') >= 0 && lines[j].trim() !== '') {
                rows.push(splitTableRow(lines[j]));
                j++;
            }
            const thead = cols.map((c) => `<th>${escapeHtml(c)}</th>`).join('');
            const tbody = rows.map((r) => '<tr>' +
                cols.map((_, k) => `<td>${escapeHtml(r[k] || '')}</td>`).join('') + '</tr>').join('');
            blocks.push(`<table><thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table>`);
            out.push(` CB${blocks.length - 1} `);
            i = j;
        } else {
            out.push(head);
            i++;
        }
    }
    return out.join('\n');
}

function basicMarkdown(src) {
    const blocks = [];
    src = src.replace(/```(\w*)\n?([\s\S]*?)```/g, (m, lang, code) => {
        blocks.push(`<pre><code>${escapeHtml(code.replace(/\n$/, ''))}</code></pre>`);
        return ` CB${blocks.length - 1} `;
    });
    src = extractTables(src, blocks);   // コードブロック退避のあと（表がコード内なら既に保護済み）
    let h = escapeHtml(src);
    h = h.replace(/`([^`]+)`/g, (m, c) => `<code>${c}</code>`);
    h = h.replace(/^\s*###\s+(.*)$/gm, '<h3>$1</h3>')
         .replace(/^\s*##\s+(.*)$/gm, '<h2>$1</h2>')
         .replace(/^\s*#\s+(.*)$/gm, '<h1>$1</h1>');
    h = h.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>').replace(/\*([^*]+)\*/g, '<em>$1</em>');
    h = h.replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    h = h.replace(/^\s*[-*]\s+(.*)$/gm, '<li>$1</li>').replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>');
    h = h.replace(/\n{2,}/g, '</p><p>').replace(/\n/g, '<br>');
    h = `<p>${h}</p>`;
    h = h.replace(/ CB(\d+) /g, (m, i) => blocks[+i]);
    return h;
}
function renderProse(part) {
    if (window.marked && window.DOMPurify) {
        try {
            const parsed = window.marked.parse(part, { breaks: true, gfm: true });
            return window.DOMPurify.sanitize(parsed, { ADD_ATTR: ['target', 'rel', 'class'] });
        } catch { /* fall through */ }
    }
    const safe = [];
    const protect = (re) => {
        part = part.replace(re, (m) => { safe.push(m); return ` SF${safe.length - 1} `; });
    };
    protect(/<div class="tool-box[^"]*">[\s\S]*?<\/div>/g);
    protect(/<div class="tool-result">[\s\S]*?<\/div>/g);
    protect(/<div class="callout">[\s\S]*?<\/div>/g);
    protect(/<span class="ground-badge[^"]*">[\s\S]*?<\/span>/g);
    let html = basicMarkdown(part);
    html = html.replace(/ SF(\d+) /g, (m, i) => safe[+i]);
    return html;
}

function renderBot(el, raw, useCot, final) {
    let s = raw;

    /* 1) インライン失敗（docs/03 の契約）を先に切り出す。
          - ローカル経路の例外: "\n[ERROR] <msg>" が末尾に付く
          - 画像パース失敗等: 本文が "Error: " で始まる */
    let failure = null;
    const iErr = s.indexOf('\n[ERROR] ');
    if (iErr >= 0) {
        failure = s.slice(iErr + '\n[ERROR] '.length).trim();
        s = s.slice(0, iErr);
    }
    if (!failure) {
        const prose = s.replace(RE_NOTICE, '').replace(RE_NOTICE_PARTIAL, '').trim();
        if (/^Error:\s/.test(prose)) {
            failure = prose.replace(/^Error:\s*/, '');
            s = (s.match(RE_NOTICE) || []).join('');   // 通知だけ残し、本文はエラーへ差し替える
        }
    }

    /* 2) ツール通知・ツールマークアップ・根拠バッジ → 表示用チップ */
    s = s.replace(RE_NOTICE, (m, sym) => (sym === '⚠️' ? CHIP_FAIL : CHIP_OK));
    s = s.replace(RE_NOTICE_PARTIAL, CHIP_RUNNING);
    s = s.replace(/"grounded"\s*:\s*true/g, BADGE.grounded)
         .replace(/"grounded"\s*:\s*false/g, BADGE.ungrounded);
    s = s.replace(/<tool_call>([\s\S]*?)<\/tool_call>/g, (m, p) => toolDone(p));
    s = s.replace(/<tool_call>([\s\S]*)$/g, (m, p) => toolRunning(p));
    s = s.replace(/<tool_result>([\s\S]*?)<\/tool_result>/g, (m, p) => toolResult(p));
    s = s.replace(/<tool_result>([\s\S]*)$/g, (m, p) => toolResult(p));

    /* 3) 思考プロセス（<think>）は折りたたみへ */
    let out = '';
    if (useCot && s.includes('<think>')) {
        s.split(/<think>|<\/think>/).forEach((part, i) => {
            if (i % 2 === 0) { if (part.trim()) out += renderProse(part); }
            else out += `<details class="think-box" open><summary class="think-label">思考プロセス</summary>` +
                        `<div class="think-content">${escapeHtml(part.trim())}</div></details>`;
        });
    } else {
        out = renderProse(s.replace(/<\/?think>/g, ''));
    }
    if (failure) out += renderProse(errorCallout(failure));

    el.innerHTML = out;
    if (final) enhance(el);          // hljs/KaTeX/コピーボタンはストリーム中に毎フレーム回さない
    scrollToBottom();
}

function enhance(el) {
    if (window.hljs) {
        el.querySelectorAll('pre code').forEach((b) => { try { window.hljs.highlightElement(b); } catch { /* noop */ } });
    }
    el.querySelectorAll('pre').forEach((pre) => {
        if (pre.querySelector('.code-copy')) return;
        const btn = document.createElement('button');
        btn.className = 'code-copy';
        btn.textContent = 'コピー';
        btn.addEventListener('click', () => {
            const codeEl = pre.querySelector('code');
            navigator.clipboard.writeText((codeEl || pre).innerText).then(() => {
                btn.textContent = '✓ コピー済み';
                setTimeout(() => { btn.textContent = 'コピー'; }, 1400);
            });
        });
        pre.appendChild(btn);
    });
    if (window.renderMathInElement) {
        try {
            window.renderMathInElement(el, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\(', right: '\\)', display: false },
                    { left: '\\[', right: '\\]', display: true },
                ],
                throwOnError: false,
            });
        } catch { /* noop */ }
    }
}

/* ストリーム中は requestAnimationFrame にまとめて再描画する
   （チャンクごとに markdown 全文をパースし直すと長文で目に見えて重くなる）。 */
let rafPending = null;
function scheduleRender(el, raw, useCot) {
    if (rafPending) { rafPending.raw = raw; return; }
    rafPending = { el, raw, useCot };
    requestAnimationFrame(() => {
        const p = rafPending;
        rafPending = null;
        if (!p) return;
        renderBot(p.el, p.raw, p.useCot, false);
        p.el.insertAdjacentHTML('beforeend', '<span class="caret"></span>');
    });
}
function flushRender() { rafPending = null; }

/* ═══════════════ message DOM ═══════════════ */
const MARK_SVG = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>';

function renderUserMessage(text, imageCount, imageSrcs) {
    heroEl.hidden = true;
    const el = document.createElement('div');
    el.className = 'msg msg--user';
    if (imageSrcs && imageSrcs.length) {
        const box = document.createElement('div');
        box.className = 'attachments';
        box.innerHTML = imageSrcs.map((s) => `<div class="thumb"><img src="${s}" alt="添付画像"></div>`).join('');
        el.appendChild(box);
    } else if (imageCount) {
        const note = document.createElement('div');
        note.className = 'imgnote';
        note.textContent = `🖼️ 画像 ${imageCount} 枚（履歴には保存されません）`;
        el.appendChild(note);
    }
    const body = document.createElement('div');
    body.className = 'msg__content';
    body.textContent = text;
    el.appendChild(body);
    messagesEl.appendChild(el);
    scrollToBottom(true);
}

function appendBotShell() {
    heroEl.hidden = true;
    const el = document.createElement('div');
    el.className = 'msg msg--bot';
    const who = personaSel.value === 'navi' ? 'NAVI' : 'JARVIS';
    el.innerHTML = `<div class="msg__head"><span class="msg__mark">${MARK_SVG}</span><span class="msg__who">${who}</span></div><div class="msg__content"></div>`;
    messagesEl.appendChild(el);
    scrollToBottom(true);
    return el.querySelector('.msg__content');
}

function appendSystem(text) {
    const el = document.createElement('div');
    el.className = 'msg msg--system';
    const body = document.createElement('div');
    body.className = 'msg__content';
    body.textContent = text;
    el.appendChild(body);
    messagesEl.appendChild(el);
    scrollToBottom();
}

let stickBottom = true;
threadEl.addEventListener('scroll', () => {
    stickBottom = threadEl.scrollHeight - threadEl.scrollTop - threadEl.clientHeight < 100;
    scrollFab.hidden = stickBottom;
});
function scrollToBottom(force) {
    if (force || stickBottom) threadEl.scrollTop = threadEl.scrollHeight;
}
scrollFab.addEventListener('click', () => { stickBottom = true; scrollToBottom(true); });

/* ═══════════════ message actions ═══════════════ */
function addActions(contentEl, query, getResponse) {
    const bar = document.createElement('div');
    bar.className = 'actions';

    const mk = (label, title, cls) => {
        const b = document.createElement('button');
        b.className = 'act' + (cls ? ` ${cls}` : '');
        b.innerHTML = label;
        b.title = title;
        b.setAttribute('aria-label', title);
        bar.appendChild(b);
        return b;
    };

    mk('⧉', '回答をコピー').addEventListener('click', (e) => {
        navigator.clipboard.writeText(getResponse()).then(() => toast('回答をコピーしました', 'success'));
        e.currentTarget.blur();
    });
    // 再生成は「最後の応答」に対してのみ意味を持つ（会話の分岐は保持しないため）。
    // 新しい応答が出たら、古い応答の ↻ は取り除いて誤操作を防ぐ。
    messagesEl.querySelectorAll('.act--regen').forEach((b) => b.remove());
    mk('↻', '再生成', 'act--regen').addEventListener('click', () => regenerate(contentEl));
    const up = mk('👍', '良い回答');
    const down = mk('👎', '悪い回答');
    const note = document.createElement('span');
    note.className = 'act__note';
    note.hidden = true;
    bar.appendChild(note);

    [[up, 1], [down, -1]].forEach(([btn, rating]) => btn.addEventListener('click', async () => {
        up.disabled = down.disabled = true;
        btn.classList.add('is-on');
        bar.classList.add('is-pinned');
        try {
            const r = await fetch('/api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, response: getResponse(), rating }),
            });
            note.textContent = r.ok ? 'フィードバックを記録しました' : '送信に失敗しました';
        } catch {
            note.textContent = '送信に失敗しました';
        }
        note.hidden = false;
    }));

    contentEl.parentElement.appendChild(bar);
}

function regenerate(contentEl) {
    if (!lastTurn || controller) return;
    // 直前の assistant ターンを（DOM と履歴の両方から）取り消してから再送する
    const msg = contentEl.closest('.msg');
    if (msg) msg.remove();
    if (active.messages.length && active.messages[active.messages.length - 1].role === 'assistant') {
        active.messages.pop();
    }
    dispatchSend(lastTurn.text, lastTurn.images);
}

/* ═══════════════ send flow ═══════════════ */
function handleSend() {
    const text = promptEl.value.trim();
    if (controller) return;
    if (!text && attachedImages.length === 0) return;

    const images = [...attachedImages];
    promptEl.value = '';
    attachedImages = [];
    autosize(); renderAttachments(); syncComposer();

    if (!active.id) active.id = 'c' + Date.now().toString(36);
    renderUserMessage(text, images.length, images);
    active.messages.push({ role: 'user', content: text, images: images.length });
    lastTurn = { text, images };
    dispatchSend(text, images);
}

async function dispatchSend(text, images) {
    turnChatId = active.id;
    setBusy(true);
    try {
        if (modes.imagegen && engineUp.image) {
            await generateImage(text || '画像');
        } else if (modes.bon && caps.best_of_n) {
            await sendBestOfN(text, images);
        } else {
            await sendStream(text, images);
        }
    } catch (err) {
        if (err.name !== 'AbortError') {
            appendSystem(`エラー: ${err.message || 'サーバーに接続できません'}`);
            toast(err.message || '通信エラー', 'error');
        }
    } finally {
        setBusy(false);
        persistActive();
    }
}

function setBusy(busy) {
    if (!busy) controller = null;
    sendBtn.hidden = busy;
    stopBtn.hidden = !busy;
    syncComposer();
}
function abortStream() {
    if (controller) { controller.abort(); controller = null; }
}

/* 送信ペイロード（モバイルと同一の契約）。現在のユーザー発話は prompt として別送するので、
   history には含めない。 */
function payloadFor(text, images, extra) {
    const history = active.messages
        .slice(0, -1)
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .slice(-MAX_HISTORY_MESSAGES)
        .map((m) => ({ role: m.role, content: m.content }));
    return Object.assign({
        prompt: text,
        use_cot: modes.cot && caps.cot,
        images,
        persona: personaSel.value,
        client: 'web',
        history,
    }, extra || {});
}

/* 既定: ストリーミング（画像理解 / CoT / ツール / Self-RAG / 停止に対応） */
async function sendStream(text, images) {
    const useCot = modes.cot && caps.cot;
    const contentEl = appendBotShell();
    contentEl.innerHTML = '<span class="caret"></span>';
    controller = new AbortController();

    let resp;
    try {
        resp = await fetch('/api/generate_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify(payloadFor(text, images)),
        });
    } catch (e) {
        if (e.name === 'AbortError') { finalizeAbort(contentEl); return; }
        throw e;
    }
    if (!resp.ok) {
        const j = await resp.json().catch(() => ({}));
        throw new Error(j.error || `HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let full = '';
    try {
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            full += decoder.decode(value, { stream: true });
            scheduleRender(contentEl, full, useCot);
        }
    } catch (e) {
        if (e.name !== 'AbortError') throw e;
        // 停止: それまでの部分応答は残す（切り捨てない）
    }
    flushRender();
    renderBot(contentEl, full, useCot, true);
    finalizeTurn(text, full, contentEl);
}

function finalizeAbort(contentEl) {
    contentEl.innerHTML = '<span style="color:var(--text-3)">（停止しました）</span>';
}
/** ストリーム開始時のチャットがまだ開かれているか。
 *  新規チャット / 別チャットへ切り替えた後に前の応答が届いても、そちらへ書き込まない。 */
function turnIsStillOpen() { return active.id === turnChatId; }

function finalizeTurn(query, full, contentEl) {
    if (!turnIsStillOpen()) return;
    active.messages.push({ role: 'assistant', content: full });
    speakText(full);
    addActions(contentEl, query, () => full);
}

/* 高品質モード: PRM Best-of-N（非ストリームの /api/generate） */
async function sendBestOfN(text, images) {
    const useCot = modes.cot && caps.cot;
    const contentEl = appendBotShell();
    const n = parseInt(bonNSel.value, 10) || 4;
    contentEl.innerHTML = `<div class="tool-box"><span class="tool-spinner"></span><span>⭐ 高品質モード: ${n} 候補を生成し PRM で選抜中…</span></div>`;
    controller = new AbortController();

    let resp;
    try {
        resp = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify(payloadFor(text, images, { best_of_n: n })),
        });
    } catch (e) {
        if (e.name === 'AbortError') { finalizeAbort(contentEl); return; }
        throw e;
    }
    if (!resp.ok) {
        const j = await resp.json().catch(() => ({}));
        throw new Error(j.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    const answer = extractAssistant(data.response || '');
    renderBot(contentEl, answer, useCot, true);
    finalizeTurn(text, answer, contentEl);
}
function extractAssistant(t) {
    const marker = '<|im_start|>assistant\n';
    const i = t.lastIndexOf(marker);
    let s = i >= 0 ? t.slice(i + marker.length) : t;
    s = s.split('<|im_end|>')[0];
    return s.trim() || t.trim();
}

/* 画像生成（SWI-Image） */
async function generateImage(prompt) {
    const contentEl = appendBotShell();
    contentEl.innerHTML = '<div class="tool-box"><span class="tool-spinner"></span><span>🎨 SWI-Image で画像を生成中…</span></div>';
    controller = new AbortController();

    let resp;
    try {
        resp = await fetch('/api/generate_image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify({ prompt }),
        });
    } catch (e) {
        if (e.name === 'AbortError') { finalizeAbort(contentEl); return; }
        throw e;
    }
    if (!resp.ok) {
        let msg = '画像生成に失敗しました';
        if (resp.status === 503) msg = '画像モデルが未学習です（swi_image を学習してください）';
        else { const j = await resp.json().catch(() => ({})); msg = j.error || msg; }
        contentEl.innerHTML = `<div class="callout"><span class="callout__ico">⚠️</span><span>${escapeHtml(msg)}</span></div>`;
        if (turnIsStillOpen()) active.messages.push({ role: 'assistant', content: `⚠️ ${msg}` });
        return;
    }
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    contentEl.innerHTML = '';
    const cap = document.createElement('div');
    cap.className = 'genimg-cap';
    cap.textContent = `🎨 "${prompt}"`;
    const img = document.createElement('img');
    img.className = 'genimg';
    img.alt = prompt;
    img.src = url;
    img.onload = () => URL.revokeObjectURL(url);
    contentEl.append(cap, img);
    // 生成画像そのものは復元できない（Blob URL のため）。履歴にはテキストで記録する。
    if (turnIsStillOpen()) {
        active.messages.push({ role: 'assistant', content: `🎨 画像を生成しました: "${prompt}"` });
    }
    scrollToBottom(true);
}

/* ═══════════════ text-to-speech ═══════════════ */
function cleanForSpeech(text) {
    return text
        .replace(/\[(?:システム|System)[:：][\s\S]*?\]/g, '')
        .replace(/<tool_result>[\s\S]*?<\/tool_result>/g, '')
        .replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '')
        .replace(/<think>[\s\S]*?<\/think>/g, '')
        .replace(/<[^>]+>/g, '')
        .replace(/[⚙️⚠️🛑✨🔧✔✓]/g, '')
        .replace(/\$\$[\s\S]*?\$\$/g, '')
        .trim();
}
async function speakText(text) {
    if (!modes.tts) return;
    const clean = cleanForSpeech(text);
    if (!clean) return;
    if (ttsEngineSel.value === 'swivoice' && engineUp.voice) {
        if (window.speechSynthesis) window.speechSynthesis.cancel();
        if (await speakSwiStream(clean)) return;
        if (await speakSwiBatch(clean)) return;
    }
    speakBrowser(clean);
}
function speakBrowser(clean) {
    const synth = window.speechSynthesis;
    if (!synth) return;
    synth.cancel();
    const u = new SpeechSynthesisUtterance(clean);
    u.lang = 'ja-JP';
    if (personaSel.value === 'navi') { u.rate = 1.15; u.pitch = 1.4; }
    else { u.rate = 1.05; u.pitch = 0.85; }
    synth.speak(u);
}
function stopCurrentTts() {
    if (currentTtsAudio) { try { currentTtsAudio.pause(); } catch { /* noop */ } currentTtsAudio = null; }
}
async function speakSwiBatch(clean) {
    try {
        const r = await fetch('/api/tts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(withRefAudio({ text: clean, lang: 'ja' })),
        });
        if (!r.ok) { await warnIfConsentDenied(r); return false; }
        const url = URL.createObjectURL(await r.blob());
        stopCurrentTts();
        const a = new Audio(url);
        currentTtsAudio = a;
        a.onended = () => URL.revokeObjectURL(url);
        await a.play();
        return true;
    } catch { return false; }
}
async function speakSwiStream(clean) {
    try {
        const r = await fetch('/api/tts_stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(withRefAudio({ text: clean, lang: 'ja' })),
        });
        if (!r.ok || !r.body) { await warnIfConsentDenied(r); return false; }
        stopCurrentTts();
        const queue = [];
        let playing = false, gotAny = false;
        const playNext = () => {
            if (playing) return;
            const url = queue.shift();
            if (!url) return;
            playing = true;
            const a = new Audio(url);
            currentTtsAudio = a;
            const next = () => { URL.revokeObjectURL(url); playing = false; playNext(); };
            a.onended = next; a.onerror = next;
            a.play().catch(next);
        };
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            let nl;
            while ((nl = buf.indexOf('\n')) >= 0) {
                const line = buf.slice(0, nl);
                buf = buf.slice(nl + 1);
                if (!line || line.startsWith('__ERROR__')) continue;
                const bytes = Uint8Array.from(atob(line), (c) => c.charCodeAt(0));
                queue.push(URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' })));
                gotAny = true;
                playNext();
            }
        }
        return gotAny;
    } catch { return false; }
}

/* ═══════════════ toasts ═══════════════ */
function toast(msg, kind) {
    const t = document.createElement('div');
    t.className = 'toast' + (kind ? ` toast--${kind}` : '');
    t.textContent = msg;
    toastsEl.appendChild(t);
    setTimeout(() => {
        t.style.opacity = '0';
        t.style.transition = 'opacity .3s';
        setTimeout(() => t.remove(), 300);
    }, 3200);
}

/* ═══════════════ init ═══════════════ */
function init() {
    const theme = lsGet(LS.theme, null) ||
        (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
    setTheme(theme);

    const persona = lsGet(LS.persona, 'jarvis');
    personaSel.value = persona;
    applyPersona(persona);

    Object.assign(modes, lsGet(LS.modes, {}));
    bonNSel.value = lsGet(LS.bonN, '4');
    ttsEngineSel.value = lsGet(LS.ttsEngine, 'browser');

    syncModeButtons();
    syncComposer();
    syncVoiceRefRow();
    renderChatList();

    const lastId = lsGet(LS.activeChat, null);
    if (lastId && loadChats().some((c) => c.id === lastId)) openChat(lastId);

    refreshStatus();
    promptEl.focus();
}
init();

})();
