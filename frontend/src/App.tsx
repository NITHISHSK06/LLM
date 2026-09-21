import { useEffect, useState } from "react";
import { Check, ChevronLeft, ChevronRight, Copy, Menu, Plus, Send, Settings, Sparkles, X } from "lucide-react";
import SettingsPanel from "./components/SettingsPanel";
import { checkHealth, sendMessage } from "./services/api";
import { DEFAULT_SETTINGS, type ChatMessage, type GenerationSettings } from "./types";

const suggestions = ["Hello! How are you?", "Explain machine learning simply.", "Tell me something interesting.", "Help me write a Python function."];
const HISTORY_KEY = "rivuni-chat-history";
const newId = () => `${Date.now()}-${Math.random().toString(36).slice(2)}`;

function App() {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [history, setHistory] = useState<ChatMessage[][]>(() => JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]"));
    const [settings, setSettings] = useState<GenerationSettings>(DEFAULT_SETTINGS);
    const [draft, setDraft] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [connected, setConnected] = useState(false);
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [copied, setCopied] = useState<string | null>(null);

    useEffect(() => {
        checkHealth().then(() => setConnected(true)).catch(() => setConnected(false));
    }, []);

    useEffect(() => {
        if (messages.length > 0) localStorage.setItem(HISTORY_KEY, JSON.stringify([messages, ...history.filter((item) => item !== messages).slice(0, 9)]));
    }, [messages, history]);

    const startNewChat = () => { if (messages.length > 0) setHistory((items) => [messages, ...items.filter((item) => item !== messages)].slice(0, 10)); setMessages([]); setError(""); setSidebarOpen(false); };

    const submit = async (text = draft) => {
        const content = text.trim();
        if (!content || loading) return;
        const userMessage: ChatMessage = { id: newId(), role: "user", content, timestamp: new Date().toISOString() };
        setMessages((items) => [...items, userMessage]); setDraft(""); setLoading(true); setError("");
        try {
            const result = await sendMessage(content, settings);
            setMessages((items) => [...items, { id: newId(), role: "assistant", content: result.response, timestamp: new Date().toISOString() }]);
        } catch {
            setError("Unable to connect to the LLM server. Make sure the backend is running."); setConnected(false);
        } finally { setLoading(false); }
    };

    const copyResponse = async (message: ChatMessage) => { await navigator.clipboard.writeText(message.content); setCopied(message.id); window.setTimeout(() => setCopied(null), 1500); };

    return (
        <div className="app-shell">
            <aside className={`sidebar ${sidebarOpen ? "is-open" : ""}`}>
                <div className="brand"><div className="brand-mark"><Sparkles size={17} /></div><div><strong>Rivuni</strong><small>LOCAL LANGUAGE LAB</small></div></div>
                <button className="new-chat" onClick={startNewChat}><Plus size={17} /> New chat</button>
                <div className="history-heading"><span>Recent chats</span><span>{history.length}</span></div>
                <div className="history-list">{history.map((chat, index) => <button className="history-item" key={`${index}-${chat[0]?.id}`} onClick={() => { setMessages(chat); setSidebarOpen(false); }}>{chat[0]?.content || "Untitled chat"}</button>)}</div>
                <div className="sidebar-bottom">
                    <button className="sidebar-link" onClick={() => setSettingsOpen((value) => !value)}><Settings size={17} /> Settings</button>
                    <div className="sidebar-model"><span className="status-dot" /> <div><strong>Exp012</strong><small>25.52M parameters</small></div></div>
                </div>
                <button className="collapse-button" aria-label="Collapse sidebar" onClick={() => setSidebarOpen(false)}><ChevronLeft size={18} /></button>
            </aside>
            {sidebarOpen && <button className="scrim" aria-label="Close sidebar" onClick={() => setSidebarOpen(false)} />}
            <main className="chat-layout">
                <header className="topbar"><button className="icon-button mobile-menu" aria-label="Open sidebar" onClick={() => setSidebarOpen(true)}><Menu size={19} /></button><div><span className="eyebrow">PRIVATE INFERENCE</span><h1>Rivuni LLM <span className="model-badge">EXP012 · 25.5M</span></h1></div><div className={`connection ${connected ? "online" : "offline"}`}><span /> {connected ? "LLM Connected" : "LLM Offline"}</div></header>
                {settingsOpen && <SettingsPanel settings={settings} onChange={setSettings} onReset={() => setSettings(DEFAULT_SETTINGS)} />}
                <section className="conversation" aria-live="polite">
                    {messages.length === 0 ? <div className="welcome"><div className="welcome-icon"><Sparkles size={25} /></div><p className="eyebrow">EXP012 / RESEARCH PREVIEW</p><h2>Chat with Rivuni LLM</h2><p>Ask something and see what Exp012 can generate.</p><div className="suggestions">{suggestions.map((suggestion) => <button key={suggestion} onClick={() => submit(suggestion)}>{suggestion}<ChevronRight size={15} /></button>)}</div></div> : <div className="message-list">{messages.map((message) => <article className={`message-row ${message.role}`} key={message.id}><div className="avatar">{message.role === "assistant" ? <Sparkles size={15} /> : "Y"}</div><div className="message-content"><div className="message-meta"><strong>{message.role === "assistant" ? "Rivuni" : "You"}</strong><time>{new Date(message.timestamp).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}</time></div><div className="message-card">{message.content}</div>{message.role === "assistant" && <button className="copy-button" onClick={() => copyResponse(message)}>{copied === message.id ? <Check size={13} /> : <Copy size={13} />} {copied === message.id ? "Copied" : "Copy"}</button>}</div></article>)}{loading && <div className="message-row assistant"><div className="avatar"><Sparkles size={15} /></div><div className="message-content"><div className="message-meta"><strong>Rivuni</strong><span>generating</span></div><div className="message-card typing"><i /><i /><i /></div></div></div>}</div>}
                    {error && <div className="error-banner"><X size={16} />{error}</div>}
                </section>
                <form className="composer-wrap" onSubmit={(event) => { event.preventDefault(); void submit(); }}><div className="composer"><textarea aria-label="Message Rivuni LLM" placeholder="Message Rivuni LLM..." value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void submit(); } }} disabled={loading} rows={1} /><button className="send-button" aria-label="Send message" disabled={loading || !draft.trim()}><Send size={18} /></button></div><span className="composer-note">Enter to send · Shift + Enter for a new line</span></form>
            </main>
            {!sidebarOpen && <button className="expand-button" aria-label="Expand sidebar" onClick={() => setSidebarOpen(true)}><ChevronRight size={18} /></button>}
        </div>
    );
}

export default App;