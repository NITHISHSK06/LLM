import { RotateCcw, SlidersHorizontal } from "lucide-react";
import type { GenerationSettings } from "../types";

type Props = {
    settings: GenerationSettings;
    onChange: (settings: GenerationSettings) => void;
    onReset: () => void;
};

export default function SettingsPanel({ settings, onChange, onReset }: Props) {
    const update = (key: keyof GenerationSettings, value: number) => onChange({ ...settings, [key]: value });

    return (
        <aside className="settings-panel" aria-label="Model settings">
            <div className="panel-heading"><SlidersHorizontal size={17} /><span>Generation</span></div>
            <label>Temperature <output>{settings.temperature.toFixed(1)}</output>
                <input type="range" min="0.1" max="1.5" step="0.1" value={settings.temperature} onChange={(event) => update("temperature", Number(event.target.value))} />
            </label>
            <label>Top-K <output>{settings.top_k}</output>
                <input type="range" min="0" max="100" step="1" value={settings.top_k} onChange={(event) => update("top_k", Number(event.target.value))} />
            </label>
            <label>Top-P <output>{settings.top_p.toFixed(1)}</output>
                <input type="range" min="0.1" max="1" step="0.1" value={settings.top_p} onChange={(event) => update("top_p", Number(event.target.value))} />
            </label>
            <label>Max new tokens <output>{settings.max_new_tokens}</output>
                <input type="range" min="20" max="150" step="1" value={settings.max_new_tokens} onChange={(event) => update("max_new_tokens", Number(event.target.value))} />
            </label>
            <button className="reset-button" onClick={onReset}><RotateCcw size={14} /> Reset defaults</button>
            <div className="model-details">
                <span>Model</span><strong>Exp012</strong>
                <span>Parameters</span><strong>25.52M</strong>
                <span>Context</span><strong>256 tokens</strong>
                <span>Tokenizer</span><strong>tokenizer_exp003</strong>
            </div>
        </aside>
    );
}