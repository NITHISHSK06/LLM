import type { GenerationSettings } from "../types";

const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

export type HealthResponse = {
    status: string;
    model: string;
    parameters: number;
    device: string;
};

export type ChatResponse = {
    response: string;
    model: string;
    parameters: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 15_000);
    try {
        const response = await fetch(`${API_URL}${path}`, { ...init, signal: controller.signal });
        if (!response.ok) {
            throw new Error(`Request failed: ${response.status}`);
        }
        return (await response.json()) as T;
    } finally {
        window.clearTimeout(timeout);
    }
}

export function checkHealth(): Promise<HealthResponse> {
    return request<HealthResponse>("/api/health");
}

export function sendMessage(message: string, settings: GenerationSettings): Promise<ChatResponse> {
    return request<ChatResponse>("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, ...settings }),
    });
}