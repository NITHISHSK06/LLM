export type Role = "user" | "assistant";

export type ChatMessage = {
    id: string;
    role: Role;
    content: string;
    timestamp: string;
};

export type GenerationSettings = {
    temperature: number;
    top_k: number;
    top_p: number;
    max_new_tokens: number;
};

export const DEFAULT_SETTINGS: GenerationSettings = {
    temperature: 0.7,
    top_k: 40,
    top_p: 0.9,
    max_new_tokens: 80,
};