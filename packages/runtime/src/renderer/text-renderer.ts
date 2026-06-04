import type { RenderAction } from "../step-engine/step-types.js";

export interface TextDisplay {
  mode: "narration" | "dialogue" | "thought";
  text: string;
  characterId?: string;
  displayName?: string;
}

export function resolveNarration(action: Extract<RenderAction, { type: "showNarration" }>): TextDisplay {
  return { mode: "narration", text: action.text };
}

export function resolveDialogue(action: Extract<RenderAction, { type: "showDialogue" }>): TextDisplay {
  return {
    mode: "dialogue",
    text: action.text,
    characterId: action.characterId,
    displayName: action.displayName,
  };
}

export function resolveThought(action: Extract<RenderAction, { type: "showThought" }>): TextDisplay {
  return {
    mode: "thought",
    text: action.text,
    characterId: action.characterId,
    displayName: action.displayName,
  };
}
