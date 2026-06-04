import type { RenderAction } from "../step-engine/step-types.js";

export interface BackgroundState {
  id: string;
  label?: string;
}

export function resolveBackground(action: Extract<RenderAction, { type: "setBackground" }>): BackgroundState {
  return { id: action.id, label: action.label };
}
