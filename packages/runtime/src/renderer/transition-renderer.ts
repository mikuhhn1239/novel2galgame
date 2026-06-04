import type { RenderAction } from "../step-engine/step-types.js";

export interface TransitionEffect {
  name: string;
  durationMs: number;
}

const TRANSITION_DURATIONS: Record<string, number> = {
  fade: 800,
  cut: 0,
  dissolve: 1200,
};

export function resolveTransition(action: Extract<RenderAction, { type: "transition" }>): TransitionEffect {
  const name = action.name ?? "cut";
  return {
    name,
    durationMs: TRANSITION_DURATIONS[name] ?? 500,
  };
}
