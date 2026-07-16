import { StateGraph, END } from "@langchain/langgraph";
import { Annotation } from "@langchain/langgraph";
import { narrativeNode } from "../nodes/narrative-node.js";
import { attributionNode } from "../nodes/attribution-node.js";
import { DecisionMemoryStore, type MemoryHit } from "../memory/decision-store.js";

export const UnderstandingState = Annotation.Root({
  projectId: Annotation<string>,
  chapterId: Annotation<string>,
  chapterTitle: Annotation<string>,
  chapterText: Annotation<string>,
  dataDir: Annotation<string>,
  narrativeResult: Annotation<any>({ default: () => null, reducer: (_prev, next) => next }),
  attributionResult: Annotation<any>({ default: () => null, reducer: (_prev, next) => next }),
  ragContext: Annotation<any>({ default: () => ({ knownCharacters: [], characterKnowledge: "" }), reducer: (_prev, next) => next }),
  modelConfig: Annotation<any>({ default: () => ({}), reducer: (_prev, next) => next }),
  provider: Annotation<any>({ default: () => null, reducer: (_prev, next) => next }),
  defaultModel: Annotation<string>,
  signal: Annotation<AbortSignal | null>({ default: () => null, reducer: (_prev, next) => next }),
  db: Annotation<any>({ default: () => null, reducer: (_prev, next) => next }),
  onProgress: Annotation<any>({ default: () => null, reducer: (_prev, next) => next }),
  onChapterFlags: Annotation<any>({ default: () => null, reducer: (_prev, next) => null }),
  error: Annotation<string | null>({ default: () => null, reducer: (_prev, next) => next }),
  retryCount: Annotation<number>({ default: () => 0, reducer: (_prev, next) => next }),
  parentState: Annotation<any>({ default: () => ({}), reducer: (_prev, next) => next }),
  // ── Memory annotations ──
  memoryStore: Annotation<DecisionMemoryStore | null>({ default: () => null, reducer: (_prev, next) => next }),
  memoryHits: Annotation<MemoryHit[]>({ default: () => [], reducer: (_prev, next) => next }),
  memoryUsed: Annotation<boolean>({ default: () => false, reducer: (_prev, next) => next }),
});

function routeAfterNarrative(state: typeof UnderstandingState.State): string {
  if (state.error) return "handle_error";
  if (!state.narrativeResult) return "handle_error";
  // If memory store is configured, search it first
  if (state.memoryStore) return "memory_search";
  return "attribution";
}

function routeAfterMemorySearch(state: typeof UnderstandingState.State): string {
  if (state.error) return "handle_error";
  const hits = state.memoryHits ?? [];
  // 找到一个高置信度精确匹配 → 跳过 LLM，直接复用
  const exactHit = hits.find((h) => h.skipLLM && h.score >= 0.95);
  if (exactHit) {
    state.onProgress?.("attribution", `Memory hit: reusing ${exactHit.memory.canonicalName} (confidence: ${exactHit.memory.confidence})`);
    return "memory_apply";
  }
  // 有部分匹配 → 注入 prompt 但不跳过推理
  const partialHits = hits.filter((h) => h.score >= 0.4);
  if (partialHits.length > 0) {
    state.onProgress?.("attribution", `Memory hints: ${partialHits.length} partial matches found`);
    return "attribution";
  }
  return "attribution";
}

function afterAttribution(state: typeof UnderstandingState.State): string {
  if (state.error) return "handle_error";
  if (!state.attributionResult) return "rag_lookup";
  // Attribution succeeded → write to memory
  if (state.memoryStore) return "memory_write";
  return "done";
}

function afterRAGLookup(state: typeof UnderstandingState.State): string {
  if (state.retryCount >= 2) return "done";
  return "attribution";
}

export function buildUnderstandingSubgraph() {
  return new StateGraph(UnderstandingState)
    .addNode("narrative", narrativeNode as any)
    .addNode("memory_search", async (state: typeof UnderstandingState.State) => {
      if (!state.memoryStore || !state.narrativeResult) return {};
      const store = state.memoryStore;
      const allHits: MemoryHit[] = [];
      const units = state.narrativeResult.units ?? [];

      // 对每个 dialogue 类型的 unit 提取模式并搜索
      let searchedCount = 0;
      for (const unit of units) {
        if (unit.type !== "dialogue" && unit.type !== "thought") continue;
        if (searchedCount > 20) break; // 最多搜索 20 个 unit，避免过度开销

        const pattern = DecisionMemoryStore.extractPattern({
          text: unit.text ?? "",
          type: unit.type ?? "dialogue",
          position: units.indexOf(unit),
        });
        const hits = store.search(state.projectId, "attribution", pattern, {
          minConfidence: 0.7,
          topK: 2,
        });
        for (const hit of hits) allHits.push(hit);
        searchedCount++;
      }

      // 去重 + 按分数排序
      const seen = new Set<string>();
      const deduped = allHits
        .filter((h) => {
          const key = h.memory.memoryId;
          if (seen.has(key)) return false;
          seen.add(key);
          return true;
        })
        .sort((a, b) => b.score - a.score);

      if (deduped.length > 0) {
        console.log(`[Memory] Searched ${searchedCount} units, found ${deduped.length} candidate memories (top score: ${deduped[0]?.score.toFixed(2)})`);
      }

      return { memoryHits: deduped };
    })
    .addNode("memory_apply", async (state: typeof UnderstandingState.State) => {
      // 直接复用记忆，不调 LLM
      const hit = state.memoryHits?.find((h) => h.skipLLM);
      if (!hit) return {};

      const mem = hit.memory;
      // 构造最小 attributionResult
      const fakeAttribution = {
        chapterId: state.chapterId,
        units: (state.narrativeResult?.units ?? []).map((u: any) => ({
          ...u,
          speaker: u.type === "dialogue" ? mem.canonicalName : u.speaker,
          characterId: u.type === "dialogue" ? mem.characterId : u.characterId,
          confidence: mem.confidence,
          fromMemory: true,
        })),
        characters: [{ characterId: mem.characterId, canonicalName: mem.canonicalName }],
      };

      state.memoryStore!.recordHit(mem.memoryId);
      state.onProgress?.("attribution", `Memory reused: ${mem.canonicalName} (saved 1 LLM call)`);

      return {
        attributionResult: fakeAttribution,
        memoryUsed: true,
        currentStage: "done",
      };
    })
    .addNode("attribution", attributionNode as any)
    .addNode("memory_write", async (state: typeof UnderstandingState.State) => {
      if (!state.memoryStore || !state.attributionResult) return {};
      const store = state.memoryStore;
      const result = state.attributionResult;
      let written = 0;

      for (const unit of result.units ?? []) {
        const confidence = unit.attributionConfidence ?? unit.confidence ?? 0;
        // 只写入高置信度决策
        if (confidence < 0.7) continue;

        const pattern = DecisionMemoryStore.extractPattern({
          text: unit.originalText ?? unit.text ?? "",
          type: unit.type ?? "dialogue",
          prevSpeaker: unit.prevSpeaker,
          position: (result.units ?? []).indexOf(unit),
        });

        const speaker = unit.speaker ?? unit.characterId ?? "unknown";
        const charId = unit.characterId ?? speaker;

        if (speaker === "unknown") continue;

        store.put({
          pattern,
          canonicalName: speaker,
          characterId: charId,
          confidence,
          chapterId: state.chapterId,
          contextDigest: (unit.originalText ?? unit.text ?? "").slice(0, 100),
        });
        written++;
      }

      if (written > 0) {
        console.log(`[Memory] Wrote ${written} new decision memories from ${state.chapterId}`);
      }
      return {};
    })
    .addNode("rag_lookup", async (state: typeof UnderstandingState.State) => {
      return { retryCount: (state.retryCount ?? 0) + 1 };
    })
    .addNode("handle_error", async (state: typeof UnderstandingState.State) => {
      return { error: state.error ?? "Understanding subgraph failed" };
    })
    // ── New flow with memory ──
    .addEdge("__start__", "narrative")
    .addConditionalEdges("narrative", routeAfterNarrative, {
      memory_search: "memory_search",
      attribution: "attribution",
      handle_error: "handle_error",
    })
    .addConditionalEdges("memory_search", routeAfterMemorySearch, {
      memory_apply: "memory_apply",
      attribution: "attribution",
      handle_error: "handle_error",
    })
    .addEdge("memory_apply", END)
    .addEdge("attribution", "rag_lookup")  // Always try RAG enhancement if available
    .addConditionalEdges("rag_lookup", afterRAGLookup, {
      attribution: "attribution",
      done: "memory_write",
    })
    .addEdge("memory_write", END)
    .addEdge("handle_error", END)
    .compile();
}
