/**
 * Training data collection for few-shot retrieval in LoRA data generation.
 *
 * The cascading labeling pipeline (narrative → scene → attribution) needs
 * few-shot examples from PREVIOUS chapters to improve labeling quality on
 * the current chapter. This collection stores labeled text snippets and
 * retrieves the most similar ones as in-context examples.
 *
 * Also provides export for LoRA training data generation.
 */

import { BaseCollection, type VectorRecord, type SearchResult, type WhereClause } from "./base.js";

export type TrainingStep = "narrative" | "scene" | "attribution";

export interface TrainingExample {
  id: string;
  /** Original text snippet that was labeled */
  text: string;
  /** The label assigned by the pipeline step */
  label: string;
  /** Structured output from the step (JSON-serializable) */
  output: Record<string, unknown>;
  /** Which pipeline step produced this example */
  step: TrainingStep;
  /** Source chapter ID */
  chapterId: string;
  /** Source novel ID (for cross-novel retrieval) */
  sourceNovelId: string;
  /** Text for embedding (searchable summary) */
  embedText: string;
  /** Label quality score (0-1), from validation or self-assessment */
  quality: number;
  _score?: number;
}

export class TrainingDataCollection extends BaseCollection {
  constructor(dataDir: string) {
    super(dataDir, "training_data");
  }

  /** Ingest labeled training examples. */
  ingest(examples: TrainingExample[], vectors: number[][]): void {
    if (examples.length === 0) return;

    const records: VectorRecord[] = examples.map((e, i) => ({
      id: e.id,
      vector: vectors[i]!,
      metadata: {
        type: "training_example",
        text: e.text,
        label: e.label,
        output: e.output,
        step: e.step,
        chapterId: e.chapterId,
        sourceNovelId: e.sourceNovelId,
        embedText: e.embedText,
        quality: e.quality,
      },
      updatedAt: new Date().toISOString(),
    }));

    this.upsert(records);
    console.log(
      `[RAG-v2] Ingested ${examples.length} training examples (total: ${this.count}, step: ${examples[0]?.step ?? "unknown"})`,
    );
  }

  /** Override: BM25 text includes the label and original text for keyword match. */
  protected override getDocText(record: VectorRecord): string {
    const m = record.metadata;
    return [
      m.label ?? "",
      m.text ?? "",
      m.embedText ?? "",
      m.step ?? "",
    ]
      .filter(Boolean)
      .join(" ");
  }

  private toExample(r: SearchResult): TrainingExample {
    const m = r.record.metadata;
    return {
      id: r.record.id,
      text: (m.text as string) ?? "",
      label: (m.label as string) ?? "",
      output: (m.output as Record<string, unknown>) ?? {},
      step: (m.step as TrainingStep) ?? "narrative",
      chapterId: (m.chapterId as string) ?? "",
      sourceNovelId: (m.sourceNovelId as string) ?? "",
      embedText: (m.embedText as string) ?? "",
      quality: (m.quality as number) ?? 0.5,
      _score: r.score,
    };
  }

  /**
   * Retrieve few-shot examples for a pipeline step.
   * Excludes the current chapter to prevent data leakage.
   */
  searchFewShot(
    queryVector: number[],
    options?: {
      step?: TrainingStep;
      topK?: number;
      minQuality?: number;
      excludeChapterId?: string;
    },
  ): TrainingExample[] {
    const where: WhereClause = {};
    if (options?.step) {
      where.step = { $eq: options.step };
    }
    if (options?.excludeChapterId) {
      where.chapterId = { $ne: options.excludeChapterId };
    }
    if (options?.minQuality !== undefined) {
      where.quality = { $gte: options.minQuality };
    }

    const results = this.search(queryVector, {
      topK: options?.topK ?? 3,
      minScore: 0.5,
      where: Object.keys(where).length > 0 ? where : undefined,
    });

    return results.map((r) => this.toExample(r));
  }

  /**
   * Hybrid few-shot search: vector + BM25 keyword fusion.
   * Keyword matching helps when the query text shares vocabulary with stored labels.
   */
  searchFewShotHybrid(
    queryVector: number[],
    queryText: string,
    options?: {
      step?: TrainingStep;
      topK?: number;
      minQuality?: number;
      excludeChapterId?: string;
      vectorWeight?: number;
    },
  ): TrainingExample[] {
    // Broad hybrid search → post-filter by step/chapter/quality
    const results = this.hybridSearch(
      queryVector,
      queryText,
      (options?.topK ?? 3) * 3,
      options?.vectorWeight ?? 0.6,
    );

    const filtered = results.filter((r) => {
      if (options?.step && r.record.metadata.step !== options.step) return false;
      if (options?.excludeChapterId && r.record.metadata.chapterId === options.excludeChapterId) return false;
      if (options?.minQuality !== undefined && (r.record.metadata.quality as number) < options.minQuality) return false;
      return r.score >= 0.5;
    });

    return filtered.slice(0, options?.topK ?? 3).map((r) => this.toExample(r));
  }

  // ── Export for LoRA training ──────────────────────────

  /** Export all examples for a step as { text, label } pairs. */
  exportByStep(step: TrainingStep): Array<{ text: string; label: string; quality: number }> {
    return this.records
      .filter((r) => r.metadata.step === step)
      .map((r) => ({
        text: (r.metadata.text as string) ?? "",
        label: (r.metadata.label as string) ?? "",
        quality: (r.metadata.quality as number) ?? 0.5,
      }));
  }

  /**
   * Export examples for a step, shuffled for training data generation.
   * Shuffling prevents order bias in LoRA fine-tuning.
   */
  exportShuffled(step: TrainingStep): Array<{ text: string; label: string; quality: number }> {
    const examples = this.exportByStep(step);
    // Fisher-Yates shuffle
    for (let i = examples.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [examples[i], examples[j]] = [examples[j]!, examples[i]!];
    }
    return examples;
  }

  /** Count examples per step. Returns { narrative: N, scene: N, attribution: N }. */
  countByStep(): Record<TrainingStep, number> {
    const counts: Record<TrainingStep, number> = {
      narrative: 0,
      scene: 0,
      attribution: 0,
    };
    for (const r of this.records) {
      const step = r.metadata.step as TrainingStep;
      if (step) counts[step] = (counts[step] ?? 0) + 1;
    }
    return counts;
  }
}
