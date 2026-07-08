/**
 * Vector store abstraction.
 * Uses a simple JSON file-based store for zero-dependency operation.
 * Can be swapped for lanceDB when running on a system with native support.
 */

import fs from "node:fs";
import path from "node:path";

export interface VectorRecord {
  id: string;
  vector: number[];
  metadata: Record<string, unknown>;
}

interface StoreData {
  records: VectorRecord[];
}

export class VectorStore {
  private storePath: string;
  private records: VectorRecord[] = [];

  constructor(dataDir: string, namespace: string) {
    this.storePath = path.join(dataDir, "rag", `${namespace}.json`);
    this.load();
  }

  private load() {
    try {
      if (fs.existsSync(this.storePath)) {
        const data: StoreData = JSON.parse(fs.readFileSync(this.storePath, "utf-8"));
        this.records = data.records ?? [];
      }
    } catch { this.records = []; }
  }

  private save() {
    fs.mkdirSync(path.dirname(this.storePath), { recursive: true });
    fs.writeFileSync(this.storePath, JSON.stringify({ records: this.records }), "utf-8");
  }

  /** Cosine similarity between two vectors */
  private cosineSim(a: number[], b: number[]): number {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
  }

  /** Add records with their embedding vectors */
  add(records: VectorRecord[]) {
    this.records.push(...records);
    this.save();
  }

  /** Upsert records — replace existing records with same id, add new ones */
  upsert(records: VectorRecord[]) {
    for (const r of records) {
      const idx = this.records.findIndex((existing) => existing.id === r.id);
      if (idx >= 0) {
        this.records[idx] = r; // Replace
      } else {
        this.records.push(r);  // Add new
      }
    }
    this.save();
  }

  /** Search by vector, return top-K with scores */
  searchWithScore(queryVector: number[], limit: number = 5): { record: VectorRecord; score: number }[] {
    const scored = this.records.map((r) => ({
      record: r,
      score: this.cosineSim(queryVector, r.vector),
    }));
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit);
  }

  /** Search by vector, return top-K results sorted by similarity */
  search(queryVector: number[], limit: number = 5): VectorRecord[] {
    return this.searchWithScore(queryVector, limit).map((s) => s.record);
  }

  /** Get all records */
  getAll(): VectorRecord[] {
    return this.records;
  }

  /** Delete all records (for reset) */
  clear() {
    this.records = [];
    this.save();
  }

  get count(): number { return this.records.length; }
}
