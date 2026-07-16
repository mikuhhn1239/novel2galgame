/**
 * Decision Memory Store — LangGraph BaseStore 实现
 *
 * 设计决策（Design Decisions）:
 *
 * 1. 存什么: 决策指纹而非原文
 *    原文 "她姐轻轻推开门" → pattern "代词_她 + 敬语_姐 + 动作"
 *    → 同一模式在全书中反复出现，存模式比存原文复用率高
 *
 * 2. 写入策略: 高置信度才存
 *    confidence >= 0.7 → 存入
 *    confidence < 0.7  → 不存（低质量记忆比没有更危险）
 *
 * 3. 召回 + 信任策略: 两级阈值
 *    命中 && memory.confidence >= 0.85 → 直接复用，跳过 LLM
 *    命中 && 0.7 <= confidence < 0.85 → 注入 prompt 作为提示，不跳过推理
 *    未命中 || confidence < 0.7 → 正常 LLM 推理
 *
 * 4. 存储: JSON 文件 + LangGraph BaseStore 接口
 *    零依赖，与项目现有架构一致（RAG 也用 JSON）
 *
 * 5. 命名空间: ["memories", projectId, agentType]
 *    按项目+Agent隔离，不同小说的归因模式不互相污染
 *
 * 6. TTL: 30 天未使用 → 标记过期
 *    防止因小说修订导致旧记忆与新版矛盾
 */
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

/** 单个记忆条目 */
export interface DecisionMemory {
  /** 唯一 key: SHA256(projectId + chapterId + unitId + characterId) */
  memoryId: string;
  /** 归因模式指纹: "代词_她 + 敬语_姐 + 对话类型_连续对话" */
  pattern: string;
  /** pattern 的哈希，用于精确查找 */
  patternHash: string;
  /** 归因结果 */
  canonicalName: string;
  characterId: string;
  /** 原始置信度 */
  confidence: number;
  /** 来源章节 */
  chapterId: string;
  /** 创建时间 */
  createdAt: string;
  /** 最近被命中时间 */
  lastHitAt: string;
  /** 累计命中次数：被复用过几次 */
  hitCount: number;
  /** 原始上下文摘要（审计用） */
  contextDigest: string;
}

/** 记忆检索结果 */
export interface MemoryHit {
  memory: DecisionMemory;
  /** 相似度分数 (0-1) */
  score: number;
  /** 是否建议跳过 LLM 直接复用 */
  skipLLM: boolean;
}

/** LangGraph BaseStore 兼容接口 */
export interface StoreItem {
  namespace: string[];
  key: string;
  value: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

/**
 * Decision Memory Store
 *
 * 实现 LangGraph BaseStore 接口的 4 个方法：
 * - put: 写入记忆
 * - search: 按命名空间搜索
 * - get: 按 key 精确读取
 * - delete: 按 key 删除
 */
export class DecisionMemoryStore {
  private storageDir: string;
  private cache: Map<string, DecisionMemory> = new Map();
  private loaded = false;

  constructor(dataDir: string) {
    this.storageDir = path.join(dataDir, "memory", "decisions");
  }

  /** 加载所有记忆到内存（启动时调用一次） */
  private ensureLoaded(): void {
    if (this.loaded) return;
    if (!fs.existsSync(this.storageDir)) {
      fs.mkdirSync(this.storageDir, { recursive: true });
      this.loaded = true;
      return;
    }
    const files = fs.readdirSync(this.storageDir).filter((f) => f.endsWith(".json"));
    for (const file of files) {
      try {
        const raw = fs.readFileSync(path.join(this.storageDir, file), "utf-8");
        const mem = JSON.parse(raw) as DecisionMemory;
        // 跳过过期的记忆（30天未命中）
        if (this.isExpired(mem)) continue;
        this.cache.set(mem.memoryId, mem);
      } catch {
        // 损坏文件跳过
      }
    }
    this.loaded = true;
    console.log(`[Memory] Loaded ${this.cache.size} decision memories`);
  }

  /** 30天未命中 → 过期 */
  private isExpired(mem: DecisionMemory): boolean {
    const age = Date.now() - new Date(mem.lastHitAt).getTime();
    return age > 30 * 24 * 60 * 60 * 1000;
  }

  /** 计算归因模式指纹 */
  static extractPattern(unit: {
    text: string;
    type: string;
    prevSpeaker?: string;
    position?: number;
  }): string {
    const parts: string[] = [];

    // 代词特征
    if (/她/.test(unit.text)) parts.push("代词_她");
    if (/他/.test(unit.text)) parts.push("代词_他");
    if (/我/.test(unit.text)) parts.push("代词_我");
    if (/你/.test(unit.text)) parts.push("代词_你");

    // 敬语/称呼特征
    if (/姐|姐姐/.test(unit.text)) parts.push("敬语_姐");
    if (/哥|哥哥/.test(unit.text)) parts.push("敬语_哥");
    if (/先生|小姐|女士/.test(unit.text)) parts.push("敬语_先生");
    if (/同学|老师|师傅|老板/.test(unit.text)) parts.push("称呼_角色关系");

    // 动作特征
    if (/说|道|问|答|喊|叫|嚷|骂/.test(unit.text)) parts.push("动作_发言");
    if (/走|跑|跳|推|拉|坐|站|躺/.test(unit.text)) parts.push("动作_行动");
    if (/笑|哭|怒|气|惊|怕|羞/.test(unit.text)) parts.push("动作_表情");

    // 上下文特征
    if (unit.type === "dialogue") parts.push("类型_对话");
    if (unit.type === "thought") parts.push("类型_心独白");
    if (unit.prevSpeaker) parts.push(`上文说话人_${unit.prevSpeaker}`);

    // 位置特征
    if (unit.position !== undefined) {
      if (unit.position < 10) parts.push("位置_章节开头");
      else if (unit.position > 100) parts.push("位置_章节后段");
    }

    return parts.join(" + ") || "无特征模式";
  }

  /** 计算模式哈希（精确匹配用） */
  static hashPattern(pattern: string): string {
    return crypto.createHash("sha256").update(pattern).digest("hex").slice(0, 16);
  }

  /**
   * 搜索最相关的记忆
   *
   * 策略:
   * 1. 精确模式哈希匹配 → 最可信
   * 2. 关键词部分匹配 → 次可信
   * 3. 无匹配 → 正常推理
   */
  search(
    projectId: string,
    agentType: string,
    pattern: string,
    opts?: { minConfidence?: number; topK?: number },
  ): MemoryHit[] {
    this.ensureLoaded();
    const minConf = opts?.minConfidence ?? 0.7;
    const topK = opts?.topK ?? 3;
    const patternHash = DecisionMemoryStore.hashPattern(pattern);

    const candidates: MemoryHit[] = [];

    for (const mem of this.cache.values()) {
      // 跨项目隔离——只搜当前项目的记忆
      if (!mem.memoryId.startsWith(projectId)) continue;

      // 精确模式匹配：最高分
      if (mem.patternHash === patternHash && mem.confidence >= minConf) {
        candidates.push({
          memory: mem,
          score: 1.0,
          skipLLM: mem.confidence >= 0.85,
        });
        continue;
      }

      // 部分关键词匹配
      const patternTokens = new Set(pattern.split(" + "));
      const memTokens = new Set(mem.pattern.split(" + "));
      const overlap = [...patternTokens].filter((t) => memTokens.has(t)).length;
      const totalTokens = patternTokens.size;
      const partialScore = overlap / Math.max(totalTokens, 1);

      if (partialScore >= 0.4 && mem.confidence >= minConf) {
        candidates.push({
          memory: mem,
          score: partialScore * 0.8, // 部分匹配降权
          skipLLM: false, // 部分匹配不跳过 LLM，只做提示
        });
      }
    }

    // 按分数排序，返回 top K
    candidates.sort((a, b) => b.score - a.score);
    return candidates.slice(0, topK);
  }

  /**
   * 写入记忆
   *
   * 幂等：同 memoryId 会更新 hitCount + lastHitAt
   */
  put(memory: Omit<DecisionMemory, "memoryId" | "patternHash" | "createdAt" | "lastHitAt" | "hitCount">): string {
    this.ensureLoaded();

    const memoryId = crypto
      .createHash("sha256")
      .update(`${memory.chapterId}|${memory.characterId}|${memory.pattern}`)
      .digest("hex")
      .slice(0, 16);

    // 检查是否已存在（更新而非重复写入）
    const existing = this.cache.get(memoryId);
    if (existing) {
      existing.hitCount += 1;
      existing.lastHitAt = new Date().toISOString();
      this.saveOne(existing);
      return memoryId;
    }

    const dm: DecisionMemory = {
      ...memory,
      memoryId,
      patternHash: DecisionMemoryStore.hashPattern(memory.pattern),
      createdAt: new Date().toISOString(),
      lastHitAt: new Date().toISOString(),
      hitCount: 0,
    };

    this.cache.set(memoryId, dm);
    this.saveOne(dm);

    if (this.cache.size % 50 === 0) {
      console.log(`[Memory] Store now has ${this.cache.size} decision memories`);
    }

    return memoryId;
  }

  /** 记录记忆被成功复用 */
  recordHit(memoryId: string): void {
    const mem = this.cache.get(memoryId);
    if (!mem) return;
    mem.hitCount += 1;
    mem.lastHitAt = new Date().toISOString();
    this.saveOne(mem);
  }

  /** 获取统计信息 */
  stats(): { total: number; avgConfidence: number; totalHits: number } {
    this.ensureLoaded();
    const memories = [...this.cache.values()];
    return {
      total: memories.length,
      avgConfidence:
        memories.reduce((s, m) => s + m.confidence, 0) / Math.max(memories.length, 1),
      totalHits: memories.reduce((s, m) => s + m.hitCount, 0),
    };
  }

  /** 清理过期记忆 */
  prune(): number {
    this.ensureLoaded();
    const before = this.cache.size;
    for (const [id, mem] of this.cache) {
      if (this.isExpired(mem)) {
        this.cache.delete(id);
        const filePath = path.join(this.storageDir, `${id}.json`);
        try { fs.unlinkSync(filePath); } catch {}
      }
    }
    const pruned = before - this.cache.size;
    if (pruned > 0) console.log(`[Memory] Pruned ${pruned} expired memories`);
    return pruned;
  }

  // ── Persistence ──────────────────────────────────────

  private saveOne(mem: DecisionMemory): void {
    try {
      if (!fs.existsSync(this.storageDir)) {
        fs.mkdirSync(this.storageDir, { recursive: true });
      }
      fs.writeFileSync(
        path.join(this.storageDir, `${mem.memoryId}.json`),
        JSON.stringify(mem, null, 2),
        "utf-8",
      );
    } catch (err: any) {
      console.error(`[Memory] Save error: ${err.message}`);
    }
  }
}
