import { Router } from "express";
import type { Request, Response } from "express";
import fs from "node:fs";
import path from "node:path";
import { v4 as uuid } from "uuid";
import type { ProjectState } from "@novel2gal/core";
import {
  createDatabase,
  ProjectRepository,
  ChapterRepository,
  SceneRepository,
} from "@novel2gal/storage";
import { FetchLLMProvider } from "@novel2gal/providers";
import type { LLMProvider } from "@novel2gal/providers";
import { buildChapterPipelineGraph, buildSupervisoryPipelineGraph } from "@novel2gal/pipeline";
import { broadcastProgress } from "./progress.js";

const now = () => new Date().toISOString();

function param(req: Request, key: string): string {
  const val = req.params[key];
  return Array.isArray(val) ? val[0]! : val!;
}

interface GraphPipelineConfig {
  dataDir: string;
  db: ReturnType<typeof createDatabase>;
  getProvider: () => LLMProvider | null;
  rag?: any;
}

export function createGraphPipelineRoutes(cfg: GraphPipelineConfig) {
  const router = Router();
  const runningGraphs = new Map<string, AbortController>();

  // POST /graph-pipeline/projects/:projectId/chapters/:chapterId/run
  router.post("/projects/:projectId/chapters/:chapterId/run", async (req: Request, res: Response) => {
    const pid = param(req, "projectId");
    const cid = param(req, "chapterId");

    const provider = cfg.getProvider();
    if (!provider) {
      return res.status(500).json({ error: "No LLM provider configured" });
    }

    const projectRepo = new ProjectRepository(cfg.db);
    const chapterRepo = new ChapterRepository(cfg.db);
    const sceneRepo = new SceneRepository(cfg.db);

    const project = projectRepo.getById(pid);
    if (!project) return res.status(404).json({ error: "Project not found" });

    const chapter = chapterRepo.getById(cid);
    if (!chapter) return res.status(404).json({ error: "Chapter not found" });

    const sourcePath = path.join(cfg.dataDir, "projects", pid, "chapters", cid, "source.json");
    if (!fs.existsSync(sourcePath)) {
      return res.status(404).json({ error: "Chapter source not found" });
    }

    const chapterText = fs.readFileSync(sourcePath, "utf-8");
    const model = req.body.model ?? project.config.defaultTextModel ?? "agnes-2.0-flash";

    // Build per-agent model config (same as existing pipeline)
    let agentModels: Record<string, { provider: LLMProvider; model: string }> = {};
    const localBaseUrl = req.body.localBaseUrl;
    const localModel = req.body.localModel ?? "qwen3-8b-sft";
    if (localBaseUrl) {
      const localProvider = new FetchLLMProvider({
        apiKey: "not-needed",
        baseUrl: localBaseUrl,
        defaultModel: localModel,
        name: "local-sft",
      });
      const trained = { provider: localProvider as LLMProvider, model: localModel };
      agentModels = { narrative: trained, attribution: trained, segmentation: trained };
    }

    // Cancel any existing pipeline
    runningGraphs.get(cid)?.abort();
    runningGraphs.delete(cid);

    // Mark chapter running
    chapterRepo.updateStatus(cid, "running");

    const runId = `run_${uuid().replace(/-/g, "").slice(0, 12)}`;
    cfg.db.prepare(`INSERT INTO pipeline_runs (run_id, project_id, chapter_id, status, started_at)
      VALUES (?, ?, ?, 'running', ?)`).run(runId, pid, cid, now());

    const ac = new AbortController();
    runningGraphs.set(cid, ac);

    const graphEngine: string = req.body.graphEngine ?? "flat";
    const engineLabel = graphEngine === "supervisory" ? "langgraph-supervisory" : "langgraph";
    // Return immediately
    res.json({ chapterId: cid, runId, status: "started", engine: engineLabel });

    // Build and invoke the graph
    const graph = graphEngine === "supervisory"
      ? buildSupervisoryPipelineGraph()
      : buildChapterPipelineGraph();

    const initialState = {
      projectId: pid,
      chapterId: cid,
      chapterTitle: chapter.title,
      chapterText,
      dataDir: cfg.dataDir,
      provider,
      defaultModel: model,
      modelConfig: agentModels,
      signal: ac.signal,
      db: cfg.db,
      sceneRepo,
      autoRunVisualPrompt: project.config.autoRunVisualPrompt ?? false,
      onProgress: (stage: string, message: string) => {
        broadcastProgress({ projectId: pid, chapterId: cid, stage, status: "progress", message });
      },
      onChapterFlags: (chId: string, flags: any) => {
        try { chapterRepo.updateFlags(chId, flags); } catch {}
      },
      onSceneCreated: (scene: any, idx: number) => {
        try { sceneRepo.create(scene, idx); } catch {}
      },
    };

    try {
      // Stream graph execution with state updates
      const stream = await graph.stream(initialState, {
        configurable: {
          thread_id: `${pid}_${cid}`,
          rag: cfg.rag,
        },
        signal: ac.signal,
      });

      let finalState: any = null;
      for await (const chunk of stream) {
        finalState = chunk;
        // Broadcast stage transitions
        const stage = chunk.currentStage;
        if (stage) {
          cfg.db.prepare("UPDATE pipeline_runs SET current_stage=? WHERE run_id=?")
            .run(stage, runId);
          broadcastProgress({
            projectId: pid, chapterId: cid, stage,
            status: "progress", message: `Stage: ${stage}`,
          });
        }
      }

      const sceneCount = finalState?.segmentationResult?.scenes?.length ?? 0;
      broadcastProgress({ projectId: pid, chapterId: cid, stage: "completed", status: "completed" });
      chapterRepo.updateStatus(cid, "chapter_ready");
      cfg.db.prepare("UPDATE pipeline_runs SET status='completed', finished_at=? WHERE run_id=?")
        .run(now(), runId);
      console.log(`[GraphPipeline] ${cid} completed: ${sceneCount} scenes`);
    } catch (err: any) {
      const msg = err?.message ?? String(err);
      const isCancelled = msg.includes("ABORTED");
      broadcastProgress({ projectId: pid, chapterId: cid, stage: "failed", status: "failed", message: msg });
      chapterRepo.updateStatus(cid, isCancelled ? "cancelled" : "failed");
      cfg.db.prepare("UPDATE pipeline_runs SET status=?, finished_at=?, error_message=? WHERE run_id=?")
        .run(isCancelled ? "cancelled" : "failed", now(), msg.slice(0, 500), runId);
      console.error(`[GraphPipeline] ${cid} ${isCancelled ? "cancelled" : "failed"}:`, msg);
    } finally {
      runningGraphs.delete(cid);
    }
  });

  // POST /graph-pipeline/projects/:projectId/chapters/:chapterId/cancel
  router.post("/projects/:projectId/chapters/:chapterId/cancel", (req: Request, res: Response) => {
    const cid = param(req, "chapterId");
    const ac = runningGraphs.get(cid);
    if (!ac) return res.status(404).json({ error: "No running graph pipeline for this chapter" });
    ac.abort();
    res.json({ chapterId: cid, status: "cancelling", engine: "langgraph" });
  });

  return router;
}
