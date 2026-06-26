import { Router } from "express";
import type { Request, Response } from "express";
import fs from "node:fs";
import path from "node:path";
import { AgnesVideoProvider } from "@novel2gal/providers";
import type { VideoProvider, VideoGenerationRequest } from "@novel2gal/providers";
import { config } from "../config/index.js";

function readModelConfig() {
  try {
    return JSON.parse(fs.readFileSync(path.join(config.dataDir, "config", "models.json"), "utf-8"));
  } catch {
    return { apiKey: "" };
  }
}

function createVideoProvider(): VideoProvider | null {
  const cfg = readModelConfig();
  const apiKey = cfg.apiKey || process.env.OPENAI_API_KEY;
  if (!apiKey) return null;
  return new AgnesVideoProvider({ apiKey, baseUrl: cfg.baseUrl });
}

export function createVideoRoutes() {
  const router = Router();

  // POST /videos/generate - Create async video generation task
  router.post("/generate", async (req: Request, res: Response) => {
    const provider = createVideoProvider();
    if (!provider) {
      return res.status(503).json({ error: "Video provider not configured (missing API key)" });
    }

    const request: VideoGenerationRequest = {
      prompt: req.body.prompt,
      negativePrompt: req.body.negativePrompt,
      width: req.body.width,
      height: req.body.height,
      numFrames: req.body.numFrames,
      frameRate: req.body.frameRate,
      model: req.body.model,
      imageUrl: req.body.imageUrl,
      imageUrls: req.body.imageUrls,
      mode: req.body.mode,
      seed: req.body.seed,
      inferenceSteps: req.body.inferenceSteps,
    };

    if (!request.prompt) {
      return res.status(400).json({ error: "prompt is required" });
    }

    try {
      const task = await provider.createTask(request);
      res.json(task);
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  // GET /videos/task/:taskId - Poll video task status
  router.get("/task/:taskId", async (req: Request, res: Response) => {
    const provider = createVideoProvider();
    if (!provider) {
      return res.status(503).json({ error: "Video provider not configured" });
    }

    try {
      const task = await provider.checkTaskStatus(String(req.params.taskId));

      // Download video to disk if completed
      if (task.status === "completed" && task.videoUrl) {
        const projectId = req.query.projectId as string | undefined;
        const sceneId = req.query.sceneId as string | undefined;
        if (projectId && sceneId) {
          const vidDir = path.join(config.dataDir, "projects", projectId, "preview", sceneId);
          fs.mkdirSync(vidDir, { recursive: true });
          task.request = { ...task.request, prompt: task.request.prompt };
          // Store the download path hint
          (task as any).localDir = vidDir;
        }
      }

      res.json(task);
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  // GET /videos/providers - List available video providers
  router.get("/providers", (_req: Request, res: Response) => {
    res.json({
      providers: [
        {
          name: "agnes-video",
          models: ["agnes-video-v2.0"],
          defaultSize: { width: 1152, height: 768 },
          capabilities: ["text-to-video", "image-to-video", "multi-image", "keyframes"],
          pricing: "$0/s (free promotional)",
        },
      ],
    });
  });

  return router;
}
