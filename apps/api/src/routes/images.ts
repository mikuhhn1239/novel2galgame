import { Router } from "express";
import type { Request, Response } from "express";
import fs from "node:fs";
import path from "node:path";
import {
  OpenAIImageProvider,
  ZhipuImageProvider,
  SiliconFlowImageProvider,
  AgnesImageProvider,
} from "@novel2gal/providers";
import type { ImageProvider, ImageGenerationRequest } from "@novel2gal/providers";
import { config, readProfilesConfig, resolveModelConfig } from "../config/index.js";

function createImageProvider(): ImageProvider | null {
  const cfg = readProfilesConfig();
  const { profile: profileName, model } = resolveModelConfig("image", cfg);
  const profile = cfg.profiles.find((p) => p.name === profileName);
  if (!profile?.apiKey) return null;

  // Determine image provider type from profile name
  const providerKey = profileName.startsWith("agnes") ? "agnes"
    : profileName === "zhipu" ? "zhipu"
    : profileName === "siliconflow" ? "siliconflow"
    : "openai";

  switch (providerKey) {
    case "zhipu":
      return new ZhipuImageProvider({ apiKey: profile.apiKey });
    case "siliconflow":
      return new SiliconFlowImageProvider({ apiKey: profile.apiKey, defaultModel: model });
    case "agnes":
      return new AgnesImageProvider({ apiKey: profile.apiKey, baseUrl: profile.baseUrl });
    default:
      return new OpenAIImageProvider({
        apiKey: profile.apiKey,
        baseUrl: profile.baseUrl || undefined,
      });
  }
}

export function createImageRoutes() {
  const router = Router();

  // POST /images/generate - Generate image from prompt
  router.post("/generate", async (req: Request, res: Response) => {
    const provider = createImageProvider();
    if (!provider) {
      return res.status(503).json({ error: "Image provider not configured (missing API key)" });
    }

    const request: ImageGenerationRequest = {
      prompt: req.body.prompt,
      negativePrompt: req.body.negativePrompt,
      width: req.body.width,
      height: req.body.height,
      numImages: req.body.numImages ?? 1,
      model: req.body.model,
      style: req.body.style,
    };

    if (!request.prompt) {
      return res.status(400).json({ error: "prompt is required" });
    }

    try {
      const result = await provider.generateImage(request);

      // Optionally save image to disk
      const projectId = req.body.projectId;
      const sceneId = req.body.sceneId;
      if (projectId && sceneId && result.images[0]?.base64) {
        const imgDir = path.join(config.dataDir, "projects", projectId, "preview", sceneId);
        fs.mkdirSync(imgDir, { recursive: true });
        for (let i = 0; i < result.images.length; i++) {
          const img = result.images[i];
          if (img.base64) {
            const fileName = `generated_${Date.now()}_${i}.png`;
            fs.writeFileSync(path.join(imgDir, fileName), Buffer.from(img.base64, "base64"));
            img.metadata = { ...(img.metadata ?? {}), savedPath: path.join(imgDir, fileName) };
          }
        }
      }

      res.json(result);
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  // GET /images/providers - List available image providers
  router.get("/providers", (_req: Request, res: Response) => {
    res.json({
      providers: [
        { name: "agnes-image", models: ["agnes-image-2.1-flash"], defaultSize: { width: 768, height: 1024 } },
        { name: "openai", models: ["gpt-image-1"], defaultSize: { width: 1024, height: 1536 } },
        { name: "zhipu", models: ["cogview-4-250304", "cogview-4", "cogview-3-flash"], defaultSize: { width: 1024, height: 1024 } },
        { name: "siliconflow", models: ["black-forest-labs/FLUX.1-schnell", "stabilityai/stable-diffusion-3-5-large"], defaultSize: { width: 1024, height: 1024 } },
      ],
    });
  });

  return router;
}
