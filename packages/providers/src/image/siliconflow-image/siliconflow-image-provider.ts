import OpenAI from "openai";
import type {
  ImageProvider,
  ImageGenerationRequest,
  ImageGenerationResult,
} from "../interfaces.js";

export interface SiliconFlowImageProviderConfig {
  apiKey: string;
  defaultModel?: string;
}

export class SiliconFlowImageProvider implements ImageProvider {
  readonly name = "siliconflow-image";
  private client: OpenAI;
  private defaultModel: string;

  constructor(config: SiliconFlowImageProviderConfig) {
    this.client = new OpenAI({
      apiKey: config.apiKey,
      baseURL: "https://api.siliconflow.cn/v1",
    });
    this.defaultModel = config.defaultModel ?? "black-forest-labs/FLUX.1-schnell";
  }

  async generateImage(request: ImageGenerationRequest): Promise<ImageGenerationResult> {
    const model = request.model ?? this.defaultModel;
    const size = this.resolveSize(request.width, request.height);

    const response = await this.client.images.generate({
      model,
      prompt: request.prompt,
      n: request.numImages ?? 1,
      size: size as "1024x1024" | "1024x1536" | "1536x1024",
    });

    return {
      images: (response.data ?? []).map((img) => ({
        url: img.url ?? undefined,
        base64: img.b64_json ?? undefined,
        revisedPrompt: img.revised_prompt ?? undefined,
      })),
      model,
      provider: this.name,
    };
  }

  getSupportedModels(): string[] {
    return [
      "black-forest-labs/FLUX.1-schnell",
      "stabilityai/stable-diffusion-3-5-large",
      "stabilityai/stable-diffusion-xl-base-1.0",
    ];
  }

  getDefaultSize(): { width: number; height: number } {
    return { width: 1024, height: 1024 };
  }

  private resolveSize(width?: number, height?: number): string {
    if (width && height) {
      if (width === height) return "1024x1024";
      if (width > height) return "1536x1024";
      return "1024x1536";
    }
    return "1024x1024";
  }
}
