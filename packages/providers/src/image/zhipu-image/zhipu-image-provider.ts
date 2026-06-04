import OpenAI from "openai";
import type {
  ImageProvider,
  ImageGenerationRequest,
  ImageGenerationResult,
} from "../interfaces.js";

export interface ZhipuImageProviderConfig {
  apiKey: string;
}

export class ZhipuImageProvider implements ImageProvider {
  readonly name = "zhipu-image";
  private client: OpenAI;

  constructor(config: ZhipuImageProviderConfig) {
    this.client = new OpenAI({
      apiKey: config.apiKey,
      baseURL: "https://open.bigmodel.cn/api/paas/v4",
    });
  }

  async generateImage(request: ImageGenerationRequest): Promise<ImageGenerationResult> {
    const model = request.model ?? "cogview-4-250304";
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
    return ["cogview-4-250304", "cogview-4", "cogview-3-flash"];
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
