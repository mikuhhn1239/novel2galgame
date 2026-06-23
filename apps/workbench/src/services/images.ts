import { request } from './api'
import type { ImageGenerationResult } from '@novel2gal/providers'

export interface GenerateImageParams {
  prompt: string;
  negativePrompt?: string;
  width?: number;
  height?: number;
  numImages?: number;
  model?: string;
  style?: string;
  projectId?: string;
  sceneId?: string;
}

export const imageService = {
  generate: (params: GenerateImageParams) =>
    request<ImageGenerationResult>('/images/generate', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  getProviders: () =>
    request<{ providers: Array<{ name: string; models: string[]; defaultSize: { width: number; height: number } }> }>('/images/providers'),
}
