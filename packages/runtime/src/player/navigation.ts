import type { VNScript } from "@novel2gal/core";

export interface SceneInfo {
  sceneId: string;
  chapterId: string;
  stepCount: number;
}

export function getSceneList(scripts: VNScript[]): SceneInfo[] {
  return scripts.map((s) => ({
    sceneId: s.sceneId,
    chapterId: s.chapterId,
    stepCount: s.steps.length,
  }));
}

export function findScriptByScene(scripts: VNScript[], sceneId: string): VNScript | undefined {
  return scripts.find((s) => s.sceneId === sceneId);
}

export function findScriptByChapter(scripts: VNScript[], chapterId: string): VNScript[] {
  return scripts.filter((s) => s.chapterId === chapterId);
}
