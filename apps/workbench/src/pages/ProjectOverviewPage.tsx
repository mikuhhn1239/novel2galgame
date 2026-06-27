import { useState, useEffect, useRef } from 'react'
import { useParams, Link } from 'react-router'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Play, FileText, Layers, Eye, ListTodo, Download, Sparkles, Package, Zap } from 'lucide-react'
import { useProject, useRunStructure } from '@/hooks/useProjects'
import { useChapters } from '@/hooks/useChapters'
import { projectService } from '@/services/projects'
import { StatusBadge } from '@/components/common/StatusBadge'

export function ProjectOverviewPage() {
  const { projectId } = useParams<{ projectId: string }>()
  const queryClient = useQueryClient()
  const { data: project } = useProject(projectId!)
  const { data: chapters } = useChapters(projectId!)
  const runStructure = useRunStructure(projectId!)

  const [exportStatus, setExportStatus] = useState<string | null>(null)
  const [autoExportLog, setAutoExportLog] = useState<string[]>([])
  const [autoExportRunning, setAutoExportRunning] = useState(false)
  const eventSourceRef = useRef<EventSource | null>(null)

  const exportRenpy = useMutation({
    mutationFn: () => projectService.exportRenpy(projectId!),
    onMutate: () => setExportStatus('导出中...'),
    onSuccess: (data) => setExportStatus(`导出成功: ${data.stats?.generatedFiles?.length ?? 0} 文件`),
    onError: (err: any) => setExportStatus(`导出失败: ${err.message}`),
  })

  const startAutoExport = async () => {
    setAutoExportRunning(true)
    setAutoExportLog(['启动一键处理...'])

    // Listen to SSE
    const es = new EventSource(`/api/projects/${projectId}/progress`)
    eventSourceRef.current = es
    es.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data)
        if (event.status === 'connected') return
        const msg = `[${event.stage}] ${event.status}: ${event.message || ''}`
        setAutoExportLog(prev => [...prev, msg])
        if (event.stage === 'complete') {
          setAutoExportRunning(false)
          es.close()
          queryClient.invalidateQueries({ queryKey: ['chapters', projectId] }
          )
        }
      } catch {}
    }
    es.onerror = () => { setAutoExportRunning(false); es.close() }

    try {
      await projectService.autoExport(projectId!, { model: 'agnes-2.0-flash', maxChapters: 3 })
    } catch (err) {
      setAutoExportLog(prev => [...prev, `错误: ${err instanceof Error ? err.message : err}`])
      setAutoExportRunning(false)
      es.close()
    }
  }

  const generateAssets = useMutation({
    mutationFn: () => projectService.generateAssets(projectId!),
    onMutate: () => setExportStatus('生成资源中...'),
    onSuccess: (data) => setExportStatus(`生成完成: ${data.generated?.length ?? 0} 张图片`),
    onError: (err: any) => setExportStatus(`生成失败: ${err.message}`),
  })

  if (!project) return <div className="p-6 text-muted-foreground">加载中...</div>

  const failedChapters = chapters?.filter((c) => c.status === 'failed') ?? []
  const readyChapters = chapters?.filter((c) => c.status === 'chapter_ready') ?? []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold bg-gradient-to-r from-deep-purple to-[#9333EA] bg-clip-text text-transparent flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-sakura" />
          {project.title}
        </h2>
        <StatusBadge status={project.status} />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <InfoCard label="原始文件" value={project.sourceFileName || '未导入'} />
        <InfoCard label="总章节数" value={String(project.totalChapters)} />
        <InfoCard label="已完成" value={`${readyChapters.length}`} />
        <InfoCard label="失败" value={`${failedChapters.length}`} accent={failedChapters.length > 0} />
      </div>

      {/* Quick Actions */}
      <div className="border border-border rounded-2xl p-5 bg-card shadow-card">
        <h3 className="font-semibold text-deep-purple mb-3 flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-sakura" /> 快捷操作
        </h3>
        <div className="flex flex-wrap gap-3">
          {project.status === 'created' && (
            <button onClick={() => runStructure.mutate()} disabled={runStructure.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-sakura to-lavender text-white rounded-xl font-medium disabled:opacity-50 transition-all hover:shadow-md">
              <Play className="w-4 h-4" />
              {runStructure.isPending ? '解析中...' : '运行结构解析'}
            </button>
          )}
          {project.status === 'structured' && (
            <Link to={`/projects/${projectId}/chapters`}
              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-sakura to-lavender text-white rounded-xl font-medium transition-all hover:shadow-md">
              <FileText className="w-4 h-4" /> 进入章节管理
            </Link>
          )}
          <Link to={`/projects/${projectId}/editor`}
            className="flex items-center gap-2 px-4 py-2 border border-lavender/30 text-deep-purple rounded-xl hover:bg-lavender/10 transition-all">
            <Sparkles className="w-4 h-4" /> 场景编辑器
          </Link>
          <Link to={`/projects/${projectId}/preview`}
            className="flex items-center gap-2 px-4 py-2 border border-lavender/30 text-deep-purple rounded-xl hover:bg-lavender/10 transition-all">
            <Eye className="w-4 h-4" /> 预览
          </Link>

          {/* Export Section */}
          <div className="w-full border-t border-border my-2" />
          <button onClick={() => exportRenpy.mutate()} disabled={exportRenpy.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-emerald-400 to-teal-400 text-white rounded-xl font-medium disabled:opacity-50 transition-all hover:shadow-md">
            <Download className="w-4 h-4" />
            {exportRenpy.isPending ? '导出中...' : '导出 Ren\'Py 项目'}
          </button>
          <button onClick={() => generateAssets.mutate()} disabled={generateAssets.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-amber-400 to-orange-400 text-white rounded-xl font-medium disabled:opacity-50 transition-all hover:shadow-md">
            <Package className="w-4 h-4" />
            {generateAssets.isPending ? '生成中...' : '生成背景/立绘'}
          </button>
          <div className="w-full border-t border-border my-2" />
          <button onClick={startAutoExport} disabled={autoExportRunning}
            className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-violet-500 to-purple-600 text-white rounded-xl font-medium disabled:opacity-50 transition-all hover:shadow-lg">
            <Zap className="w-4 h-4" />
            {autoExportRunning ? '处理中...' : '一键处理 (管线→导出)'}
          </button>
          {exportStatus && (
            <span className="text-sm text-muted-foreground self-center">{exportStatus}</span>
          )}
        </div>

        {/* Auto-export progress log */}
        {autoExportLog.length > 0 && (
          <div className="mt-4 rounded-xl bg-gray-900 p-3 max-h-40 overflow-y-auto font-mono text-xs">
            {autoExportLog.map((line, i) => (
              <p key={i} className="text-green-400">{line}</p>
            ))}
            {autoExportRunning && <p className="text-yellow-400 animate-pulse">处理中...</p>}
          </div>
        )}
      </div>

      {failedChapters.length > 0 && (
        <div className="border border-destructive/50 rounded-2xl p-4 bg-destructive/5">
          <h3 className="font-medium text-destructive mb-2">异常章节</h3>
          <ul className="text-sm space-y-1">
            {failedChapters.map((ch) => (
              <li key={ch.chapterId}>第{ch.index + 1}章 {ch.title}: {ch.lastError || '处理失败'}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function InfoCard({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="border border-border rounded-2xl p-4 bg-card shadow-card">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <p className={`text-lg font-semibold ${accent ? 'text-destructive' : 'text-deep-purple'}`}>{value}</p>
    </div>
  )
}
