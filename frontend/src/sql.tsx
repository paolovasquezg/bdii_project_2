import React, { useEffect, useMemo, useState } from "react"
import { Database, Play, Table2, CheckCircle, AlertCircle, ChevronRight, ChevronDown, X, Upload, Wand2 } from "lucide-react"
import { Button } from "../src/components/button.tsx"
import { Card } from "../src/components/card.tsx"
import { ScrollArea } from "../src/components/scroll.tsx"
import { loadTables, execQuery, uploadMedia, createTableFromCsv } from "./data/data"

const SQLQueryInterface = () => {
  const apiBase = (import.meta.env.VITE_API_URL as string | undefined) || "http://127.0.0.1:8000"
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<any[]>([])
  const [isExecuting, setIsExecuting] = useState(false)
  const [tables, set_tables] = useState<any>({})
  const [selectedTable, setSelectedTable] = useState<string>("")
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})

  const [success, setsuccess] = useState<boolean>(false)
  const [error, seterror] = useState<boolean>(false)
  const [message, setmessage] = useState<string>("")
  const [lastIO, setLastIO] = useState<any | null>(null)
  const [lastPlan, setLastPlan] = useState<any | null>(null)
  const [showIO, setShowIO] = useState(false)
  const [showPlan, setShowPlan] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadPath, setUploadPath] = useState<string>("")
  const [uploadMsg, setUploadMsg] = useState<string>("")
  const [uploadError, setUploadError] = useState<string>("")
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [lastResult, setLastResult] = useState<any | null>(null)
  const [csvTable, setCsvTable] = useState<string>("")
  const [csvMsg, setCsvMsg] = useState<string>("")
  const [csvErr, setCsvErr] = useState<string>("")
  const [csvProcessing, setCsvProcessing] = useState(false)
  const [csvCount, setCsvCount] = useState<number | null>(null)

  const executeQuery = async () => {
    setIsExecuting(true)
    seterror(false)
    setsuccess(false)
    const response = await execQuery({ content: query })
    console.log("Query response:", response)
    const tables = await loadTables()

    if (response.success) {
      if (response.data.ok) {
        const first = (response.data.results && response.data.results[0]) || {}
        const dataRows = Array.isArray(first?.data) ? first.data : []
        setResults(dataRows)
        setLastIO(first?.meta?.io || null)
        setLastPlan(first?.plan || null)
        setLastResult(first || null)
        setmessage(`Executed: ${first?.meta?.time_ms ?? response.data?.stats?.time_ms ?? ""} ms`)
        setsuccess(true)
      } else {
        setmessage(
          `Executed: ${response.data?.stats?.time_ms ?? ""} ms\n` +
            `Error: ${response.data.results?.[0]?.error?.message || "Error"}`
        )
        seterror(true)
        setLastIO(null)
        setLastPlan(null)
        setLastResult(null)
        setResults([])
      }
    } else {
      seterror(true)
      setmessage("Ocurrió un error, inténtelo de nuevo")
      setLastIO(null)
      setLastPlan(null)
      setLastResult(null)
      setResults([])
    }

    if (tables.success) {
      set_tables(tables.data)
    }

    setIsExecuting(false)
  }

  const fetchTables = async () => {
    const response = await loadTables()
    if (response.success) {
      set_tables(response.data)
    } else {
      seterror(true)
      setmessage("Ocurrió un error, inténtelo de nuevo")
    }
  }

  useEffect(() => {
    fetchTables()
  }, [])

  useEffect(() => {
    if (tables && typeof tables === "object" && !selectedTable) {
      const names = Object.keys(tables)
      if (names.length > 0) setSelectedTable(names[0])
    }
  }, [tables, selectedTable])

  const isKnn = useMemo(() => {
    return (lastPlan?.action === "knn") || false
  }, [lastPlan])

  const similarityRows = useMemo(() => {
    if (!isKnn || results.length === 0) return []
    return results.map((r) => {
      const sim = (r as any).similarity ?? (r as any).score ?? null
      return { ...r, similarity: sim }
    })
  }, [results, isKnn])

  const [viewerSrc, setViewerSrc] = useState<string | null>(null)
  const [viewerKind, setViewerKind] = useState<"image" | "audio" | "text" | null>(null)
  const [viewerTitle, setViewerTitle] = useState<string | null>(null)
  const [sidebarWidth, setSidebarWidth] = useState<number>(260)
  const [resizing, setResizing] = useState<boolean>(false)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    setSelectedFile(f || null)
    setUploadError("")
    setUploadMsg("")
    setCsvErr("")
    setCsvMsg("")
  }

  const handleMediaUpload = async () => {
    if (!selectedFile) {
      setUploadError("Selecciona un archivo primero.")
      return
    }
    setUploading(true)
    setUploadError("")
    setUploadMsg("")
    const resp = await uploadMedia(selectedFile)
    if (resp.success) {
      setUploadPath(resp.data?.path || "")
      setUploadMsg(`Subido: ${resp.data?.path}`)
    } else {
      setUploadError(`Error al subir: ${resp.error}`)
    }
    setUploading(false)
  }

  const handleCsvUpload = async () => {
    if (!selectedFile || !csvTable.trim()) {
      setCsvErr("Selecciona CSV y nombre de tabla")
      setCsvMsg("")
      return
    }
    setCsvErr("")
    setCsvMsg("Procesando...")
    setCsvProcessing(true)
    setCsvCount(null)
    const resp = await createTableFromCsv(selectedFile, csvTable.trim())
    setCsvProcessing(false)
    if (resp.success) {
      const count =
        (resp.data?.result?.results?.[0]?.meta?.affected as number | undefined) ??
        (resp.data?.result?.count as number | undefined) ??
        null
      if (count !== null) setCsvCount(count)
      setCsvMsg(`Tabla ${csvTable} creada desde CSV${count ? ` (${count} filas)` : ""}`)
      setCsvErr("")
      fetchTables()
    } else {
      setCsvErr(`Error: ${resp.error}`)
      setCsvMsg("")
    }
  }

  const defaultImgPath = "/app/backend/runtime/uploads/<imagen>"
  const defaultAudioPath = "/app/backend/runtime/uploads/<audio>"
  const defaultCsvPath = "/app/backend/runtime/uploads/csv/<archivo>"

  const insertTemplate = (kind: "img" | "audio" | "text") => {
    const tbl = selectedTable || "<tabla>"
    if (kind === "img") {
      // Si no subiste nada, usa el path local por defecto de Screenshots.
      const path = uploadPath || defaultImgPath
      setQuery(`SELECT * FROM ${tbl} WHERE image_path KNN <-> IMG('${path}') LIMIT 5;`)
    } else if (kind === "audio") {
      const path = uploadPath || defaultAudioPath
      setQuery(`SELECT * FROM ${tbl} WHERE file_path KNN <-> AUDIO('${path}') LIMIT 5;`)
    } else {
      setQuery(`SELECT * FROM ${tbl} WHERE content KNN <-> 'ruta o texto de consulta' LIMIT 5;`)
    }
  }

  const insertCsvTemplate = () => {
    const path = uploadPath || defaultCsvPath
    setQuery(`CREATE TABLE <tabla> FROM '${path}';`)
  }

  // drag-to-resize sidebar
  useEffect(() => {
    const onMove = (ev: MouseEvent) => {
      if (!resizing) return
      const next = Math.min(Math.max(ev.clientX, 200), 480)
      setSidebarWidth(next)
      ev.preventDefault()
    }
    const onUp = () => setResizing(false)
    window.addEventListener("mousemove", onMove)
    window.addEventListener("mouseup", onUp)
    return () => {
      window.removeEventListener("mousemove", onMove)
      window.removeEventListener("mouseup", onUp)
    }
  }, [resizing])

  return (
    <div className="flex min-h-screen bg-background">
      <aside
        className="border-r border-border bg-sidebar flex flex-col sticky top-0 h-screen"
        style={{ width: sidebarWidth }}
      >
        <div className="p-4 border-b border-sidebar-border">
          <div className="flex items-center gap-2 text-sidebar-foreground">
            <Database className="h-5 w-5" />
            <h2 className="font-semibold text-sm">Tables</h2>
          </div>
        </div>

        <ScrollArea className="flex-1">
          <div className="p-2 space-y-1">
            {Object.keys(tables || {}).map((tableName) => {
              const isActive = tableName === selectedTable
              const isOpen = !!expanded[tableName]
              const meta = (tables as any)[tableName] || {}
              const rel = meta.relation || {}
              const idxs = meta.indexes || {}
              const miniRows = Object.keys(rel).map((col) => {
                const spec = rel[col] || {}
                const type = spec.type + (spec.length ? `(${spec.length})` : "")
                let ix = idxs[col]
                if (!ix && spec.key === "primary") ix = idxs["primary"]
                const isPK = spec.key === "primary"
                const ixType = (ix?.index as string | undefined) || undefined
                const ixFile = (ix?.filename as string | undefined) || undefined
                return { name: col, type, isPK, ixType, ixFile }
              })
              return (
                <div key={tableName} className="rounded-md">
                  <div
                    className={`flex items-center justify-between px-2 py-1 rounded-md ${
                      isActive ? "bg-accent/40" : ""
                    }`}
                  >
                    <button
                      onClick={() => setExpanded((prev) => ({ ...prev, [tableName]: !isOpen }))}
                      className="p-1 rounded hover:bg-accent/40 text-sidebar-foreground"
                      aria-label={isOpen ? "Collapse" : "Expand"}
                    >
                      {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                    </button>
                    <button
                      onClick={() => setSelectedTable(tableName)}
                      className={`flex-1 flex items-center gap-2 px-2 py-1 rounded-md text-sm transition-colors hover:bg-accent/30 ${
                        isActive ? "text-foreground" : "text-sidebar-foreground"
                      }`}
                    >
                      <Table2 className="h-4 w-4 shrink-0" />
                      <span className="font-mono truncate">{tableName}</span>
                    </button>
                  </div>
                  {isOpen && miniRows.length > 0 && (
                    <div className="pl-7 pr-2 pb-2 space-y-1 mr-6">
                      {miniRows.map((r) => {
                        const badgeBase = "inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium"
                        const ixColor =
                          r.ixType === "hash"
                            ? "bg-blue-500/15 text-blue-400 border border-blue-500/30"
                            : r.ixType === "bplus"
                              ? "bg-purple-500/15 text-purple-400 border border-purple-500/30"
                              : r.ixType === "rtree"
                                ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                                : r.ixType === "heap"
                                  ? "bg-amber-500/15 text-amber-400 border border-amber-500/30"
                                  : r.ixType === "sequential"
                                    ? "bg-indigo-500/15 text-indigo-400 border border-indigo-500/30"
                                    : r.ixType === "isam"
                                      ? "bg-rose-500/15 text-rose-400 border border-rose-500/30"
                                      : "bg-muted text-muted-foreground border border-border"
                        return (
                          <div
                            key={r.name}
                            className="flex items-center justify-between gap-2 text-xs text-muted-foreground"
                          >
                            <span className="font-mono truncate">{r.name}</span>
                            <span className="truncate flex items-center gap-1">
                              <span className="truncate">{r.type}</span>
                              {r.isPK && (
                                <span
                                  className={`${badgeBase} bg-sky-500/15 text-sky-400 border border-sky-500/30`}
                                  title="Primary Key"
                                >
                                  PK
                                </span>
                              )}
                              {r.ixType && (
                                <span className={`${badgeBase} ${ixColor}`} title={r.ixFile || "Index file"}>
                                  {r.ixType}
                                </span>
                              )}
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </ScrollArea>
      </aside>

      <div
        className={`w-1 hover:bg-foreground/30 cursor-col-resize transition-colors ${
          resizing ? "bg-foreground/50" : ""
        }`}
        onMouseDown={(e) => {
          e.preventDefault()
          setResizing(true)
        }}
        title="Arrastra para cambiar el ancho"
      />

      <main className="flex-1 flex flex-col overflow-hidden">
        <div className="border-b border-border bg-card">
          <div className="p-4">
            <div className="flex items-start gap-3">
              <div className="flex-1">
                <div className="mb-3 flex flex-wrap items-center gap-2">
                  <label className="text-xs font-semibold text-muted-foreground">Subir archivo</label>
                  <input
                    type="file"
                    accept="image/*,audio/*,.csv,text/csv"
                    onChange={handleFileSelect}
                    className="text-xs"
                  />
                  <Button size="sm" onClick={handleMediaUpload} disabled={uploading}>
                    <Upload className="h-4 w-4 mr-1" />
                    {uploading ? "Subiendo..." : "Subir archivo"}
                  </Button>
                  {uploadPath && (
                    <span className="text-xs font-mono text-foreground truncate max-w-xs" title={uploadPath}>
                      {uploadPath}
                    </span>
                  )}
                </div>
                {(uploadMsg || uploadError) && (
                  <p className={`text-xs ${uploadError ? "text-red-500" : "text-emerald-500"}`}>
                    {uploadError || uploadMsg}
                  </p>
                )}
                <div className="mb-3 flex flex-wrap gap-2">
                  <Button size="sm" variant="outline" onClick={() => insertTemplate("img")}>
                    <Wand2 className="h-4 w-4 mr-1" />
                    Plantilla IMG KNN
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => insertTemplate("audio")}>
                    <Wand2 className="h-4 w-4 mr-1" />
                    Plantilla AUDIO KNN
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => insertTemplate("text")}>
                    <Wand2 className="h-4 w-4 mr-1" />
                    Plantilla TEXTO KNN
                  </Button>
                  <Button size="sm" variant="outline" onClick={insertCsvTemplate}>
                    <Wand2 className="h-4 w-4 mr-1" />
                    Plantilla CSV
                  </Button>
                </div>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter your SQL statement..."
                  className="w-full min-h-[120px] bg-background border border-input rounded-lg px-4 py-3 font-mono text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring resize-none"
                />
                {(success || error) && (
                  <div
                    className={`mt-3 flex items-start gap-3 rounded-lg border px-4 py-3 animate-in fade-in slide-in-from-top-2 duration-300 ${
                      success
                        ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                        : "bg-red-500/10 border-red-500/20 text-red-400"
                    }`}
                  >
                    {success ? (
                      <CheckCircle className="h-5 w-5 shrink-0 mt-0.5" />
                    ) : (
                      <AlertCircle className="h-5 w-5 shrink-0 mt-0.5" />
                    )}
                    <div className="flex-1 space-y-1">
                      <p className="text-sm font-medium leading-relaxed whitespace-pre-line">{message}</p>
                    </div>
                  </div>
                )}
              </div>
              <Button onClick={executeQuery} disabled={isExecuting} className="shrink-0" size="lg">
                <Play className="h-4 w-4 mr-2" />
                {isExecuting ? "Executing..." : "Execute"}
              </Button>
            </div>
            {(lastIO || lastPlan) && (
              <div className="mt-3 flex items-center gap-2">
                {lastIO && (
                  <Button onClick={() => setShowIO(true)} variant="outline" size="sm">
                    IO
                  </Button>
                )}
                {lastPlan && (
                  <Button onClick={() => setShowPlan(true)} variant="outline" size="sm">
                    Plan
                  </Button>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto overflow-x-hidden p-4 min-h-0">
          {results.length > 0 ? (
          <div className="h-full overflow-hidden flex flex-col gap-3">
              {isKnn && (
                <Card className="p-4">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-semibold text-foreground">Top-k similitudes</h3>
                    {uploadPath && <span className="text-xs text-muted-foreground truncate max-w-sm">{uploadPath}</span>}
                  </div>
                  {similarityRows.length > 0 ? (
                    <ScrollArea className="max-h-[50vh] min-h-[200px] pr-2">
                      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                        {similarityRows.map((r, i) => (
                          <div
                            key={i}
                            className="rounded-lg border border-border p-3 bg-muted/30 cursor-pointer hover:border-foreground/50"
                            onClick={() => {
                              const pImg = (r as any).image_path as string | undefined
                              const pAud = (r as any).file_path as string | undefined
                              const pText = (r as any).content as string | undefined
                              if (pImg) {
                                  setViewerSrc(pImg)
                                  setViewerKind("image")
                                  setViewerTitle((r as any).title || (r as any).file_name || "Imagen")
                              } else if (pAud) {
                                  setViewerSrc(pAud)
                                  setViewerKind("audio")
                                  setViewerTitle((r as any).file_name || "Audio")
                              } else if (pText) {
                                  setViewerSrc(pText)
                                  setViewerKind("text")
                                  setViewerTitle((r as any).title || (r as any).id || "Texto")
                              }
                            }}
                            title="Click para ver imagen (si hay image_path)"
                          >
                            <div className="text-xs text-muted-foreground mb-1">#{i + 1}</div>
                            <div className="text-sm font-semibold text-foreground">
                              {(r.title as string) || (r.file_name as string) || (r.content as string) || r.id || "row"}
                            </div>
                            {typeof r.similarity !== "undefined" && r.similarity !== null && (
                              <div className="text-xs text-emerald-500 font-mono">
                                sim: {Number(r.similarity).toFixed(6)}
                              </div>
                            )}
                            <div className="text-xs text-muted-foreground break-all">
                              {Object.entries(r)
                                .filter(([k]) => k !== "similarity")
                                .slice(0, 3)
                                .map(([k, v]) => `${k}: ${v}`)
                                .join(" • ")}
                            </div>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  ) : (
                    <p className="text-xs text-muted-foreground">Sin resultados (0 filas retornadas).</p>
                  )}
                </Card>
              )}
              <Card className="h-full min-h-0 overflow-hidden flex flex-col">
                <div className="border-b border-border px-4 py-3">
                  <h3 className="text-sm font-semibold text-foreground">Results ({results.length} rows)</h3>
                </div>

                <ScrollArea className="flex-1">
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead className="sticky top-0 bg-muted/50 backdrop-blur-sm">
                        <tr>
                          {Object.keys(results[0]).map((key) => (
                            <th
                              key={key}
                              className="text-left px-4 py-3 text-xs font-semibold text-muted-foreground uppercase tracking-wider border-b border-border"
                            >
                              {key}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {results.map((row, rowIndex) => (
                          <tr key={rowIndex} className="border-b border-border hover:bg-accent/50 transition-colors">
                            {Object.values(row).map((value, colIndex) => (
                              <td key={colIndex} className="px-4 py-3 text-sm text-foreground font-mono">
                                {String(value)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </ScrollArea>
              </Card>
            </div>
          ) : lastResult ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center space-y-2">
                <Database className="h-12 w-12 text-muted-foreground mx-auto mb-2" />
                <h3 className="text-lg font-semibold text-foreground">Consulta ejecutada sin filas</h3>
                <p className="text-sm text-muted-foreground">
                  {isKnn ? "kNN no devolvió vecinos." : "La consulta no retornó filas."}
                </p>
                {isKnn && uploadPath && (
                  <p className="text-xs text-muted-foreground font-mono break-all">{uploadPath}</p>
                )}
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <Database className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-foreground mb-2">No data</h3>
                <p className="text-sm text-muted-foreground">Execute a query to see the results here</p>
              </div>
            </div>
          )}
        </div>
      </main>

      {showIO && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm bg-black/20 p-4"
          onClick={() => setShowIO(false)}
        >
          <Card
            className="w-full max-w-2xl max-h-[80vh] flex flex-col shadow-2xl bg-white"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="border-b border-border px-6 py-4 flex items-center justify-between shrink-0">
              <h3 className="text-lg font-semibold text-card-foreground">IO Stats</h3>
              <Button
                onClick={() => setShowIO(false)}
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0 hover:bg-accent"
              >
                <X className="h-4 w-4" />
                <span className="sr-only">Close</span>
              </Button>
            </div>
            <ScrollArea className="flex-1">
              <div className="p-6">
                {lastIO ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border">
                          <th className="text-left px-3 py-3 text-xs font-semibold text-muted-foreground uppercase">
                            Tipo
                          </th>
                          <th className="text-left px-3 py-3 text-xs font-semibold text-muted-foreground uppercase">
                            Read
                          </th>
                          <th className="text-left px-3 py-3 text-xs font-semibold text-muted-foreground uppercase">
                            Write
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(lastIO).map(([k, v]: any) => {
                          const rc =
                            typeof v === "object" && v ? ((v as any).read_count ?? (v as any).readCount ?? 0) : 0
                          const wc =
                            typeof v === "object" && v ? ((v as any).write_count ?? (v as any).writeCount ?? 0) : 0
                          return (
                            <tr key={k as string} className="border-b border-border hover:bg-accent/50">
                              <td className="px-3 py-3 font-mono text-card-foreground">{k as string}</td>
                              <td className="px-3 py-3 text-card-foreground">{rc}</td>
                              <td className="px-3 py-3 text-card-foreground">{wc}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No IO data.</p>
                )}
              </div>
            </ScrollArea>
          </Card>
        </div>
      )}

      {showPlan && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm bg-black/20 p-4"
          onClick={() => setShowPlan(false)}
        >
          <Card
            className="w-full max-w-2xl max-h-[80vh] flex flex-col shadow-2xl bg-white"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="border-b border-border px-6 py-4 flex items-center justify-between shrink-0">
              <h3 className="text-lg font-semibold text-card-foreground">Execution Plan</h3>
              <Button
                onClick={() => setShowPlan(false)}
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0 hover:bg-accent"
              >
                <X className="h-4 w-4" />
                <span className="sr-only">Close</span>
              </Button>
            </div>
            <ScrollArea className="flex-1">
              <div className="p-6 space-y-4">
                {lastPlan ? (
                  <>
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      {lastPlan?.action && (
                        <div className="space-y-1">
                          <span className="text-xs text-muted-foreground uppercase font-semibold">Action</span>
                          <div className="font-mono text-card-foreground">{String(lastPlan.action)}</div>
                        </div>
                      )}
                      {lastPlan?.table && (
                        <div className="space-y-1">
                          <span className="text-xs text-muted-foreground uppercase font-semibold">Table</span>
                          <div className="font-mono text-card-foreground">{String(lastPlan.table)}</div>
                        </div>
                      )}
                      {typeof lastPlan?.columns !== "undefined" && (
                        <div className="col-span-2 space-y-1">
                          <span className="text-xs text-muted-foreground uppercase font-semibold">Columns</span>
                          <div className="font-mono text-card-foreground">
                            {lastPlan.columns === null
                              ? "null"
                              : Array.isArray(lastPlan.columns)
                                ? lastPlan.columns.join(", ")
                                : String(lastPlan.columns)}
                          </div>
                        </div>
                      )}
                      {typeof lastPlan?.where !== "undefined" && (
                        <div className="col-span-2 space-y-1">
                          <span className="text-xs text-muted-foreground uppercase font-semibold">Where</span>
                          <div className="font-mono text-card-foreground">
                            {lastPlan.where === null
                              ? "null"
                              : typeof lastPlan.where === "object"
                                ? JSON.stringify(lastPlan.where)
                                : String(lastPlan.where)}
                          </div>
                        </div>
                      )}
                    </div>
                    <div className="space-y-2">
                      <div className="text-xs text-muted-foreground uppercase font-semibold">Raw Data</div>
                      <pre className="text-xs bg-muted/40 border border-border rounded-lg p-4 overflow-auto max-h-64 font-mono text-card-foreground">
                        {JSON.stringify(lastPlan, null, 2)}
                      </pre>
                    </div>
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground">No plan available.</p>
                )}
              </div>
            </ScrollArea>
          </Card>
        </div>
      )}

      {viewerSrc && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm bg-black/30 p-4"
          onClick={() => {
            setViewerSrc(null)
            setViewerTitle(null)
          }}
        >
          <Card
            className="w-full max-w-6xl max-h-[90vh] flex flex-col shadow-2xl bg-white"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="border-b border-border px-6 py-4 flex items-center justify-between shrink-0">
              <div>
                <h3 className="text-lg font-semibold text-card-foreground">Previsualización</h3>
                {viewerTitle && <p className="text-xs text-muted-foreground">{viewerTitle}</p>}
              </div>
              <Button
                onClick={() => {
                  setViewerSrc(null)
                  setViewerTitle(null)
                }}
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0 hover:bg-accent"
              >
                <X className="h-4 w-4" />
                <span className="sr-only">Close</span>
              </Button>
            </div>
            <ScrollArea className="flex-1 p-4 max-h-[80vh]">
              {viewerKind === "image" && (() => {
                const resolved = viewerSrc.startsWith("http")
                  ? viewerSrc
                  : `${apiBase}/file?path=${encodeURIComponent(viewerSrc)}`
                return (
                  <img
                    src={resolved}
                    alt="preview"
                    className="max-h-[75vh] mx-auto object-contain border border-border rounded-lg"
                  />
                )
              })()}
              {viewerKind === "audio" && (() => {
                const resolved = viewerSrc.startsWith("http")
                  ? viewerSrc
                  : `${apiBase}/file?path=${encodeURIComponent(viewerSrc)}`
                return (
                  <audio controls className="w-full">
                    <source src={resolved} />
                    Tu navegador no soporta audio.
                  </audio>
                )
              })()}
              {viewerKind === "text" && (
                <pre className="whitespace-pre-wrap text-sm text-card-foreground bg-muted/30 border border-border rounded-lg p-4 max-h-[70vh] overflow-auto">
                  {viewerSrc}
                </pre>
              )}
              {viewerKind !== "text" && (
                <div className="mt-2 text-xs text-muted-foreground break-all text-center font-mono">{viewerSrc}</div>
              )}
            </ScrollArea>
          </Card>
        </div>
      )}
    </div>
  )
}

export default SQLQueryInterface
