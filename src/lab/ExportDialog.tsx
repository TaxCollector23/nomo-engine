import { useEffect, useMemo, useRef, useState } from "react";

import { FOLDERS, downloadBlob, readme, zipFiles, type ExportFile } from "../planner/exporters";

const FOLDER_TITLES: Record<string, string> = {
  "1-summary": "Summary", "2-data": "Data", "3-figures": "Figures", "4-launch-configs": "Launch configs", "5-paper-materials": "Paper materials",
};

export default function ExportDialog({ files, root, onClose }: { files: ExportFile[]; root: string; onClose: () => void }) {
  const [picked, setPicked] = useState<Set<string>>(new Set(files.map((f) => f.id)));
  const [preview, setPreview] = useState<ExportFile | null>(null);
  const [busy, setBusy] = useState(false);
  const dialog = useRef<HTMLDivElement>(null);
  const folders = useMemo(() => Object.keys(FOLDERS).filter((k) => files.some((f) => f.folder === k)), [files]);

  useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", key);
    dialog.current?.focus();
    return () => window.removeEventListener("keydown", key);
  }, [onClose]);

  const one = (f: ExportFile) => downloadBlob(new Blob([f.content], { type: f.mime }), f.name);
  const zip = async () => {
    setBusy(true);
    try {
      const chosen = files.filter((f) => picked.has(f.id));
      downloadBlob(await zipFiles(chosen, root), `${root}.zip`);
    } finally { setBusy(false); }
  };
  const toggle = (id: string) => setPicked((p) => { const n = new Set(p); if (n.has(id)) n.delete(id); else n.add(id); return n; });
  const toggleFolder = (k: string) => {
    const ids = files.filter((f) => f.folder === k).map((f) => f.id);
    const allOn = ids.every((i) => picked.has(i));
    setPicked((p) => { const n = new Set(p); ids.forEach((i) => (allOn ? n.delete(i) : n.add(i))); return n; });
  };
  const chosen = files.filter((f) => picked.has(f.id));

  return (
    <div className="lab-modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="lab-modal" role="dialog" aria-modal="true" aria-labelledby="export-title" tabIndex={-1} ref={dialog}>
        <div className="lab-modal-head">
          <div>
            <p className="section-kicker">Export</p>
            <h2 id="export-title">Take this plan with you</h2>
            <p className="lab-muted">Download any file on its own, or pick several and get one zip. Every export includes a README.</p>
          </div>
          <button type="button" className="lab-close" onClick={onClose} aria-label="Close export">×</button>
        </div>
        <div className="lab-modal-body">
          <div className="lab-export-list">
            {folders.map((k) => {
              const fs = files.filter((f) => f.folder === k);
              const allOn = fs.every((f) => picked.has(f.id));
              return (
                <section key={k} className="lab-export-folder">
                  <label className="lab-export-folder-head">
                    <input type="checkbox" checked={allOn} onChange={() => toggleFolder(k)} />
                    <span><b>{FOLDER_TITLES[k]}</b> <code>{k}/</code></span>
                  </label>
                  <p className="lab-muted">{FOLDERS[k]}</p>
                  <ul>
                    {fs.map((f) => (
                      <li key={f.id} className={preview?.id === f.id ? "is-previewing" : ""}>
                        <label>
                          <input type="checkbox" checked={picked.has(f.id)} onChange={() => toggle(f.id)} />
                          <span className="lab-export-name">{f.name}</span>
                        </label>
                        <span className="lab-export-desc">{f.description}</span>
                        <span className="lab-export-actions">
                          <button type="button" className="lab-link" onClick={() => setPreview(preview?.id === f.id ? null : f)}>
                            {preview?.id === f.id ? "Hide" : "Preview"}
                          </button>
                          <button type="button" className="lab-link" onClick={() => one(f)}>Download</button>
                        </span>
                      </li>
                    ))}
                  </ul>
                </section>
              );
            })}
          </div>
          <div className="lab-export-side">
            {preview ? (
              <>
                <p className="lab-muted"><code>{preview.folder}/{preview.name}</code></p>
                {preview.mime === "image/svg+xml"
                  ? <div className="lab-export-svg" dangerouslySetInnerHTML={{ __html: preview.content }} />
                  : <pre className="lab-pre">{preview.content.slice(0, 6000)}{preview.content.length > 6000 ? "\n…" : ""}</pre>}
              </>
            ) : (
              <>
                <p className="lab-muted">Zip structure</p>
                <pre className="lab-pre">{readme(chosen, root).split("\n\nFOLDERS")[0]}</pre>
              </>
            )}
          </div>
        </div>
        <div className="lab-modal-foot">
          <span className="lab-muted">{chosen.length} of {files.length} files selected</span>
          <button type="button" className="ui-button ui-button--primary ui-button--default" disabled={!chosen.length || busy} onClick={zip}>
            {busy ? "Preparing…" : `Download ${chosen.length} file${chosen.length === 1 ? "" : "s"} as .zip`}
          </button>
        </div>
      </div>
    </div>
  );
}
