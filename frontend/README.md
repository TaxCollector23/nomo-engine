# Nomo dashboard (Next.js 16 · React 19 · TypeScript strict · Tailwind · zustand · react-three-fiber)

    # production builds target https://nomo-engine.onrender.com (.env.production)
    cp .env.example .env.local        # local dev: NEXT_PUBLIC_NOMO_API=http://127.0.0.1:8765
    npm install
    npm run dev                       # or: npm run build && npm start
    # backend, in another shell:  nomo serve --port 8765

Layout (architecture rationale: ../docs/SPEC.md, especially sections 1 and 5)

    src/lib/telemetry/protocol.ts       wire types (mirror of nomo/telemetry/schema.py) + runtime guard
    src/lib/telemetry/client.ts         WebSocket: backoff w/ full jitter, heartbeat + dead-man timer,
                                        resume via ?since=, gap detection -> recycle, snapshot reset
    src/lib/telemetry/store.ts          zustand store; one state update per animation-frame batch
    src/lib/telemetry/useRunTelemetry.ts  binds client -> store with requestAnimationFrame coalescing
    src/lib/pareto.ts                   log/linear axes, robust bounds, unit-cube mapping, SI formatting
    src/components/ParetoFront3D.tsx    single InstancedMesh (O(1) draw calls), click-to-inspect, ASF marker
    src/components/PartitionGraph.tsx   SVG layered layout of the execution stream, labelled crossings
    src/components/RunHUD.tsx           status, HV sparkline, adaptive-pursuit operator probabilities
    src/components/LayerInspector.tsx   selected-layer metrics, locks, and Copilot actions
    src/components/DesignPanel.tsx      selected-design metrics and assumptions
    src/components/CopilotDrawer.tsx    grounded questions and settings actions
    src/components/ExportDrawer.tsx     capability-aware artifact selection and download
    src/app/page.tsx                    catalog-driven run launcher and run list
    src/app/runs/[runId]/page.tsx       live dashboard
    src/app/admin/page.tsx              logs (6 streams, filters, live tail, NDJSON), users, runs, stats
    src/lib/api.ts                      client id, fetch wrapper, cold-start wait

Verified: `npm run typecheck`, `npm run lint`, `npm audit --omit=dev` (zero production vulnerabilities), and `next build` clean. A live built-in
Event-camera/AKD1500 search completed through the hosted dashboard; a complete automated every-control accessibility
matrix remains an explicit roadmap limitation.
