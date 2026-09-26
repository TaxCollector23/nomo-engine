# Nomo dashboard (Next.js 14 · TypeScript strict · Tailwind · zustand · react-three-fiber)

    # production builds target https://nomo-engine.onrender.com (.env.production)
    cp .env.example .env.local        # local dev: NEXT_PUBLIC_NOMO_API=http://127.0.0.1:8765
    npm install
    npm run dev                       # or: npm run build && npm start
    # backend, in another shell:  nomo serve --port 8765

Layout (architecture rationale: docs/SPEC.md §10)

    src/lib/telemetry/protocol.ts       wire types (mirror of nomo/telemetry/schema.py) + runtime guard
    src/lib/telemetry/client.ts         WebSocket: backoff w/ full jitter, heartbeat + dead-man timer,
                                        resume via ?since=, gap detection -> recycle, snapshot reset
    src/lib/telemetry/store.ts          zustand store; one state update per animation-frame batch
    src/lib/telemetry/useRunTelemetry.ts  binds client -> store with requestAnimationFrame coalescing
    src/lib/pareto.ts                   log/linear axes, robust bounds, unit-cube mapping, SI formatting
    src/components/ParetoFront3D.tsx    single InstancedMesh (O(1) draw calls), click-to-inspect, ASF marker
    src/components/PartitionGraph.tsx   SVG layered layout of the execution stream, labelled crossings
    src/components/RunHUD.tsx           status, HV sparkline, adaptive-pursuit operator probabilities
    src/components/CandidatePanel.tsx   metrics + gene codes of the selected design
    src/app/page.tsx                    catalog-driven run launcher and run list
    src/app/runs/[runId]/page.tsx       live dashboard
    src/app/admin/page.tsx              logs (6 streams, filters, live tail, NDJSON), users, runs, stats
    src/lib/api.ts                      client id, fetch wrapper, cold-start wait

Verified: `tsc --noEmit` (strict, noUncheckedIndexedAccess) clean; `next build` clean; headless-Chromium
end-to-end against a live `nomo serve` (launch -> stream -> completion -> reload/resume) with zero console errors.
