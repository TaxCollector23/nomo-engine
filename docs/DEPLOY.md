# Nomo Engine deployment guide

Nomo is deployed as three cooperating surfaces. Deploying one does not automatically deploy the others.

## Root mode shell and Lab

From the repository root:

```bash
npm ci
npm run build
```

The root Vite app is the source for the `nomo-engine` Vercel project and the stable mode-shell alias
`https://frontend-gray-ten-c3tj1luab7.vercel.app/`. The root `vercel.json` rewrites client-side Lab routes to
`index.html`; it does not run the hosted search API.

## Neuromorphic dashboard

From `frontend/`:

```bash
npm ci
npm run typecheck
npm run build
```

The Next.js app is the source for the separate `nomo-engine-dashboard` Vercel project. Set
`NEXT_PUBLIC_NOMO_API` when using a different API; otherwise the client uses
`https://nomo-engine.onrender.com`. The dashboard cannot produce a search without a reachable API.

## Hosted API

The dashboard's default API is the Render deployment at `https://nomo-engine.onrender.com`. Its server source and
deployment configuration are outside this repository. A deployment of the frontend alone must not be described as a
backend redeploy.

The dependency-free platform store under `nomo-planner/` is a separate project/run/artifact API. It can be run locally
with `PYTHONPATH=. python -m nomo_planner.cli serve`; when exposed to a browser, configure `NOMO_API_TOKEN` and
`NOMO_CORS_ORIGINS` and bind it behind TLS. It is not the source of the existing Render dashboard API, and it is not
multi-tenant until authentication, tenant isolation, backups, and operational monitoring are supplied by deployment.

## Landing page

The separate `nomo-ai` repository is the source for `https://nomoailanding.vercel.app/`. Deploy it only when landing
source changes are pushed. The landing page links to both the mode shell and the neuromorphic dashboard so users can
choose the intended surface.

## Post-deploy checks

1. Request `/` from each changed deployment and verify HTTP 200 and the expected title.
2. Open the mode shell and click every Lab module, including the Neuromorphic chips link.
3. In the dashboard, run a built-in model, wait for `completed`, open a layer, and inspect export capabilities.
4. Check `/docs/SPEC.md`, `/docs/DEPLOY.md`, and `/docs/OBSERVABILITY.md` from the landing page's documentation links.
5. Record the commit, deployment id, URL, build result, and any data-dependent limitation in `PROGRESS.md`.
