import { lazy, Suspense, useEffect, useState, type ReactNode } from "react";

import { Button } from "./components/ui/button";

const LabPage = lazy(() => import("./lab/LabPage"));

const ENGINE_URL = "https://nomo-engine-dashboard.vercel.app/";
const ENGINE_REPO_URL = "https://github.com/TaxCollector23/nomo-engine";
const ENGINE_README_URL = "https://github.com/TaxCollector23/nomo-engine#readme";
const ENGINE_SPEC_URL = "https://github.com/TaxCollector23/nomo-engine/blob/main/docs/SPEC.md";
const ENGINE_NIR_URL = "https://github.com/TaxCollector23/nomo-engine/blob/main/docs/SPEC.md#4-nir-export";
const ENGINE_DEPLOY_URL = "https://github.com/TaxCollector23/nomo-engine/blob/main/docs/DEPLOY.md";
const ENGINE_OBSERVABILITY_URL = "https://github.com/TaxCollector23/nomo-engine/blob/main/docs/OBSERVABILITY.md";
const NOMO_REPO_URL = "https://github.com/TaxCollector23/nomo-ai";
const HOSTED_BACKEND_URL = "https://nomo-engine.onrender.com/";

const BIBTEX_PLACEHOLDER = `@article{nomo_hardware_aware_partitioning,
  title = {Hardware-Aware NSGA-II Partitioning for Hybrid Spiking-Continuous Computational Graphs},
  author = {Nomo AI research group},
  note = {Placeholder. Preprint not published},
  year = {2026}
}`;

const navLinks = [
  { href: "/lab", label: "Lab" },
  { href: "/research", label: "Research & Papers" },
  { href: "/architecture", label: "Architecture (NIR)" },
  { href: "/benchmarks", label: "Benchmarks" },
  { href: "/docs", label: "Documentation" },
];

const externalLinkProps = { target: "_blank" as const, rel: "noreferrer" as const };

function getLinkProps(href: string) {
  return /^(https?:|mailto:)/.test(href) ? externalLinkProps : {};
}

function ExternalArrow() {
  return <span aria-hidden="true">↗</span>;
}

function Logo({ footer = false }: { footer?: boolean }) {
  return (
    <img
      className={footer ? "brand-image brand-image--footer" : "brand-image"}
      src="/nomo-logo.png"
      alt="Nomo AI"
    />
  );
}

function SiteHeader() {
  return (
    <header className="site-header">
      <a className="brand-link" href="/" aria-label="Nomo AI home">
        <Logo />
      </a>

      <nav className="site-nav" aria-label="Primary navigation">
        {navLinks.map((link) => (
          <a key={link.href} href={link.href}>
            {link.label}
          </a>
        ))}
      </nav>

      <Button asChild variant="outline" size="compact">
        <a href={ENGINE_URL} target="_blank" rel="noreferrer">
          Open Engine <ExternalArrow />
        </a>
      </Button>
    </header>
  );
}

function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="footer-brand">
        <a className="brand-link" href="/" aria-label="Nomo AI home">
          <Logo footer />
        </a>
        <p>Open research and compiler infrastructure for physical AI.</p>
      </div>

      <div className="footer-links" aria-label="Project links">
        <a href={NOMO_REPO_URL} target="_blank" rel="noreferrer">
          Nomo AI on GitHub <ExternalArrow />
        </a>
        <a href={ENGINE_REPO_URL} target="_blank" rel="noreferrer">
          Engine source <ExternalArrow />
        </a>
        <a href={ENGINE_URL} target="_blank" rel="noreferrer">
          Open Engine <ExternalArrow />
        </a>
      </div>

      <p className="footer-credit">Nomo AI · open source research project</p>
    </footer>
  );
}

function SiteShell({ children }: { children: ReactNode }) {
  return (
    <div className="site-shell">
      <SiteHeader />
      {children}
      <SiteFooter />
    </div>
  );
}

function PageHeader({ eyebrow, title, children }: { eyebrow: string; title: string; children: ReactNode }) {
  return (
    <div className="page-header">
      <p className="page-kicker">{eyebrow}</p>
      <h1>{title}</h1>
      <div className="page-intro">{children}</div>
    </div>
  );
}

function ArrowLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a className="arrow-link" href={href} {...getLinkProps(href)}>
      {children} <ExternalArrow />
    </a>
  );
}

function CompilerPipeline() {
  return (
    <section className="pipeline-panel" aria-labelledby="pipeline-title">
      <div className="panel-heading">
        <p className="panel-kicker">Compiler pipeline</p>
        <h2 id="pipeline-title">From graph to deployable output</h2>
      </div>

      <div className="pipeline-flow">
        <div className="pipeline-step">
          <div className="pipeline-step-top">
            <span className="pipeline-step-type">Input</span>
            <span className="pipeline-step-mark">A</span>
          </div>
          <h3>Model ingestion</h3>
          <p>Catalog a supported graph and its target hardware constraints.</p>
          <div className="tag-row">
            <span className="code-tag">.pt</span>
            <span className="code-tag">.pth</span>
            <span className="code-tag code-tag--muted">NIR graph JSON placeholder</span>
          </div>
        </div>

        <div className="pipeline-connector" aria-hidden="true" />

        <div className="pipeline-step">
          <div className="pipeline-step-top">
            <span className="pipeline-step-type">Search</span>
            <span className="pipeline-step-mark">B</span>
          </div>
          <h3>NSGA-II multi-domain search</h3>
          <p>Explore standard, spiking, and symbolic stages against hardware limits.</p>
          <div className="tag-row">
            <span className="code-tag">ANN ↔ SNN</span>
            <span className="code-tag">symbolic</span>
            <span className="code-tag">current engine</span>
          </div>
        </div>

        <div className="pipeline-connector" aria-hidden="true" />

        <div className="pipeline-step">
          <div className="pipeline-step-top">
            <span className="pipeline-step-type">Output</span>
            <span className="pipeline-step-mark">C</span>
          </div>
          <h3>Export backends</h3>
          <p>Emit supported NIR and zero-dependency bare-metal C11 output.</p>
          <div className="tag-row">
            <span className="code-tag">NIR</span>
            <span className="code-tag">C11</span>
            <span className="code-tag code-tag--muted">ONNX export placeholder</span>
          </div>
        </div>
      </div>
    </section>
  );
}

function ResearchPreview() {
  return (
    <section className="page-section research-preview" aria-labelledby="research-preview-title">
      <div className="section-heading-row">
        <div>
          <p className="section-kicker">Open research</p>
          <h2 id="research-preview-title">Research and preprints</h2>
        </div>
        <p className="section-heading-copy">
          Planned publications and standards work are listed as placeholders until they exist in public form.
        </p>
      </div>

      <div className="paper-grid">
        <article className="paper-card" id="preprint-placeholder">
          <div className="paper-card-top">
            <span className="paper-status">Featured preprint placeholder</span>
            <span className="paper-index">01</span>
          </div>
          <h3>Hardware-Aware NSGA-II Partitioning for Hybrid Spiking-Continuous Computational Graphs</h3>
          <p>
            Placeholder for a planned paper on graph partitioning across continuous, spiking, and symbolic stages.
            No arXiv publication exists yet.
          </p>
          <div className="card-links">
            <ArrowLink href="/research#preprint-placeholder">arXiv PDF placeholder</ArrowLink>
            <ArrowLink href="/research#preprint-placeholder">BibTeX placeholder</ArrowLink>
          </div>
        </article>

        <article className="paper-card">
          <div className="paper-card-top">
            <span className="paper-status">Open-source standard placeholder</span>
            <span className="paper-index">02</span>
          </div>
          <h3>Extending NIR 1.1 for Symbolic Guardrails &amp; Temporal Spike Encoding</h3>
          <p>
            Placeholder for a future proposal. The current public engine documents strict and extended NIR behavior
            separately.
          </p>
          <div className="card-links">
            <ArrowLink href="/architecture#nir-extensions">Read architecture page</ArrowLink>
            <ArrowLink href={ENGINE_NIR_URL}>Read current NIR notes</ArrowLink>
          </div>
        </article>
      </div>

      <div className="section-link-row">
        <ArrowLink href="/research">View research page</ArrowLink>
      </div>
    </section>
  );
}

function FeatureRows() {
  const rows = [
    {
      title: "Hardware-aware search",
      copy: "The engine searches model and hardware choices together, using latency, energy, memory, and accuracy objectives.",
    },
    {
      title: "Multi-domain graph partitioning",
      copy: "A candidate graph can combine standard neural operations, event-driven spiking stages, and symbolic constraints.",
    },
    {
      title: "Low-level output",
      copy: "Supported integer graphs can be emitted as zero-dependency C11 for embedded targets, with NIR exports for inspection and exchange.",
    },
    {
      title: "Measured claims are still ahead",
      copy: "Energy and latency coefficients are placeholders until the measurement LUT is calibrated on target hardware. This site does not present them as silicon benchmarks.",
    },
  ];

  return (
    <div className="feature-list">
      {rows.map((row) => (
        <div className="feature-row" key={row.title}>
          <h3>{row.title}</h3>
          <p>{row.copy}</p>
        </div>
      ))}
    </div>
  );
}

function LabResults() {
  return (
    <section className="page-section section-with-heading" aria-labelledby="lab-results-title">
      <div className="section-heading-row section-heading-row--wide">
        <div>
          <p className="section-kicker">New: Nomo Lab</p>
          <h2 id="lab-results-title">Plan how AI runs on hardware, and check the evidence.</h2>
        </div>
        <p className="section-heading-copy">
          The Lab plans LLM training, LLM serving and model design in your browser: every possible plan checked, best
          trade-offs shown, every choice explained, and the predictions tested against published measurements.
        </p>
      </div>
      <div className="lab-home-stats">
        <div><b>5.9%</b><span>average error predicting published training runs the model never saw</span></div>
        <div><b>22</b><span>measured runs from 1.7B to 1T parameters used as evidence</span></div>
        <div><b>13,440</b><span>plans checked in about a quarter of a second, without a server</span></div>
      </div>
      <div className="section-link-row">
        <ArrowLink href="/lab">Open the Lab</ArrowLink>
      </div>
    </section>
  );
}

function HomePage() {
  return (
    <SiteShell>
      <main>
        <section className="page-section hero" aria-labelledby="hero-title">
          <div className="hero-copy">
            <p className="hero-kicker">Nomo AI research and compiler suite</p>
            <h1 id="hero-title">Open science for physical AI and neuromorphic hardware.</h1>
            <p className="hero-lede">
              Nomo AI is an open research project for hardware-aware neural architecture search and compilation. It
              explores standard, spiking, and symbolic graph stages, then emits supported intermediate
              representations and low-level code for edge systems.
            </p>

            <div className="hero-actions">
              <Button asChild>
                <a href="/lab">Open the Lab</a>
              </Button>
              <Button asChild variant="outline">
                <a href={ENGINE_URL} target="_blank" rel="noreferrer">
                  Launch Compiler Engine <ExternalArrow />
                </a>
              </Button>
              <Button asChild variant="outline">
                <a href="/lab#evidence">See the evidence</a>
              </Button>
              <Button asChild variant="text">
                <a href={ENGINE_NIR_URL} target="_blank" rel="noreferrer">
                  View NIR Extensions (GitHub) <ExternalArrow />
                </a>
              </Button>
            </div>
          </div>

          <CompilerPipeline />
        </section>

        <LabResults />

        <ResearchPreview />

        <section className="page-section section-with-heading" aria-labelledby="current-engine-title">
          <div className="section-heading-row section-heading-row--wide">
            <div>
              <p className="section-kicker">What exists today</p>
              <h2 id="current-engine-title">The current engine</h2>
            </div>
            <p className="section-heading-copy">
              Nomo is being built in the open. These are the capabilities the public backend and source code describe
              today.
            </p>
          </div>
          <FeatureRows />
        </section>

        <section className="page-section closing-section" aria-labelledby="closing-title">
          <div>
            <p className="section-kicker">Start with the implementation</p>
            <h2 id="closing-title">Read the source, then run the engine.</h2>
          </div>
          <p>
            The research pages explain the current scope. The engine and its technical specification contain the
            implementation details, export rules, and known limitations.
          </p>
          <div className="final-actions">
            <Button asChild>
              <a href={ENGINE_URL} target="_blank" rel="noreferrer">
                Open Engine <ExternalArrow />
              </a>
            </Button>
            <Button asChild variant="outline">
              <a href={ENGINE_REPO_URL} target="_blank" rel="noreferrer">
                Read Engine Source <ExternalArrow />
              </a>
            </Button>
          </div>
        </section>
      </main>
    </SiteShell>
  );
}

function BibTeXButton() {
  const [copied, setCopied] = useState(false);

  async function copyBibTeX() {
    try {
      await navigator.clipboard.writeText(BIBTEX_PLACEHOLDER);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2200);
    } catch {
      setCopied(false);
    }
  }

  return (
    <button className="copy-button" type="button" onClick={copyBibTeX}>
      {copied ? "BibTeX copied" : "Copy BibTeX placeholder"}
    </button>
  );
}

function ResearchPage() {
  return (
    <SiteShell>
      <main className="page-main">
        <PageHeader eyebrow="Research & papers" title="Research that stays close to the implementation.">
          <p>
            This page tracks Nomo's public research direction. Items marked placeholder are planned work and are not
            published papers or accepted standards.
          </p>
        </PageHeader>

        <section className="content-section" id="results" aria-labelledby="results-title">
          <div className="content-section-heading">
            <p className="section-kicker">Results and working papers</p>
            <h2 id="results-title">A planner calibrated against published measurements</h2>
          </div>
          <div className="content-two-column">
            <div>
              <p>
                <strong>Study 1: predicting large-scale training.</strong> We transcribed the 22 measured training runs of
                Narayanan et al. (2021, Megatron-LM), from 1.7B to 1T parameters on up to 3,072 A100 GPUs. Fitted on 10 runs,
                the planner predicts 6 unseen runs of the same strategy within 5.9% on average and ranks all 12 unseen runs
                correctly (Spearman 0.94); leave-one-out error over all 22 runs is 6.4%. A strategy absent from the fit (ZeRO-3)
                needs its own measurements; with them, it transfers across model size within 10.3%.
              </p>
              <p>
                <strong>Study 2: what to build for a given usage.</strong> Combining the Chinchilla scaling law (as replicated by
                Besiroglu et al., 2024) with training and serving cost models, the planner recovers the compute-optimal ~20
                tokens per parameter when a model is never served, and shows the best model shrinking, training longer and moving
                to grouped-query attention as lifetime usage grows. It also shows serving hardware capping model size through
                latency before usage cost does.
              </p>
              <div className="inline-actions">
                <ArrowLink href="/lab#evidence">Explore the evidence interactively</ArrowLink>
                <ArrowLink href="/lab#methods">Read the methods</ArrowLink>
              </div>
            </div>
            <div className="placeholder-note">
              <strong>Status</strong>
              <p>Working papers. Results are reproducible in the Lab and the Python reference implementation; not yet peer reviewed or posted to arXiv.</p>
            </div>
          </div>
        </section>

        <section className="content-section" id="preprint-placeholder" aria-labelledby="featured-paper-title">
          <div className="content-section-heading">
            <p className="section-kicker">Featured preprint placeholder</p>
            <h2 id="featured-paper-title">
              Hardware-Aware NSGA-II Partitioning for Hybrid Spiking-Continuous Computational Graphs
            </h2>
          </div>
          <div className="content-two-column">
            <div>
              <p>
                This is a placeholder for a future preprint about partitioning hybrid computational graphs under
                hardware-aware objectives. There is no arXiv record or published PDF yet.
              </p>
              <div className="inline-actions">
                <ArrowLink href="#preprint-placeholder">arXiv PDF placeholder</ArrowLink>
                <BibTeXButton />
              </div>
            </div>
            <div className="placeholder-note">
              <strong>Publication status</strong>
              <p>Not published. The title is included to make the research direction clear without inventing a link.</p>
            </div>
          </div>
        </section>

        <section className="content-section" aria-labelledby="standard-title">
          <div className="content-section-heading">
            <p className="section-kicker">Open-source standard placeholder</p>
            <h2 id="standard-title">Extending NIR 1.1 for Symbolic Guardrails &amp; Temporal Spike Encoding</h2>
          </div>
          <div className="content-two-column">
            <p>
              This is a proposed direction for a future NIR extension. It is not an existing NIR 1.1 proposal and is
              separate from the current engine's strict and extended export paths.
            </p>
            <ArrowLink href="/architecture#nir-extensions">See the current NIR architecture</ArrowLink>
          </div>
        </section>

        <section className="content-section" aria-labelledby="source-material-title">
          <div className="content-section-heading">
            <p className="section-kicker">Source material</p>
            <h2 id="source-material-title">Read the code behind the claims.</h2>
          </div>
          <div className="resource-list">
            <ResourceRow title="Engine repository" copy="The current backend, CLI, exporters, and tests." href={ENGINE_REPO_URL} />
            <ResourceRow title="Technical specification" copy="Search objectives, telemetry, and NIR export behavior." href={ENGINE_SPEC_URL} />
            <ResourceRow title="Nomo AI repository" copy="The public landing page and project entry point." href={NOMO_REPO_URL} />
          </div>
        </section>
      </main>
    </SiteShell>
  );
}

function ArchitecturePage() {
  const rows = [
    {
      title: "Strict NIR",
      status: "Implemented",
      copy: "The current engine exports strict NIR and tests it against stock NIR 1.0.8 behavior.",
    },
    {
      title: "Extended NIR",
      status: "Implemented",
      copy: "Nomo-specific node types are available through the extended path. Stock NIR rejects unknown types by design.",
    },
    {
      title: "NIR 1.1 extension proposal",
      status: "Placeholder",
      copy: "The proposed symbolic guardrails and temporal spike encoding work is not a public NIR 1.1 proposal yet.",
    },
    {
      title: "Bare-metal C11",
      status: "Supported coverage",
      copy: "The backend emits zero-dependency C11 for supported integer graph coverage. It is not a universal compiler.",
    },
    {
      title: "ONNX export",
      status: "Placeholder",
      copy: "ONNX is part of the planned interchange surface and is not emitted by the current engine.",
    },
  ];

  return (
    <SiteShell>
      <main className="page-main">
        <PageHeader eyebrow="Architecture (NIR)" title="A compiler pipeline with explicit boundaries.">
          <p>
            Nomo connects model graphs, hardware-aware search, intermediate representations, and low-level output.
            The table below separates what the current engine supports from what remains a placeholder.
          </p>
        </PageHeader>

        <section className="content-section" aria-labelledby="architecture-flow-title">
          <div className="content-section-heading">
            <p className="section-kicker">Pipeline</p>
            <h2 id="architecture-flow-title">Graph in, constrained candidates out.</h2>
          </div>
          <div className="architecture-flow">
            <FlowRow label="Model graph" copy="Supported model inputs and catalog workloads" />
            <FlowRow label="Multi-domain search" copy="ANN, SNN, and symbolic graph choices under hardware objectives" />
            <FlowRow label="Integer graph" copy="A lowered representation suitable for supported embedded targets" />
            <FlowRow label="Export" copy="Strict or extended NIR, plus bare-metal C11 where coverage allows" />
          </div>
        </section>

        <section className="content-section" id="nir-extensions" aria-labelledby="nir-surface-title">
          <div className="content-section-heading">
            <p className="section-kicker">NIR surface</p>
            <h2 id="nir-surface-title">Current exports and future work.</h2>
          </div>
          <div className="status-table">
            {rows.map((row) => (
              <div className="status-row" key={row.title}>
                <div>
                  <h3>{row.title}</h3>
                  <p>{row.copy}</p>
                </div>
                <span className={row.status === "Placeholder" ? "status-badge status-badge--placeholder" : "status-badge"}>
                  {row.status}
                </span>
              </div>
            ))}
          </div>
          <div className="section-note">
            <p>
              The current public engine does not claim an MLIR, snn-mlir, or microTVM emitter. Those are possible
              future integrations, not present features.
            </p>
            <ArrowLink href={ENGINE_NIR_URL}>Read the NIR export specification</ArrowLink>
          </div>
        </section>

        <section className="content-section" aria-labelledby="architecture-source-title">
          <div className="content-section-heading">
            <p className="section-kicker">Implementation</p>
            <h2 id="architecture-source-title">Follow the source.</h2>
          </div>
          <div className="resource-list">
            <ResourceRow title="Engine source" copy="Compiler code and export implementations on GitHub." href={ENGINE_REPO_URL} />
            <ResourceRow title="Nomo AI source" copy="The public home for this project." href={NOMO_REPO_URL} />
          </div>
        </section>
      </main>
    </SiteShell>
  );
}

function BenchmarksPage() {
  const rows = [
    {
      title: "Energy and latency coefficients",
      status: "Placeholder",
      copy: "The current objective coefficients are provisional until calibrated with a device measurement LUT.",
    },
    {
      title: "Accuracy model",
      status: "Proxy",
      copy: "Search uses a representative accuracy proxy. It is not a measured accuracy claim for a deployed model.",
    },
    {
      title: "Bare-metal C11 backend",
      status: "Supported tests",
      copy: "The backend has tests for supported integer graph coverage. No external performance number is published here.",
    },
    {
      title: "On-device measurements",
      status: "Not published",
      copy: "No measured silicon benchmark is published on this landing page yet.",
    },
  ];

  return (
    <SiteShell>
      <main className="page-main">
        <PageHeader eyebrow="Benchmarks" title="A benchmark page that does not overstate the data.">
          <p>
            Nomo's search objectives are designed for hardware-aware exploration. The measured hardware results are
            still being built, so provisional coefficients and proxy accuracy are labeled as such.
          </p>
        </PageHeader>

        <section className="content-section" aria-labelledby="benchmark-status-title">
          <div className="content-section-heading">
            <p className="section-kicker">Evidence status</p>
            <h2 id="benchmark-status-title">What is measured, and what is not.</h2>
          </div>
          <div className="status-table">
            {rows.map((row) => (
              <div className="status-row" key={row.title}>
                <div>
                  <h3>{row.title}</h3>
                  <p>{row.copy}</p>
                </div>
                <span className={row.status === "Placeholder" ? "status-badge status-badge--placeholder" : "status-badge"}>
                  {row.status}
                </span>
              </div>
            ))}
          </div>
        </section>

        <section className="content-section" aria-labelledby="benchmark-method-title">
          <div className="content-section-heading">
            <p className="section-kicker">Method</p>
            <h2 id="benchmark-method-title">The next useful result is a calibrated result.</h2>
          </div>
          <div className="content-two-column">
            <p>
              A meaningful hardware comparison needs paired model outputs, target-device measurements, and a clear
              record of the graph and compiler settings. Until that record exists, Nomo keeps the public language
              conservative.
            </p>
            <div className="resource-list resource-list--compact">
              <ResourceRow title="Technical specification" copy="Objective definitions and telemetry fields." href={ENGINE_SPEC_URL} />
              <ResourceRow title="Engine repository" copy="Tests and implementation details." href={ENGINE_REPO_URL} />
            </div>
          </div>
        </section>
      </main>
    </SiteShell>
  );
}

function DocsPage() {
  return (
    <SiteShell>
      <main className="page-main">
        <PageHeader eyebrow="Documentation" title="Start with the source and the specification.">
          <p>
            These links point to the current engine documentation. The live backend and compiler dashboard are also
            available if you want to inspect the system directly.
          </p>
        </PageHeader>

        <section className="content-section" aria-labelledby="docs-links-title">
          <div className="content-section-heading">
            <p className="section-kicker">Project documents</p>
            <h2 id="docs-links-title">The implementation is the reference.</h2>
          </div>
          <div className="resource-list">
            <ResourceRow title="Engine README" copy="Install, run, and understand the current backend." href={ENGINE_README_URL} />
            <ResourceRow title="Technical specification" copy="Search behavior, graph domains, telemetry, and exports." href={ENGINE_SPEC_URL} />
            <ResourceRow title="Deployment guide" copy="Run the API and dashboard in your own environment." href={ENGINE_DEPLOY_URL} />
            <ResourceRow title="Observability guide" copy="Inspect metrics, events, and runtime signals." href={ENGINE_OBSERVABILITY_URL} />
            <ResourceRow title="Hosted backend" copy="The current public API deployment on Render." href={HOSTED_BACKEND_URL} />
          </div>
        </section>

        <section className="content-section" aria-labelledby="quickstart-title">
          <div className="content-section-heading">
            <p className="section-kicker">Quick start</p>
            <h2 id="quickstart-title">Run a search, then compile a candidate.</h2>
          </div>
          <div className="code-block" role="region" aria-label="Nomo engine quick start commands">
            <code>
              <span>$ pip install -e &quot;.[server,dev]&quot;</span>
              <span>$ nomo search --model perception_cnn --hardware akd1500 --out run.json</span>
              <span>$ nomo compile --model attitude_policy --hardware akd1500 --genome run.json --out build/</span>
            </code>
          </div>
          <div className="section-note">
            <p>Available model and hardware combinations depend on the current engine catalog and supported graph coverage.</p>
            <ArrowLink href={ENGINE_README_URL}>Read the full README</ArrowLink>
          </div>
        </section>

        <section className="content-section" aria-labelledby="docs-source-title">
          <div className="content-section-heading">
            <p className="section-kicker">Open source</p>
            <h2 id="docs-source-title">Keep following the work on GitHub.</h2>
          </div>
          <div className="inline-actions">
            <Button asChild>
              <a href={ENGINE_REPO_URL} target="_blank" rel="noreferrer">
                Engine GitHub <ExternalArrow />
              </a>
            </Button>
            <Button asChild variant="outline">
              <a href={NOMO_REPO_URL} target="_blank" rel="noreferrer">
                Nomo AI GitHub <ExternalArrow />
              </a>
            </Button>
          </div>
        </section>
      </main>
    </SiteShell>
  );
}

function FlowRow({ label, copy }: { label: string; copy: string }) {
  return (
    <div className="flow-row">
      <span className="flow-dot" aria-hidden="true" />
      <div>
        <h3>{label}</h3>
        <p>{copy}</p>
      </div>
    </div>
  );
}

function ResourceRow({ title, copy, href }: { title: string; copy: string; href: string }) {
  return (
    <div className="resource-row">
      <div>
        <h3>{title}</h3>
        <p>{copy}</p>
      </div>
      <ArrowLink href={href}>Open resource</ArrowLink>
    </div>
  );
}

function App() {
  const path = window.location.pathname.replace(/\/+$/, "") || "/";

  useEffect(() => {
    const titles: Record<string, string> = {
      "/": "Nomo Lab | Nomo AI",
      "/home": "Nomo AI | Open science for physical AI",
      "/research": "Research & Papers | Nomo AI",
      "/architecture": "Architecture (NIR) | Nomo AI",
      "/benchmarks": "Benchmarks | Nomo AI",
      "/docs": "Documentation | Nomo AI",
      "/lab": "Lab | Nomo AI",
    };

    document.title = titles[path] ?? "Nomo AI";
  }, [path]);

  if (path === "/" || path === "/lab") {
    return (
      <div className="site-shell">
        <SiteHeader />
        <Suspense fallback={<main className="lab lab-loading" aria-live="polite">Loading Nomo Lab…</main>}>
          <LabPage engineUrl={ENGINE_URL} />
        </Suspense>
        <SiteFooter />
      </div>
    );
  }
  if (path === "/home") return <HomePage />;
  if (path === "/research") return <ResearchPage />;
  if (path === "/architecture") return <ArchitecturePage />;
  if (path === "/benchmarks") return <BenchmarksPage />;
  if (path === "/docs") return <DocsPage />;
  return <HomePage />;
}

export default App;
