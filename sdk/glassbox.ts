/**
 * Glassbox TypeScript / JavaScript SDK
 * =====================================
 * Official SDK for the Glassbox EU AI Act Compliance Audit API.
 *
 * Works in Node.js ≥18, Deno, Bun, and any browser that supports fetch().
 *
 * Installation (npm):
 *   npm install glassbox-sdk
 *
 * Installation (CDN / browser):
 *   <script type="module">
 *     import { GlassboxClient } from 'https://cdn.jsdelivr.net/npm/glassbox-sdk/dist/glassbox.js'
 *   </script>
 *
 * Quick start:
 *   import { GlassboxClient } from 'glassbox-sdk'
 *   const gb = new GlassboxClient()
 *   const report = await gb.auditWhiteBox({ modelName: 'gpt2', prompt: '...', correctToken: ' Mary', incorrectToken: ' John' })
 *   console.log(report.explainabilityGrade)   // "A — Fully Explainable"
 *
 * References
 * ----------
 * EU AI Act, Regulation (EU) 2024/1689 — enforcement August 2026
 * Annex IV — technical documentation requirements for high-risk AI
 * Article 13 — transparency obligations
 */

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export type DeploymentContext =
  | 'financial_services'
  | 'healthcare'
  | 'hr_employment'
  | 'legal'
  | 'critical_infrastructure'
  | 'education'
  | 'other_high_risk'

export type AttributionMethod = 'taylor' | 'integrated_gradients'

export type ExplainabilityGrade = 'A' | 'B' | 'C' | 'D'

export type ComplianceStatus =
  | 'conditionally_compliant'
  | 'incomplete'
  | 'non_compliant'

export type JobStatus = 'queued' | 'running' | 'completed' | 'failed'

export interface ProviderInfo {
  /** Legal name of the AI system provider (for Annex IV Section 1). */
  providerName?: string
  /** Registered address of the provider. */
  providerAddress?: string
  /** Human-readable description of the AI system's intended purpose. */
  systemPurpose?: string
  /** Deployment context — determines Annex III high-risk category. */
  deploymentContext?: DeploymentContext
  /** Short description of the specific use case being audited. */
  useCase?: string
}

export interface WhiteBoxRequest extends ProviderInfo {
  /** TransformerLens-compatible model name (e.g. 'gpt2', 'gpt2-medium'). */
  modelName: string
  /** The decision prompt to analyse. */
  prompt: string
  /** Expected correct output token (e.g. ' Mary'). */
  correctToken: string
  /** Distractor / incorrect token (e.g. ' John'). */
  incorrectToken: string
  /** Attribution method. Default: 'taylor' (fastest). */
  method?: AttributionMethod
  /** Generate a PDF report in addition to JSON. Default: false. */
  generatePdf?: boolean
}

export interface BlackBoxRequest extends ProviderInfo {
  /** Target API provider (openai | anthropic | together | groq). */
  targetProvider: 'openai' | 'anthropic' | 'together' | 'groq'
  /** Model name as accepted by the provider's API (e.g. 'gpt-4'). */
  targetModel: string
  /** The decision prompt to probe. */
  decisionPrompt: string
  /** Token/phrase the model should output for a "positive" decision. */
  expectedPositive: string
  /** Token/phrase the model should output for a "negative" decision. */
  expectedNegative: string
  /** Context variables to inject into the prompt. */
  contextVariables?: Record<string, unknown>
  /** Number of rephrased probes to run. Default: 3. */
  nRephrases?: number
  /** Generate a PDF report. Default: false. */
  generatePdf?: boolean
  /**
   * Your provider API key (OpenAI, Anthropic, etc.).
   * Sent as an HTTP header — never in the request body, never logged.
   */
  providerApiKey?: string
}

export interface Faithfulness {
  sufficiency: number
  comprehensiveness: number
  f1: number
  category: string
}

export interface AuditReport {
  reportId: string
  status: string
  complianceStatus: ComplianceStatus
  /** Full grade string e.g. "A — Fully Explainable" */
  explainabilityGrade: string
  /** Single letter grade: A | B | C | D */
  grade: ExplainabilityGrade
  faithfulness: Faithfulness
  nCircuitComponents: number
  analysisMode: string
  elapsedSeconds: number
  jsonReportUrl: string | null
  pdfReportUrl: string | null
  fullReport: Record<string, unknown>
}

export interface AsyncJobResponse {
  jobId: string
  status: JobStatus
  statusUrl: string
  reportId: string | null
  error: string | null
  createdAt: number
}

export interface AttentionPatternsResponse {
  heads: string[]
  patterns: Record<string, number[][]>
  entropy: Record<string, number>
  headTypes: Record<string, string>
  tokenStrs: string[]
}

export interface GlassboxClientOptions {
  /**
   * Base URL of the Glassbox API.
   * Default: 'https://glassbox-ai-2-0-mechanistic.onrender.com'
   */
  baseUrl?: string
  /** Request timeout in milliseconds. Default: 120_000 (2 minutes). */
  timeoutMs?: number
  /**
   * Custom fetch implementation (useful in environments without global fetch,
   * or for testing with mock fetch).
   */
  fetch?: typeof fetch
}

// ─────────────────────────────────────────────────────────────────────────────
// Error class
// ─────────────────────────────────────────────────────────────────────────────

export class GlassboxError extends Error {
  constructor(
    message: string,
    public readonly statusCode?: number,
    public readonly detail?: unknown,
  ) {
    super(message)
    this.name = 'GlassboxError'
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Client
// ─────────────────────────────────────────────────────────────────────────────

export class GlassboxClient {
  private readonly baseUrl: string
  private readonly timeoutMs: number
  private readonly _fetch: typeof fetch

  constructor(options: GlassboxClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? 'https://glassbox-ai-2-0-mechanistic.onrender.com').replace(/\/$/, '')
    this.timeoutMs = options.timeoutMs ?? 120_000
    this._fetch = options.fetch ?? globalThis.fetch.bind(globalThis)
  }

  // ── internal helpers ───────────────────────────────────────────────────────

  private async _post<T>(
    path: string,
    body: Record<string, unknown>,
    headers: Record<string, string> = {},
  ): Promise<T> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.timeoutMs)

    let resp: Response
    try {
      resp = await this._fetch(`${this.baseUrl}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...headers },
        body: JSON.stringify(body),
        signal: controller.signal,
      })
    } catch (err: unknown) {
      clearTimeout(timer)
      if (err instanceof Error && err.name === 'AbortError') {
        throw new GlassboxError(`Request timed out after ${this.timeoutMs}ms`)
      }
      throw new GlassboxError(`Network error: ${err instanceof Error ? err.message : String(err)}`)
    }
    clearTimeout(timer)

    if (!resp.ok) {
      const payload = await resp.json().catch(() => ({ detail: resp.statusText }))
      throw new GlassboxError(
        `API error ${resp.status}: ${payload?.detail ?? resp.statusText}`,
        resp.status,
        payload,
      )
    }
    return resp.json() as Promise<T>
  }

  private async _get<T>(path: string): Promise<T> {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.timeoutMs)

    let resp: Response
    try {
      resp = await this._fetch(`${this.baseUrl}${path}`, {
        method: 'GET',
        signal: controller.signal,
      })
    } catch (err: unknown) {
      clearTimeout(timer)
      throw new GlassboxError(`Network error: ${err instanceof Error ? err.message : String(err)}`)
    }
    clearTimeout(timer)

    if (!resp.ok) {
      const payload = await resp.json().catch(() => ({ detail: resp.statusText }))
      throw new GlassboxError(
        `API error ${resp.status}: ${payload?.detail ?? resp.statusText}`,
        resp.status,
        payload,
      )
    }
    return resp.json() as Promise<T>
  }

  private static _parseReport(raw: Record<string, unknown>): AuditReport {
    const gradeStr = (raw.explainability_grade as string) ?? 'D — Unknown'
    return {
      reportId: raw.report_id as string,
      status: raw.status as string,
      complianceStatus: raw.compliance_status as ComplianceStatus,
      explainabilityGrade: gradeStr,
      grade: (gradeStr[0] ?? 'D') as ExplainabilityGrade,
      faithfulness: raw.faithfulness as Faithfulness,
      nCircuitComponents: raw.n_circuit_components as number,
      analysisMode: raw.analysis_mode as string,
      elapsedSeconds: raw.elapsed_seconds as number,
      jsonReportUrl: raw.json_report_url as string | null,
      pdfReportUrl: raw.pdf_report_url as string | null,
      fullReport: (raw.full_report as Record<string, unknown>) ?? {},
    }
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /**
   * Check the API health.
   * Returns the API version and current server time.
   */
  async health(): Promise<{ status: string; glassbox_version: string; timestamp: string }> {
    return this._get('/health')
  }

  /**
   * Run a white-box mechanistic interpretability audit.
   *
   * Requires a TransformerLens-compatible model (e.g. 'gpt2').
   * The API server must have enough RAM to load the model weights
   * (~500 MB for GPT-2 small).
   *
   * @param req - White-box audit request parameters.
   * @returns Structured Annex IV compliance report.
   *
   * @example
   * const report = await gb.auditWhiteBox({
   *   modelName: 'gpt2',
   *   prompt: 'When Mary and John went to the store, John gave a drink to',
   *   correctToken: ' Mary',
   *   incorrectToken: ' John',
   *   providerName: 'Acme Bank NV',
   *   deploymentContext: 'financial_services',
   * })
   * console.log(report.grade)  // 'A'
   */
  async auditWhiteBox(req: WhiteBoxRequest): Promise<AuditReport> {
    const raw = await this._post<Record<string, unknown>>('/v1/audit/analyze', {
      model_name:         req.modelName,
      prompt:             req.prompt,
      correct_token:      req.correctToken,
      incorrect_token:    req.incorrectToken,
      method:             req.method ?? 'taylor',
      provider_name:      req.providerName ?? '',
      provider_address:   req.providerAddress ?? '',
      system_purpose:     req.systemPurpose ?? '',
      deployment_context: req.deploymentContext ?? 'other_high_risk',
      use_case:           req.useCase ?? '',
      generate_pdf:       req.generatePdf ?? false,
    })
    return GlassboxClient._parseReport(raw)
  }

  /**
   * Run a black-box behavioural audit against any model API.
   * No model weights needed — probes the model via its public API.
   *
   * @param req - Black-box audit request (includes providerApiKey if needed).
   * @returns Structured Annex IV compliance report.
   *
   * @example
   * const report = await gb.auditBlackBox({
   *   targetProvider: 'openai',
   *   targetModel: 'gpt-4',
   *   decisionPrompt: 'The loan application should be',
   *   expectedPositive: 'approved',
   *   expectedNegative: 'denied',
   *   providerApiKey: process.env.OPENAI_API_KEY,
   *   deploymentContext: 'financial_services',
   * })
   */
  async auditBlackBox(req: BlackBoxRequest): Promise<AuditReport> {
    const headers: Record<string, string> = {}
    if (req.providerApiKey) {
      headers['X-Provider-Api-Key'] = req.providerApiKey
    }
    const raw = await this._post<Record<string, unknown>>(
      '/v1/audit/black-box',
      {
        target_provider:    req.targetProvider,
        target_model:       req.targetModel,
        decision_prompt:    req.decisionPrompt,
        expected_positive:  req.expectedPositive,
        expected_negative:  req.expectedNegative,
        context_variables:  req.contextVariables ?? null,
        n_rephrases:        req.nRephrases ?? 3,
        provider_name:      req.providerName ?? '',
        provider_address:   req.providerAddress ?? '',
        system_purpose:     req.systemPurpose ?? '',
        deployment_context: req.deploymentContext ?? 'other_high_risk',
        use_case:           req.useCase ?? req.targetModel + ' audit',
        generate_pdf:       req.generatePdf ?? false,
      },
      headers,
    )
    return GlassboxClient._parseReport(raw)
  }

  /**
   * Start a black-box audit as a background job.
   * Returns immediately with a job ID. Poll with {@link pollJob}.
   *
   * @param req - Same as {@link auditBlackBox}.
   * @returns Job object with `jobId` and `statusUrl`.
   */
  async startBlackBoxJob(req: BlackBoxRequest): Promise<AsyncJobResponse> {
    const headers: Record<string, string> = {}
    if (req.providerApiKey) {
      headers['X-Provider-Api-Key'] = req.providerApiKey
    }
    return this._post<AsyncJobResponse>(
      '/v1/audit/black-box/async',
      {
        target_provider:    req.targetProvider,
        target_model:       req.targetModel,
        decision_prompt:    req.decisionPrompt,
        expected_positive:  req.expectedPositive,
        expected_negative:  req.expectedNegative,
        context_variables:  req.contextVariables ?? null,
        n_rephrases:        req.nRephrases ?? 3,
        provider_name:      req.providerName ?? '',
        provider_address:   req.providerAddress ?? '',
        system_purpose:     req.systemPurpose ?? '',
        deployment_context: req.deploymentContext ?? 'other_high_risk',
        generate_pdf:       req.generatePdf ?? false,
      },
      headers,
    )
  }

  /**
   * Poll the status of a background job started with {@link startBlackBoxJob}.
   *
   * @param jobId - Job ID returned by startBlackBoxJob.
   * @returns Current job status and report ID when completed.
   */
  async pollJob(jobId: string): Promise<AsyncJobResponse> {
    return this._get<AsyncJobResponse>(`/v1/jobs/${jobId}`)
  }

  /**
   * Wait for a background job to complete, polling every `intervalMs`.
   * Resolves with the final job response (status: 'completed' | 'failed').
   *
   * @param jobId      - Job ID to wait on.
   * @param intervalMs - Polling interval in ms. Default: 2000.
   * @param maxWaitMs  - Maximum wait time. Default: 300_000 (5 min).
   */
  async waitForJob(
    jobId: string,
    intervalMs = 2_000,
    maxWaitMs = 300_000,
  ): Promise<AsyncJobResponse> {
    const deadline = Date.now() + maxWaitMs
    while (Date.now() < deadline) {
      const job = await this.pollJob(jobId)
      if (job.status === 'completed' || job.status === 'failed') return job
      await new Promise(r => setTimeout(r, intervalMs))
    }
    throw new GlassboxError(`Job ${jobId} did not complete within ${maxWaitMs}ms`)
  }

  /**
   * Retrieve a previously generated report by ID.
   */
  async getReport(reportId: string): Promise<AuditReport> {
    const raw = await this._get<Record<string, unknown>>(`/v1/audit/report/${reportId}`)
    return GlassboxClient._parseReport(raw)
  }

  /**
   * Get the URL to download the PDF version of a report.
   * The PDF is only available if `generatePdf: true` was set in the request.
   */
  pdfUrl(reportId: string): string {
    return `${this.baseUrl}/v1/audit/pdf/${reportId}`
  }

  /**
   * List all reports stored in the current API session.
   */
  async listReports(): Promise<{ reports: unknown[]; total: number }> {
    return this._get('/v1/audit/reports')
  }

  /**
   * Fetch attention patterns for specific heads in a white-box model.
   * Useful for visualizing what each circuit head is attending to.
   *
   * @param modelName - TransformerLens model name.
   * @param prompt    - Input prompt to analyse.
   * @param heads     - Optional list of head labels to inspect (e.g. ['L9H9', 'L9H6']).
   * @param topK      - If heads is null, return the top-K most interesting heads.
   */
  async attentionPatterns(
    modelName: string,
    prompt: string,
    heads?: string[],
    topK = 10,
  ): Promise<AttentionPatternsResponse> {
    return this._post<AttentionPatternsResponse>('/v1/attention-patterns', {
      model_name: modelName,
      prompt,
      heads: heads ?? null,
      top_k: topK,
    })
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factory
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a GlassboxClient with default options.
 * Equivalent to `new GlassboxClient(options)`.
 *
 * @example
 * import { createClient } from 'glassbox-sdk'
 * const gb = createClient({ baseUrl: 'http://localhost:8000' })
 */
export function createClient(options?: GlassboxClientOptions): GlassboxClient {
  return new GlassboxClient(options)
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports for CommonJS / default import compatibility
// ─────────────────────────────────────────────────────────────────────────────

export default GlassboxClient
