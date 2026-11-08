export * as OpencodeTelemetry from "./opencode-telemetry.js"

import type { Context } from "@opencode/plugin/effect/plugin"
import type { Money } from "@opencode/schema/money"
import type { Model } from "@opencode/schema/model"
import { SessionEvent } from "@opencode/schema/session-event"
import type { SessionMessage } from "@opencode/schema/session-message"
import { TokenUsage } from "@opencode/schema/token-usage"
import { Clock, Duration, Effect, Exit, Fiber, Latch, Option, Schema, Scope, Stream } from "effect"
import { FetchHttpClient, HttpClient, HttpClientRequest } from "effect/http"
import { Bus } from "../../bus.js"
import type { Config } from "../../config.js"
import { Credential } from "../../credential.js"
import { foldModelSettings, foldSettings } from "./configured.js"

/** The most records Console stores from one export. */
export const BATCH_SIZE = 200
export const MAX_RECORDS = 1_000
const MAX_AGE = Duration.days(7)
const FLUSH_INTERVAL = Duration.seconds(30)
const SEND_TIMEOUT = Duration.seconds(10)
/** Closing is the last chance to send, and a dead network must not hold up an exiting CLI. */
const CLOSE_TIMEOUT = Duration.seconds(3)

const ENDPOINT_SETTINGS = ["baseURL", "apiKey", "authToken", "accessToken"]
const CHATGPT_METHODS = new Set(["chatgpt-token-sharing", "chatgpt-browser", "chatgpt-headless"])

/** Where one Console workspace takes its members' usage, as `GET /api/v2/telemetry` answered. */
export interface Target {
  readonly key: string
  readonly url: string
  readonly headers: Record<string, string>
  /** Providers the Console's config defines: the gateway already counts their usage. */
  readonly consoleProviders: ReadonlySet<string>
}

/** Provider metadata a step on a Claude plan finishes with. `subscription` alone says a plan answered this step. */
const ClaudeStep = Schema.Struct({
  subscription: Schema.optional(Schema.Boolean),
  model: Schema.optional(Schema.String),
  requestID: Schema.optional(Schema.String),
  cacheWrite5m: Schema.optional(Schema.Number),
  cacheWrite1h: Schema.optional(Schema.Number),
})
const decodeClaudeStep = Schema.decodeUnknownOption(ClaudeStep)

interface Step {
  readonly sessionID: string
  readonly stepID: string
  readonly created: number
  readonly model: Model.Ref | undefined
  readonly cost: Money.USD | undefined
  readonly tokens: TokenUsage.Info | undefined
  readonly native: SessionMessage.Native | undefined
}

interface Pending {
  readonly stepID: string
  readonly created: number
  readonly record: unknown
}

interface Hub {
  target: Target
  http: HttpClient.HttpClient
  records: Pending[]
  members: number
  readonly version: string
  readonly pending: Latch.Latch
  readonly full: Latch.Latch
  fiber?: Fiber.Fiber<void>
}

// Every Location the OpenCode server loads runs this plugin; they share one queue per Console workspace.
const hubs = new Map<string, Hub>()

/** Reports token usage of steps a member's own subscription covered, while `sync` is given a Target. */
export const make = Effect.fnUntraced(function* (
  ctx: Context,
  http: HttpClient.HttpClient,
  config: Pick<Config.Interface, "entries">,
) {
  const bus = yield* Bus.Service
  const credentials = yield* Credential.Service
  const scope = yield* Scope.Scope
  let current: { readonly key: string; readonly hub: Hub; readonly lease: Scope.Closeable } | undefined
  let warned = false

  const claudeStep = Effect.fnUntraced(function* (step: Step, modelID: string) {
    const native = Option.getOrUndefined(decodeClaudeStep(step.native))
    if (native?.subscription !== true) return undefined
    if (!exportedByClaudeCode()) return { provider: "anthropic", model: native.model ?? modelID, native }
    if (!warned)
      yield* Effect.logWarning(
        "skipping native claude-agent usage report because OPENCODE_CLAUDE_PATH/OTEL_EXPORTER_OTLP_LOGS_ENDPOINT already exports to Console",
      )
    warned = true
    return undefined
  })

  const chatGPTStep = Effect.fnUntraced(function* (modelID: string) {
    // Only the method is needed. Resolving can refresh the token, holding this serial consumer on the inference lock.
    const connection = yield* ctx.integration.connection.active("openai")
    const stored = connection?.type === "credential" ? yield* credentials.get(Credential.ID.make(connection.id)) : undefined
    if (stored?.value.type !== "oauth" || !CHATGPT_METHODS.has(stored.value.methodID)) return undefined
    return { provider: "openai", model: modelID, native: undefined }
  })

  const report = Effect.fnUntraced(
    function* (hub: Hub, step: Step) {
      if (!step.model || !step.tokens || TokenUsage.total(step.tokens) <= 0 || step.cost !== 0) return
      const providerID = step.model.providerID
      if (hub.target.consoleProviders.has(providerID)) return
      if (providerID !== "claude-agent" && providerID !== "openai") return
      const id = step.model.id
      const info = (yield* ctx.model.list()).data.find((model) => model.providerID === providerID && model.id === id)
      if (!info || info.cost.length > 0) return
      // From config alone: `info.settings` inherits the endpoint a ChatGPT sign-in installs on its provider.
      const entries = yield* config.entries()
      if (overridesEndpoint(foldSettings(entries, providerID, undefined))) return
      if (overridesEndpoint(foldModelSettings(entries, providerID, id))) return
      const modelID = info.modelID
      const usage = providerID === "openai" ? yield* chatGPTStep(modelID) : yield* claudeStep(step, modelID)
      if (!usage || usage.model.startsWith("opencode-go/") || usage.model.startsWith("console-")) return
      enqueue(hub, {
        stepID: step.stepID,
        created: step.created,
        record: logRecord(step, step.tokens, usage),
      })
    },
    Effect.catch((cause) => Effect.logWarning("failed to report usage to OpenCode Console", { cause })),
  )

  const start = Effect.fnUntraced(function* (hub: Hub) {
    // Per subscription, so a step that straddles a workspace switch has no model here and is never sent to the new one.
    const models = new Map<string, Model.Ref>()
    // The bus rather than `ctx.event`: `session.usage.recorded` is internal, so it isn't on the public stream.
    yield* bus
      .subscribe([
        SessionEvent.Step.Started,
        SessionEvent.Step.Ended,
        SessionEvent.Step.Failed,
        SessionEvent.UsageRecorded,
      ])
      .pipe(
        Stream.runForEach((event) => {
          if (event.type === "session.step.started") {
            models.set(event.data.assistantMessageID, event.data.model)
            return Effect.void
          }
          if (event.type === "session.usage.recorded")
            return report(hub, {
              sessionID: event.data.sessionID,
              stepID: event.id,
              created: event.created,
              model: event.data.model,
              cost: event.data.cost,
              tokens: event.data.tokens,
              native: event.data.native,
            })
          const model = models.get(event.data.assistantMessageID)
          models.delete(event.data.assistantMessageID)
          return report(hub, {
            sessionID: event.data.sessionID,
            stepID: event.data.assistantMessageID,
            created: event.created,
            model,
            cost: event.data.cost,
            tokens: event.data.tokens,
            native: event.data.native,
          })
        }),
        Effect.forkScoped({ startImmediately: true }),
      )
  })

  /** Starts reporting to `next`, or stops when it is undefined; a new workspace replaces the old one. */
  const sync = Effect.fnUntraced(function* (next: Target | undefined) {
    if (current && next && current.key === next.key) {
      current.hub.target = next
      current.hub.http = http
      return
    }
    if (current) {
      // No target means reporting is off or the sign-in is gone, so what's buffered stays here.
      if (!next && current.hub.members === 1) current.hub.records = []
      yield* Scope.close(current.lease, Exit.void)
      current = undefined
    }
    if (!next) return
    const lease = yield* Scope.fork(scope)
    const hub = yield* join(next, http, ctx.app.version).pipe(Scope.provide(lease))
    yield* start(hub).pipe(Scope.provide(lease))
    current = { key: next.key, hub, lease }
  })

  return { sync }
})

// The `claude` wrapper that exports to Console already reports these steps.
function exportedByClaudeCode() {
  return (
    process.env.OPENCODE_CLAUDE_PATH?.includes("claude-console-usage") === true ||
    process.env.OTEL_EXPORTER_OTLP_LOGS_ENDPOINT?.includes("/api/otel/v1/logs") === true
  )
}

function overridesEndpoint(settings: object | undefined) {
  return settings !== undefined && ENDPOINT_SETTINGS.some((key) => key in settings)
}

function logRecord(
  step: Step,
  tokens: TokenUsage.Info,
  usage: {
    readonly provider: string
    readonly model: string
    readonly native: typeof ClaudeStep.Type | undefined
  },
) {
  return {
    timeUnixNano: `${step.created}000000`,
    attributes: [
      text("event.name", "opencode.step_usage"),
      text("event.timestamp", new Date(step.created).toISOString()),
      text("provider", usage.provider),
      text("model", usage.model),
      text("step_id", step.stepID),
      text("session.id", step.sessionID),
      ...(usage.native?.requestID ? [text("request_id", usage.native.requestID)] : []),
      count("input_tokens", tokens.input),
      // Console counts reasoning inside output; OpenCode keeps them apart.
      count("output_tokens", tokens.output + tokens.reasoning),
      count("reasoning_tokens", tokens.reasoning),
      count("cache_read_tokens", tokens.cache.read),
      count("cache_write_5m_tokens", usage.native?.cacheWrite5m ?? tokens.cache.write),
      count("cache_write_1h_tokens", usage.native?.cacheWrite1h ?? 0),
    ],
  }
}

function text(key: string, stringValue: string) {
  return { key, value: { stringValue } }
}

function count(key: string, intValue: number) {
  return { key, value: { intValue } }
}

function enqueue(hub: Hub, pending: Pending) {
  // Console counts every record in one export, so a step two Locations heard must leave here once.
  if (hub.records.some((queued) => queued.stepID === pending.stepID)) return
  hub.records.push(pending)
  if (hub.records.length > MAX_RECORDS) hub.records.shift()
  hub.pending.openUnsafe()
  if (hub.records.length >= BATCH_SIZE) hub.full.openUnsafe()
}

function join(target: Target, http: HttpClient.HttpClient, version: string) {
  return Effect.acquireRelease(
    Effect.gen(function* () {
      const existing = hubs.get(target.key)
      if (existing) {
        existing.members++
        existing.target = target
        return existing
      }
      const hub: Hub = {
        target,
        http,
        version,
        records: [],
        members: 1,
        pending: Latch.makeUnsafe(),
        full: Latch.makeUnsafe(),
      }
      hubs.set(target.key, hub)
      // Detached so the queue outlives the Location that opened it while other Locations still use it.
      hub.fiber = yield* Effect.forkDetach(run(hub), { startImmediately: true })
      return hub
    }),
    (hub) =>
      Effect.gen(function* () {
        hub.members--
        if (hub.members > 0) return
        hubs.delete(hub.target.key)
        if (hub.fiber) yield* Fiber.interrupt(hub.fiber)
        // Finalizers are uninterruptible, and the timeout has to be able to stop a hung request.
        yield* flush(hub).pipe(Effect.interruptible, Effect.timeout(CLOSE_TIMEOUT), Effect.ignore)
      }),
  )
}

function run(hub: Hub) {
  return Effect.gen(function* () {
    let failures = 0
    while (true) {
      yield* hub.pending.await
      if (failures === 0) yield* Effect.raceFirst(Effect.sleep(FLUSH_INTERVAL), hub.full.await)
      if (yield* flush(hub)) {
        failures = 0
        continue
      }
      failures++
      yield* Effect.sleep(Duration.seconds(Math.min(5 * 2 ** (failures - 1), 60)))
    }
  })
}

/** Sends batches until none are left. False when Console should be asked again later, with the batch still queued. */
function flush(hub: Hub) {
  return Effect.gen(function* () {
    while (true) {
      const now = yield* Clock.currentTimeMillis
      hub.records = hub.records.filter((pending) => now - pending.created <= Duration.toMillis(MAX_AGE))
      const batch = hub.records.slice(0, BATCH_SIZE)
      if (batch.length === 0) {
        hub.pending.closeUnsafe()
        hub.full.closeUnsafe()
        return true
      }
      const status = yield* post(hub, batch)
      if (status === undefined || status === 429 || status >= 500) return false
      // Any other answer is final for these records: 2xx took them, and a 4xx would refuse them again. By identity,
      // because a full queue may have dropped some while the request ran.
      const sent = new Set(batch)
      hub.records = hub.records.filter((pending) => !sent.has(pending))
      if (hub.records.length < BATCH_SIZE) hub.full.closeUnsafe()
    }
  })
}

function post(hub: Hub, batch: ReadonlyArray<Pending>) {
  return HttpClient.withScope(hub.http)
    .execute(
      HttpClientRequest.post(hub.target.url).pipe(
        HttpClientRequest.acceptJson,
        HttpClientRequest.setHeaders(hub.target.headers),
        HttpClientRequest.bodyJsonUnsafe({
          resourceLogs: [
            {
              resource: {
                attributes: [text("service.name", "opencode"), text("service.version", hub.version)],
              },
              scopeLogs: [{ scope: { name: "opencode" }, logRecords: batch.map((pending) => pending.record) }],
            },
          ],
        }),
      ),
    )
    .pipe(
      Effect.provideService(FetchHttpClient.RequestInit, { redirect: "manual" }),
      Effect.map((response) => response.status),
      Effect.scoped,
      Effect.timeout(SEND_TIMEOUT),
      Effect.orElseSucceed(() => undefined),
    )
}
