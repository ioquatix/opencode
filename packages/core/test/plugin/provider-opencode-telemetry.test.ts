import { afterEach, beforeEach, describe, expect } from "bun:test"
import { SessionEvent } from "@opencode/schema/session-event"
import { Agent } from "@opencode/schema/agent"
import { Document, Info } from "@opencode/schema/config"
import { SessionMessage } from "@opencode/schema/session-message"
import { Money } from "@opencode/schema/money"
import { Effect, Exit, Fiber, Schema, Scope } from "effect"
import { TestClock } from "effect/testing"
import { HttpClient, HttpClientRequest, HttpClientResponse } from "effect/http"
import { Bus } from "@opencode/core/bus"
import { Config } from "@opencode/core/config"
import { ConfigIntegration } from "@opencode/core/config/integration"
import { Credential } from "@opencode/core/credential"
import { Integration } from "@opencode/core/integration"
import { Location } from "@opencode/core/location"
import { Model } from "@opencode/core/model"
import { Plugin } from "@opencode/core/plugin"
import { PluginHost } from "@opencode/core/plugin/host"
import { ChatGPTPlugin } from "@opencode/core/plugin/provider/chatgpt"
import { OpenAIPlugin } from "@opencode/core/plugin/provider/openai"
import { OpencodePlugin } from "@opencode/core/plugin/provider/opencode"
import { OpencodeTelemetry } from "@opencode/core/plugin/provider/opencode-telemetry"
import { Provider } from "@opencode/core/provider"
import { Session } from "@opencode/core/session"
import { withEnv } from "../fixture/env"
import { advance, drain } from "../lib/clock"
import { testEffect } from "../lib/effect"
import { PluginTestLayer } from "./fixture"

const it = testEffect(PluginTestLayer)

const server = "http://console.test"
const logsPath = "/api/otel/v1/logs"
const opus = Model.Ref.make({ providerID: Provider.ID.make("claude-agent"), id: Model.ID.make("opus") })
const gpt = Model.Ref.make({ providerID: Provider.ID.openai, id: Model.ID.make("gpt-6.1-sol") })
const consoleModel = Model.Ref.make({ providerID: Provider.ID.opencode, id: Model.ID.make("opus") })
const localModel = Model.Ref.make({ providerID: Provider.ID.make("ollama"), id: Model.ID.make("llama") })
const tokens = { input: 10, output: 20, reasoning: 5, cache: { read: 30, write: 40 } }
const marked = { subscription: true, model: "claude-opus-5-5", cacheWrite5m: 15, cacheWrite1h: 25 }

type Attribute = { key: string; value: { stringValue?: string; intValue?: number } }

/** Console's telemetry routes, called in-process so each exchange fits inside `drain`. */
function fakeConsole() {
  const state = {
    config: 200,
    telemetry: 200 as number | "hang",
    logs: [] as number[],
    hang: false,
    providers: {} as Record<string, object>,
    posts: [] as { authorization: string | null; records: Attribute[][]; body: unknown }[],
  }
  return {
    state,
    fetch: async (request: Request) => {
      const url = new URL(request.url)
      if (url.pathname === "/api/v2/config") {
        if (state.config === 401) return Response.json({ _tag: "Unauthorized" }, { status: 401 })
        if (state.config !== 200) return new Response(null, { status: state.config })
        return Response.json({ providers: state.providers })
      }
      if (url.pathname === "/api/v2/telemetry") {
        if (state.telemetry === "hang") return new Promise<Response>(() => {})
        if (state.telemetry !== 200) return new Response(null, { status: state.telemetry })
        return Response.json({ otlpLogsUrl: `${server}${logsPath}`, headers: { authorization: "Bearer octel_token" } })
      }
      if (url.pathname !== logsPath) return new Response(null, { status: 404 })
      if (state.hang) return new Promise<Response>(() => {})
      const body = (await request.json()) as {
        resourceLogs: { scopeLogs: { logRecords: { attributes: Attribute[] }[] }[] }[]
      }
      state.posts.push({
        authorization: request.headers.get("authorization"),
        records: body.resourceLogs.flatMap((resource) =>
          resource.scopeLogs.flatMap((scope) => scope.logRecords.map((record) => record.attributes)),
        ),
        body,
      })
      const status = state.logs.shift() ?? 200
      return Response.json({}, { status })
    },
  }
}

const inProcess = (console: { fetch: (request: Request) => Response | Promise<Response> }) =>
  Effect.provideService(
    HttpClient.HttpClient,
    HttpClient.make((request) =>
      HttpClientRequest.toWeb(request).pipe(
        Effect.orDie,
        Effect.flatMap((web) => Effect.promise(async () => console.fetch(web))),
        Effect.map((response) => HttpClientResponse.fromWeb(request, response)),
      ),
    ),
  )

const signIn = Credential.OAuth.make({
  type: "oauth",
  methodID: Integration.MethodID.make("device"),
  access: "access",
  refresh: "refresh",
  expires: Number.MAX_SAFE_INTEGER,
  metadata: { server, accountID: "user", orgID: "org" },
})

// models.dev and the provider plugins' catalogs, reduced to what classifying a step reads.
const usd = { input: Money.USDPerMillionTokens.make(1), output: Money.USDPerMillionTokens.make(1) }

const addModels = Effect.fn(function* (options: { readonly openaiCost?: boolean } = {}) {
  const providers = yield* Provider.Service
  yield* providers.transform((editor) => {
    const model = (ref: Model.Ref) => ({
      ...Model.Info.default(ref.providerID, ref.id),
      cost:
        options.openaiCost && ref.providerID === Provider.ID.openai
          ? [{ ...usd, cache: { read: usd.input, write: usd.input } }]
          : [],
    })
    for (const ref of [opus, gpt, consoleModel, localModel])
      editor.add({ info: { ...Provider.Info.empty(ref.providerID), activation: "enabled" }, models: [model(ref)] })
  })
})

// Every test publishes on the session the last `start` created.
let sessionID: Session.Info["id"]

const start = Effect.fn(function* (
  console: ReturnType<typeof fakeConsole>,
  options: {
    readonly consoleCredential?: Credential.OAuth | Credential.Key | null
    readonly openai?: Credential.OAuth | Credential.Key
    readonly openaiCost?: boolean
    readonly denied?: boolean
    /** A provider plugin to load for real, so its own transforms shape the catalog. */
    readonly plugin?: typeof OpenAIPlugin | typeof ChatGPTPlugin
    readonly onResolve?: (connection: { readonly type: string; readonly id?: string }) => void
  } = {},
) {
  const credentials = yield* Credential.Service
  const sessions = yield* Session.Service
  const location = yield* Location.Service
  sessionID = (yield* sessions.create({ location: { directory: location.directory } })).id
  yield* addModels({ openaiCost: options.openaiCost })
  if (options.consoleCredential !== null)
    yield* credentials.create({
      integrationID: Integration.ID.make("opencode"),
      value: options.consoleCredential ?? signIn,
    })
  if (options.openai)
    yield* credentials.create({ integrationID: Integration.ID.make("openai"), value: options.openai })
  if (options.plugin) {
    const providerHost = yield* PluginHost.make(yield* Plugin.Service, options.plugin.id)
    yield* options.plugin.effect(providerHost).pipe(
      Effect.provideService(
        HttpClient.HttpClient,
        HttpClient.make((request) => Effect.succeed(HttpClientResponse.fromWeb(request, new Response(null, { status: 503 })))),
      ),
    )
  }
  const host = yield* PluginHost.make(yield* Plugin.Service, "test")
  const config = yield* Config.Service
  const onResolve = options.onResolve
  const observed = onResolve
    ? {
        ...host,
        integration: {
          ...host.integration,
          connection: {
            ...host.integration.connection,
            resolve: (connection: Parameters<typeof host.integration.connection.resolve>[0]) => {
              onResolve(connection)
              return host.integration.connection.resolve(connection)
            },
          },
        },
      }
    : host
  yield* OpencodePlugin.effect(observed).pipe(
    inProcess(console),
    Effect.provideService(Config.Service, {
      ...config,
      // Managed `integration.use: deny` makes config keep a policies-only copy of the document.
      integration: (id, document) =>
        config.integration!(
          id,
          options.denied && document
            ? new ConfigIntegration.Document({ ...document, info: new Info({}) })
            : document,
        ),
    }),
  )
  yield* drain
})

const chatgpt = (methodID: string) =>
  Credential.OAuth.make({
    type: "oauth",
    methodID: Integration.MethodID.make(methodID),
    access: "chatgpt",
    refresh: "chatgpt",
    expires: Number.MAX_SAFE_INTEGER,
  })

let step = 0
const publishStep = Effect.fn(function* (
  model: Model.Ref,
  options: {
    readonly native?: SessionMessage.Native
    readonly cost?: number
    readonly failed?: boolean
    readonly id?: SessionMessage.ID
  } = {},
) {
  const bus = yield* Bus.Service
  const assistantMessageID = options.id ?? SessionMessage.ID.create()
  step++
  yield* bus.publish(SessionEvent.Step.Started, {
    sessionID,
    assistantMessageID,
    agent: Agent.ID.make("build"),
    model,
    started: step,
  })
  const finished = { sessionID, assistantMessageID, cost: Money.USD.make(options.cost ?? 0), tokens, native: options.native }
  if (options.failed) {
    yield* bus.publish(SessionEvent.Step.Failed, {
      ...finished,
      error: { type: "aborted", message: "interrupted" },
    })
    return assistantMessageID
  }
  yield* bus.publish(SessionEvent.Step.Ended, { ...finished, finish: "stop" })
  return assistantMessageID
})

const attribute = (record: Attribute[], key: string) => {
  const value = record.find((item) => item.key === key)?.value
  return value?.stringValue ?? value?.intValue
}
/** Lets real work finish without moving the clock, or stops after 2 seconds of real time. */
const settled = Effect.fnUntraced(function* (condition: () => boolean) {
  const deadline = performance.now() + 2_000
  while (!condition() && performance.now() < deadline) yield* TestClock.withLive(Effect.sleep("5 millis"))
})
const flushed = (console: ReturnType<typeof fakeConsole>, count = 1) =>
  TestClock.adjust("30 seconds").pipe(Effect.andThen(advance(() => console.state.posts.length >= count)))

// A developer's own Console exporter must not switch off what these tests exercise.
const exporterEnv = ["OPENCODE_CLAUDE_PATH", "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"]
const savedEnv = new Map<string, string | undefined>()

describe("OpencodeTelemetry", () => {
  beforeEach(() => {
    exporterEnv.forEach((key) => {
      savedEnv.set(key, process.env[key])
      delete process.env[key]
    })
  })

  afterEach(() => {
    exporterEnv.forEach((key) => {
      const value = savedEnv.get(key)
      if (value === undefined) delete process.env[key]
      else process.env[key] = value
    })
  })

  it.effect("reports a Claude subscription step with the record Console reads", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console)
      const id = yield* publishStep(opus, { native: { ...marked, requestID: "req_1" } })
      yield* flushed(console)
      const post = console.state.posts[0]
      expect(post.authorization).toBe("Bearer octel_token")
      expect(Object.fromEntries(post.records[0].map((item) => [item.key, attribute(post.records[0], item.key)]))).toMatchObject({
        "event.name": "opencode.step_usage",
        provider: "anthropic",
        model: "claude-opus-5-5",
        step_id: id,
        "session.id": sessionID,
        request_id: "req_1",
        input_tokens: 10,
        // Reasoning counts inside Console's output.
        output_tokens: 25,
        reasoning_tokens: 5,
        cache_read_tokens: 30,
        cache_write_5m_tokens: 15,
        cache_write_1h_tokens: 25,
      })
    }),
  )

  it.effect("sends every counter, zeros included, since Console drops a record missing one", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console)
      const bus = yield* Bus.Service
      yield* bus.publish(SessionEvent.UsageRecorded, {
        sessionID,
        source: "title",
        cost: Money.USD.zero,
        tokens: { input: 7, output: 0, reasoning: 0, cache: { read: 0, write: 0 } },
        model: opus,
        native: { subscription: true, model: "claude-opus-5-5" },
      })
      yield* flushed(console)
      const record = console.state.posts[0].records[0]
      expect(
        Object.fromEntries(
          [
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cache_read_tokens",
            "cache_write_5m_tokens",
            "cache_write_1h_tokens",
          ].map((key) => [key, attribute(record, key)]),
        ),
      ).toEqual({
        input_tokens: 7,
        output_tokens: 0,
        reasoning_tokens: 0,
        cache_read_tokens: 0,
        cache_write_5m_tokens: 0,
        cache_write_1h_tokens: 0,
      })
    }),
  )

  it.effect("sends a step every Location heard once in a batch", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console)
      const another = yield* PluginHost.make(yield* Plugin.Service, "other-location")
      yield* OpencodePlugin.effect(another).pipe(inProcess(console))
      yield* drain
      const id = yield* publishStep(opus, { native: marked })
      yield* flushed(console)
      expect(console.state.posts[0].records.map((record) => attribute(record, "step_id"))).toEqual([id])
    }),
  )

  it.effect("sends all cache writes as 5 minute ones when the step has no split", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console)
      yield* publishStep(opus, { native: { subscription: true, model: "claude-opus-5-5" } })
      yield* flushed(console)
      const record = console.state.posts[0].records[0]
      expect(attribute(record, "cache_write_5m_tokens")).toBe(40)
      expect(attribute(record, "cache_write_1h_tokens")).toBe(0)
    }),
  )

  it.effect("reports an interrupted step that recorded tokens", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console)
      yield* publishStep(opus, { native: marked, failed: true })
      yield* flushed(console)
      expect(console.state.posts).toHaveLength(1)
    }),
  )

  for (const source of ["title", "compaction"] as const)
    it.effect(`reports ${source} calls under their event ID`, () =>
      Effect.gen(function* () {
        const console = fakeConsole()
        yield* start(console)
        const bus = yield* Bus.Service
        yield* bus.publish(SessionEvent.UsageRecorded, {
          sessionID,
          source,
          cost: Money.USD.zero,
          tokens,
          model: opus,
          native: marked,
        })
        yield* flushed(console)
        expect(String(attribute(console.state.posts[0].records[0], "step_id"))).toStartWith("evt_")
      }),
    )

  it.effect("leaves out a summary a hook supplied, since no model was called", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console)
      const bus = yield* Bus.Service
      yield* bus.publish(SessionEvent.UsageRecorded, {
        sessionID,
        source: "compaction",
        cost: Money.USD.zero,
        tokens,
        native: marked,
      })
      yield* TestClock.adjust("30 seconds")
      yield* drain
      expect(console.state.posts).toEqual([])
    }),
  )

  for (const methodID of ["chatgpt-token-sharing", "chatgpt-browser", "chatgpt-headless"])
    it.effect(`reports an OpenAI step signed in with ${methodID}`, () =>
      Effect.gen(function* () {
        const console = fakeConsole()
        yield* start(console, { openai: chatgpt(methodID) })
        yield* publishStep(gpt)
        yield* flushed(console)
        const record = console.state.posts[0].records[0]
        expect(attribute(record, "provider")).toBe("openai")
        expect(attribute(record, "model")).toBe("gpt-6.1-sol")
        expect(attribute(record, "cache_write_5m_tokens")).toBe(40)
      }),
    )

  const signIns = [
    ["chatgpt-token-sharing", ChatGPTPlugin],
    ["chatgpt-browser", OpenAIPlugin],
    ["chatgpt-headless", OpenAIPlugin],
  ] as const
  for (const [methodID, plugin] of signIns)
    it.effect(`reports a step when ${plugin.id} really signs in with ${methodID}`, () =>
      Effect.gen(function* () {
        const console = fakeConsole()
        yield* start(console, { openai: chatgpt(methodID), plugin, openaiCost: true })
        yield* publishStep(gpt)
        yield* flushed(console)
        expect(attribute(console.state.posts[0].records[0], "provider")).toBe("openai")
      }),
    )

  it.effect("does not report a ChatGPT model the user gave its own endpoint in config", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      const config = yield* Config.Test
      yield* config.setEntries([
        new Document({
          type: "document",
          info: Schema.decodeUnknownSync(Info)({
            providers: { openai: { models: { [gpt.id]: { settings: { baseURL: "https://proxy.test/v1" } } } } },
          }),
        }),
      ])
      yield* start(console, { openai: chatgpt("chatgpt-browser"), plugin: OpenAIPlugin, openaiCost: true })
      yield* publishStep(gpt)
      yield* TestClock.adjust("30 seconds")
      yield* drain
      expect(console.state.posts).toEqual([])
    }),
  )

  it.effect("reports a ChatGPT step without resolving the stored credential", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      const resolved: (string | undefined)[] = []
      yield* start(console, {
        openai: chatgpt("chatgpt-browser"),
        onResolve: (connection) => resolved.push(connection.id),
      })
      const credentials = yield* Credential.Service
      const stored = yield* credentials.list(Integration.ID.make("openai"))
      yield* publishStep(gpt)
      yield* flushed(console)
      expect(console.state.posts).toHaveLength(1)
      expect(resolved).not.toContain(stored[0]?.id)
    }),
  )

  const neverReported: {
    readonly name: string
    readonly start?: Parameters<typeof start>[1]
    readonly publish: Effect.Effect<unknown, never, Bus.Service>
  }[] = [
    { name: "a Claude step without the plan's marker", publish: publishStep(opus) },
    {
      name: "a Claude step a wrapper sent to the gateway",
      publish: publishStep(opus, { native: { model: "opus", fallback: "opencode" } }),
    },
    { name: "a step with a recorded cost", publish: publishStep(opus, { native: marked, cost: 1 }) },
    { name: "the Console's own provider", publish: publishStep(consoleModel, { native: marked }) },
    { name: "a local provider", publish: publishStep(localModel, { native: marked }) },
    {
      name: "a gateway model name",
      publish: publishStep(opus, { native: { ...marked, model: "opencode-go/claude-opus-5-5" } }),
    },
    { name: "an OpenAI step on an API key", start: { openai: Credential.Key.make({ type: "key", key: "sk" }) }, publish: publishStep(gpt) },
    {
      name: "a priced OpenAI model even when signed in with ChatGPT",
      start: { openai: chatgpt("chatgpt-browser"), openaiCost: true },
      publish: publishStep(gpt),
    },
  ]
  for (const item of neverReported)
    it.effect(`does not report ${item.name}`, () =>
      Effect.gen(function* () {
        const console = fakeConsole()
        yield* start(console, item.start)
        yield* item.publish
        yield* TestClock.adjust("30 seconds")
        yield* drain
        expect(console.state.posts).toEqual([])
      }),
    )

  it.effect("does not report a provider the Console's config defines", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      console.state.providers = { "claude-agent": {} }
      yield* start(console)
      yield* publishStep(opus, { native: marked })
      yield* TestClock.adjust("30 seconds")
      yield* drain
      expect(console.state.posts).toEqual([])
    }),
  )

  it.effect("does not report a provider with its own endpoint in config", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      const config = yield* Config.Test
      yield* config.setEntries([
        new Document({
          type: "document",
          info: Schema.decodeUnknownSync(Info)({
            providers: { openai: { settings: { baseURL: "https://proxy.test/v1" } } },
          }),
        }),
      ])
      yield* start(console, { openai: chatgpt("chatgpt-browser") })
      yield* publishStep(gpt)
      yield* TestClock.adjust("30 seconds")
      yield* drain
      expect(console.state.posts).toEqual([])
    }),
  )

  it.effect("does not report while the Console's workspace answers 404", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      console.state.telemetry = 404
      yield* start(console)
      yield* publishStep(opus, { native: marked })
      yield* TestClock.adjust("30 seconds")
      yield* drain
      expect(console.state.posts).toEqual([])
    }),
  )

  it.effect("does not report while managed policy denies the integration", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console, { denied: true })
      yield* publishStep(opus, { native: marked })
      yield* TestClock.adjust("30 seconds")
      yield* drain
      expect(console.state.posts).toEqual([])
    }),
  )

  it.effect("does not report while signed out of the Console", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console, { consoleCredential: null })
      yield* publishStep(opus, { native: marked })
      yield* TestClock.adjust("30 seconds")
      yield* drain
      expect(console.state.posts).toEqual([])
    }),
  )

  it.effect("does not report on an API key sign-in", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      const requests: string[] = []
      const counted = {
        ...console,
        fetch: (request: Request) => {
          requests.push(new URL(request.url).pathname)
          return console.fetch(request)
        },
      }
      yield* start(counted, { consoleCredential: Credential.Key.make({ type: "key", key: "key", metadata: { server } }) })
      yield* publishStep(opus, { native: marked })
      yield* TestClock.adjust("30 seconds")
      yield* drain
      expect(requests).not.toContain("/api/v2/telemetry")
      expect(console.state.posts).toEqual([])
    }),
  )

  it.effect("keeps the telemetry target out of the config document", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console)
      const config = yield* Config.Service
      const document = (yield* config.entries()).find(
        (entry): entry is ConfigIntegration.Document => entry instanceof ConfigIntegration.Document,
      )
      expect(document).toBeDefined()
      expect(JSON.stringify(document)).not.toContain("octel_token")
      expect(JSON.stringify(document)).not.toContain(logsPath)
    }),
  )

  it.effect("holds records until the 30 second window ends", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console)
      yield* publishStep(opus, { native: marked })
      yield* publishStep(opus, { native: marked })
      yield* drain
      expect(console.state.posts).toEqual([])
      yield* flushed(console)
      expect(console.state.posts).toHaveLength(1)
      expect(console.state.posts[0].records).toHaveLength(2)
    }),
  )

  it.effect("sends at once when a full batch accumulates", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console)
      yield* Effect.forEach(
        Array.from({ length: OpencodeTelemetry.BATCH_SIZE }),
        () => publishStep(opus, { native: marked }),
        { discard: true },
      )
      // The clock stays still: this must not wait for the 30 second window.
      yield* settled(() => console.state.posts.length >= 1)
      expect(console.state.posts[0].records).toHaveLength(OpencodeTelemetry.BATCH_SIZE)
    }),
  )

  it.effect("keeps only the newest records when Console stays out of reach", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console)
      console.state.logs = [503]
      const ids = yield* Effect.forEach(Array.from({ length: OpencodeTelemetry.MAX_RECORDS + 1 }), () =>
        publishStep(opus, { native: marked }),
      )
      yield* advance(() => console.state.posts.length >= 1)
      console.state.posts.length = 0
      yield* TestClock.adjust("5 seconds")
      yield* advance(() => console.state.posts.length >= 5)
      const sent = console.state.posts.flatMap((post) => post.records.map((record) => attribute(record, "step_id")))
      expect(sent).toHaveLength(OpencodeTelemetry.MAX_RECORDS)
      expect(sent).not.toContain(ids[0])
      expect(sent).toContain(ids[ids.length - 1])
    }),
  )

  it.effect("retries with the same record after a 503, then stops once Console took it", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console)
      console.state.logs = [503]
      yield* publishStep(opus, { native: marked })
      yield* flushed(console)
      yield* TestClock.adjust("2 seconds")
      yield* drain
      expect(console.state.posts).toHaveLength(1)
      yield* TestClock.adjust("1 second")
      yield* advance(() => console.state.posts.length >= 2)
      expect(console.state.posts[1].body).toEqual(console.state.posts[0].body)
      yield* TestClock.adjust("2 minutes")
      yield* drain
      expect(console.state.posts).toHaveLength(2)
    }),
  )

  for (const status of [400, 401, 403, 404])
    it.effect(`drops a batch Console answers ${status}`, () =>
      Effect.gen(function* () {
        const console = fakeConsole()
        yield* start(console)
        console.state.logs = [status]
        yield* publishStep(opus, { native: marked })
        yield* flushed(console)
        yield* TestClock.adjust("2 minutes")
        yield* drain
        expect(console.state.posts).toHaveLength(1)
      }),
    )

  it.effect("drops what's buffered when the workspace stops collecting usage", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console)
      // Published 20 seconds before the next check, so the 30 second window can't send it first.
      yield* TestClock.adjust("40 seconds")
      yield* publishStep(opus, { native: marked })
      yield* drain
      console.state.telemetry = 404
      yield* TestClock.adjust("20 seconds")
      yield* drain
      yield* TestClock.adjust("2 minutes")
      yield* drain
      expect(console.state.posts).toEqual([])
    }),
  )

  for (const [name, telemetry] of [
    ["hanging", "hang"],
    ["failing", 500],
  ] as const)
    it.effect(`loads config while the telemetry endpoint is ${name}`, () =>
      Effect.gen(function* () {
        const console = fakeConsole()
        console.state.telemetry = telemetry
        yield* start(console)
        const config = yield* Config.Service
        expect(
          (yield* config.entries()).some((entry) => entry instanceof ConfigIntegration.Document),
        ).toBe(true)
      }),
    )

  it.effect("stops reporting when the Console revokes the sign-in, and keeps its policy", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console)
      console.state.config = 401
      yield* TestClock.adjust("1 minute")
      yield* drain
      yield* publishStep(opus, { native: marked })
      yield* TestClock.adjust("30 seconds")
      yield* drain
      expect(console.state.posts).toEqual([])
      const config = yield* Config.Service
      expect(
        (yield* config.entries()).some((entry) => entry instanceof ConfigIntegration.Document),
      ).toBe(true)
    }),
  )

  it.effect("keeps reporting to the last target through a Console outage", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      yield* start(console)
      console.state.config = 500
      console.state.telemetry = 500
      yield* TestClock.adjust("1 minute")
      yield* drain
      yield* TestClock.adjust("1 minute")
      yield* drain
      yield* publishStep(opus, { native: marked })
      yield* flushed(console)
      expect(console.state.posts).toHaveLength(1)
    }),
  )

  const closeable = Effect.fn(function* (console: ReturnType<typeof fakeConsole>) {
    const lease = yield* Scope.fork(yield* Scope.Scope)
    yield* start(console).pipe(Scope.provide(lease))
    return lease
  })

  it.effect("sends what's left when the scope closes", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      const lease = yield* closeable(console)
      yield* publishStep(opus, { native: marked })
      yield* drain
      yield* Scope.close(lease, Exit.void)
      expect(console.state.posts).toHaveLength(1)
    }),
  )

  it.effect("gives up on the closing flush after 3 seconds", () =>
    Effect.gen(function* () {
      const console = fakeConsole()
      const lease = yield* closeable(console)
      console.state.hang = true
      yield* publishStep(opus, { native: marked })
      yield* drain
      const closing = yield* Scope.close(lease, Exit.void).pipe(Effect.forkChild({ startImmediately: true }))
      yield* drain
      yield* TestClock.adjust("3 seconds")
      yield* Fiber.join(closing)
      expect(console.state.posts).toEqual([])
    }),
  )

  const exporters: Record<string, Record<string, string>> = {
    "a wrapper that exports to Console": { OPENCODE_CLAUDE_PATH: "/Users/a/.local/bin/claude-console-usage" },
    "an OTLP exporter pointed at Console": {
      OTEL_EXPORTER_OTLP_LOGS_ENDPOINT: "https://opencode.ai/console/api/otel/v1/logs",
    },
  }
  for (const [name, env] of Object.entries(exporters))
    it.effect(`skips Claude steps while ${name} already reports them`, () =>
      withEnv(env, () =>
        Effect.gen(function* () {
          const console = fakeConsole()
          yield* start(console)
          yield* publishStep(opus, { native: marked })
          yield* TestClock.adjust("30 seconds")
          yield* drain
          expect(console.state.posts).toEqual([])
        }),
      ),
    )
})
