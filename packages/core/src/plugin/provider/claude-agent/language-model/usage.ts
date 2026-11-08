export * as ClaudeUsage from "./usage.js"

import type { LanguageModelV3Usage } from "@ai-sdk/provider"

interface Usage {
  readonly input_tokens?: number | null
  readonly cache_read_input_tokens?: number | null
  readonly cache_creation_input_tokens?: number | null
  readonly output_tokens?: number
  readonly output_tokens_details?: { readonly thinking_tokens?: number } | null
  readonly cache_creation?: {
    readonly ephemeral_5m_input_tokens?: number
    readonly ephemeral_1h_input_tokens?: number
  } | null
}

export function make() {
  const messages = new Map<
    string,
    { input: number; read: number; write: number; write5m: number; write1h: number; output: number; reasoning: number }
  >()
  let last: string | undefined

  function update(id: string, usage: Usage) {
    const saved = messages.get(id)
    const counts = saved ?? { input: 0, read: 0, write: 0, write5m: 0, write1h: 0, output: 0, reasoning: 0 }
    if (!saved) {
      messages.set(id, counts)
      last = id
    }
    counts.input = Math.max(counts.input, usage.input_tokens ?? 0)
    counts.read = Math.max(counts.read, usage.cache_read_input_tokens ?? 0)
    counts.write = Math.max(counts.write, usage.cache_creation_input_tokens ?? 0)
    counts.write5m = Math.max(counts.write5m, usage.cache_creation?.ephemeral_5m_input_tokens ?? 0)
    counts.write1h = Math.max(counts.write1h, usage.cache_creation?.ephemeral_1h_input_tokens ?? 0)
    counts.output = Math.max(counts.output, usage.output_tokens ?? 0)
    counts.reasoning = Math.max(counts.reasoning, usage.output_tokens_details?.thinking_tokens ?? 0)
  }

  return {
    update,
    result(usage: Usage) {
      // Result usage sums the whole turn; only apply it when there was a single model call.
      if (messages.size <= 1) update(last ?? "result", usage)
    },
    /** The last call's cache writes by lifetime, as `snapshot` counts its cache writes. Undefined when unreported. */
    cacheWrites() {
      const lastCall = last === undefined ? undefined : messages.get(last)
      if (!lastCall || lastCall.write5m + lastCall.write1h === 0) return undefined
      return { cacheWrite5m: lastCall.write5m, cacheWrite1h: lastCall.write1h }
    },
    /** How many model calls this step made. */
    calls: () => messages.size,
    snapshot(): LanguageModelV3Usage {
      let output = 0
      let reasoning = 0
      for (const counts of messages.values()) {
        output += counts.output
        reasoning += counts.reasoning
      }
      // Context occupancy is the last API call's input, not the sum of repeated contexts.
      const lastCall = last === undefined ? undefined : messages.get(last)
      const inputTotal = lastCall ? lastCall.input + lastCall.read + lastCall.write : undefined
      return {
        inputTokens: {
          total: inputTotal,
          noCache: lastCall?.input,
          cacheRead: lastCall?.read,
          cacheWrite: lastCall?.write,
        },
        outputTokens: { total: output, reasoning, text: output - reasoning },
      }
    },
  }
}
