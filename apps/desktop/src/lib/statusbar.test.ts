import { describe, expect, it } from 'vitest'

import { cacheHitLabel, contextBarLabel, tokensPerSecondLabel, usageContextLabel } from '@/lib/statusbar'
import type { UsageStats } from '@/types/hermes'

const base = { calls: 0, input: 0, output: 0, total: 0 }

describe('statusbar usage readouts', () => {
  it('shows the detected model window without inventing an initial measurement', () => {
    const unknownContext: UsageStats = {
      ...base,
      context_max: 900_000,
      context_percent: null,
      context_used: null
    }

    expect(usageContextLabel(unknownContext)).toBe('—/900k')
    expect(contextBarLabel(unknownContext)).toBe('')
  })

  it('paints the backend cache-hit and throughput fields, and stays blank when they are absent', () => {
    // The backend omits both fields (rather than sending 0) when it has no data
    // — a provider with no cache reads, or a session before its first call.
    expect(cacheHitLabel(base)).toBe('')
    expect(tokensPerSecondLabel(base)).toBe('')

    expect(cacheHitLabel({ ...base, cache_hit_pct: 87 })).toBe('87%')
    expect(tokensPerSecondLabel({ ...base, avg_tps: 41.6 })).toBe('42 t/s')
  })
})
