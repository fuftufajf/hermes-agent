import { describe, expect, it } from 'vitest'

import type { UsageStats } from '@/types/hermes'

import { contextBarLabel, usageContextLabel } from './statusbar'

const unknownContext: UsageStats = {
  calls: 0,
  context_max: 900_000,
  context_percent: null,
  context_used: null,
  input: 0,
  output: 0,
  total: 0
}

describe('statusbar context usage', () => {
  it('shows the detected model window without inventing an initial measurement', () => {
    expect(usageContextLabel(unknownContext)).toBe('—/900k')
    expect(contextBarLabel(unknownContext)).toBe('')
  })
})
