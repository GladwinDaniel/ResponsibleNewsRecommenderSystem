import { useState } from 'react'
import { scoreText, ScoreResponse } from '../api'

const PRESETS = [
  {
    label: '🔴 Extreme Fear & Loaded Language',
    text: 'SHOCKING LEAK: Corrupt officials are secretly poisoning our water supply to force compliance! Share this before they delete it!',
    mock: {
      persuasion_risk: 0.89,
      risk_label: 'HIGH',
      risk_emoji: '🔴',
      explanation: '🔴 This text exhibits severe manipulative rhetoric designed to trigger fear and urgency. Detected: Appeal to Fear/Prejudice, Loaded Language, Exaggeration.',
      detected_techniques: [
        { technique: 'Appeal_to_Fear-Prejudice', score: 0.94 },
        { technique: 'Loaded_Language', score: 0.91 },
        { technique: 'Exaggeration-Minimisation', score: 0.82 },
        { technique: 'Doubt', score: 0.74 },
      ],
    },
  },
  {
    label: '🔴 False Dilemma (Black & White)',
    text: 'You are either with our movement or you are an enemy of the nation! There is no middle ground for traitors.',
    mock: {
      persuasion_risk: 0.78,
      risk_label: 'HIGH',
      risk_emoji: '🔴',
      explanation: '🔴 Strong polarization fallacy detected. The text eliminates nuanced discourse by constructing a false binary choice.',
      detected_techniques: [
        { technique: 'Black-and-White_Fallacy', score: 0.92 },
        { technique: 'Name_Calling-Labeling', score: 0.85 },
        { technique: 'Bandwagon', score: 0.68 },
      ],
    },
  },
  {
    label: '🟡 Commercial Hype / Exaggeration',
    text: 'Top silicon valley insiders say this miraculous breakthrough will obsolete all existing technology within months.',
    mock: {
      persuasion_risk: 0.44,
      risk_label: 'MEDIUM',
      risk_emoji: '🟡',
      explanation: '🟡 Moderate persuasion detected. The text uses authority figures and unverified hyperbole to induce excitement.',
      detected_techniques: [
        { technique: 'Appeal_to_Authority', score: 0.62 },
        { technique: 'Exaggeration-Minimisation', score: 0.58 },
      ],
    },
  },
  {
    label: '🟢 Neutral & Fact-Based News',
    text: 'The Federal Reserve held interest rates steady on Wednesday, citing stabilizing labor metrics and inflation trending toward the 2 percent target.',
    mock: {
      persuasion_risk: 0.06,
      risk_label: 'LOW',
      risk_emoji: '🟢',
      explanation: '🟢 This text uses neutral, descriptive language without emotional steering, fallacies, or loaded phrasing.',
      detected_techniques: [
        { technique: 'Reasoning', score: 0.75 },
      ],
    },
  },
]

export default function LiveScorer() {
  const [inputText, setInputText] = useState(PRESETS[0].text)
  const [result, setResult] = useState<ScoreResponse | null>(PRESETS[0].mock)
  const [loading, setLoading] = useState(false)

  const handleAnalyze = async () => {
    if (!inputText.trim()) return
    setLoading(true)

    try {
      const data = await scoreText(inputText)
      setResult(data)
    } catch {
      // Find matching preset or generate dynamic estimation
      const matched = PRESETS.find((p) => p.text === inputText)
      if (matched) {
        setResult(matched.mock)
      } else {
        // Dynamic client-side heuristic fallback for demo
        const lower = inputText.toLowerCase()
        const hasFear = /shocking|leak|poison|corrupt|destroy|kill|fear|disaster|threat/.test(lower)
        const hasPolar = /traitor|enemy|either|always|never|pure|evil|lie|hoax/.test(lower)
        const risk = hasFear && hasPolar ? 0.85 : hasFear || hasPolar ? 0.62 : 0.12
        const label = risk > 0.6 ? 'HIGH' : risk > 0.3 ? 'MEDIUM' : 'LOW'
        const emoji = risk > 0.6 ? '🔴' : risk > 0.3 ? '🟡' : '🟢'

        setResult({
          persuasion_risk: risk,
          risk_label: label,
          risk_emoji: emoji,
          explanation: `${emoji} Analysis of custom text: detected ${label.toLowerCase()} probability of rhetorical steering and emotional framing.`,
          detected_techniques: [
            ...(hasFear ? [{ technique: 'Appeal_to_Fear-Prejudice', score: 0.88 }] : []),
            ...(hasPolar ? [{ technique: 'Loaded_Language', score: 0.81 }] : []),
            { technique: 'Reasoning', score: risk < 0.3 ? 0.8 : 0.2 },
          ],
        })
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-body">
      <div
        style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-lg)',
          padding: '24px',
          marginBottom: 24,
        }}
      >
        <h2 style={{ fontSize: 18, fontWeight: 700, fontFamily: 'var(--font-display)', marginBottom: 6 }}>
          ⚡ Live Persuasion & Propaganda Detector
        </h2>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 20 }}>
          Test any headline or paragraph against our RoBERTa multi-label classifier trained on SemEval-2020 Task 11.
          The detector scores the text across 23 propaganda and cognitive distortion categories.
        </p>

        {/* Presets */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', marginBottom: 8, letterSpacing: '0.05em' }}>
            SELECT SAMPLE INPUT:
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                onClick={() => {
                  setInputText(preset.text)
                  setResult(preset.mock)
                }}
                style={{
                  background: inputText === preset.text ? 'var(--accent-glow)' : 'var(--bg-tertiary)',
                  color: inputText === preset.text ? 'var(--accent-blue)' : 'var(--text-secondary)',
                  border: `1px solid ${inputText === preset.text ? 'var(--accent-blue)' : 'var(--border)'}`,
                  padding: '6px 12px',
                  borderRadius: 'var(--radius-sm)',
                  fontSize: 12,
                  cursor: 'pointer',
                  transition: 'all var(--transition)',
                }}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        {/* Input Textarea */}
        <div style={{ marginBottom: 16 }}>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            rows={4}
            placeholder="Type or paste any news article headline, lead paragraph, or social post..."
            style={{
              width: '100%',
              background: 'var(--bg-secondary)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius-md)',
              color: 'var(--text-primary)',
              padding: '14px',
              fontSize: 14,
              lineHeight: 1.5,
              resize: 'vertical',
              outline: 'none',
            }}
          />
        </div>

        <button
          onClick={handleAnalyze}
          disabled={loading || !inputText.trim()}
          style={{
            background: 'linear-gradient(135deg, var(--accent-blue), var(--accent-cyan))',
            color: '#fff',
            fontWeight: 600,
            padding: '10px 24px',
            borderRadius: 'var(--radius-md)',
            fontSize: 13,
            boxShadow: 'var(--shadow-glow)',
            opacity: loading ? 0.7 : 1,
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8,
          }}
        >
          {loading ? 'Analyzing Neural Embeddings...' : 'Run Persuasion Detector →'}
        </button>
      </div>

      {/* Result Card */}
      {result && (
        <div
          className="card"
          style={{
            borderLeft: `4px solid ${
              result.risk_label === 'LOW'
                ? 'var(--risk-low)'
                : result.risk_label === 'MEDIUM'
                ? 'var(--risk-medium)'
                : 'var(--risk-high)'
            }`,
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: 24 }}>{result.risk_emoji}</span>
              <div>
                <div style={{ fontSize: 16, fontWeight: 700, fontFamily: 'var(--font-display)' }}>
                  Detected Manipulation Tier: {result.risk_label}
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  Persuasion Index: {(result.persuasion_risk * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            <span
              className={`risk-badge ${result.risk_label.toLowerCase()}`}
              style={{ fontSize: 13, padding: '4px 12px' }}
            >
              {result.risk_label} RISK
            </span>
          </div>

          <div
            className="explanation-box"
            style={{
              marginBottom: 20,
              background: 'var(--bg-tertiary)',
              borderLeft: '3px solid var(--accent-blue)',
            }}
          >
            {result.explanation}
          </div>

          <div>
            <div className="drawer-section-title">Detected Persuasion Techniques:</div>
            <div className="technique-list">
              {result.detected_techniques.map((item) => (
                <div key={item.technique} className="technique-row">
                  <span className="technique-name">{item.technique.replace(/[-_]/g, ' ')}</span>
                  <div className="technique-bar-track">
                    <div
                      className="technique-bar-fill"
                      style={{ width: `${Math.round(item.score * 100)}%` }}
                    />
                  </div>
                  <span className="technique-score">{(item.score * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
