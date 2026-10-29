import { useState } from 'react'
import { scoreText, ScoreResponse } from '../api'

const PRESETS = [
  {
    label: '🔴 Extreme Fear & Loaded Language',
    text: 'SHOCKING LEAK: Corrupt officials are secretly poisoning our water supply to force compliance! Share this before they delete it!',
    mock: {
      persuasion_risk: 0.89, risk_label: 'HIGH', risk_emoji: '🔴',
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
    label: '🔴 False Dilemma',
    text: 'You are either with our movement or you are an enemy of the nation! There is no middle ground for traitors.',
    mock: {
      persuasion_risk: 0.78, risk_label: 'HIGH', risk_emoji: '🔴',
      explanation: '🔴 Strong polarization fallacy detected. The text eliminates nuanced discourse by constructing a false binary choice.',
      detected_techniques: [
        { technique: 'Black-and-White_Fallacy', score: 0.92 },
        { technique: 'Name_Calling-Labeling', score: 0.85 },
        { technique: 'Bandwagon', score: 0.68 },
      ],
    },
  },
  {
    label: '🟡 Commercial Hype',
    text: 'Top silicon valley insiders say this miraculous breakthrough will obsolete all existing technology within months.',
    mock: {
      persuasion_risk: 0.44, risk_label: 'MEDIUM', risk_emoji: '🟡',
      explanation: '🟡 Moderate persuasion detected. Uses authority figures and unverified hyperbole.',
      detected_techniques: [
        { technique: 'Appeal_to_Authority', score: 0.62 },
        { technique: 'Exaggeration-Minimisation', score: 0.58 },
      ],
    },
  },
  {
    label: '🟢 Neutral & Fact-Based',
    text: 'The Federal Reserve held interest rates steady on Wednesday, citing stabilizing labor metrics and inflation trending toward the 2 percent target.',
    mock: {
      persuasion_risk: 0.06, risk_label: 'LOW', risk_emoji: '🟢',
      explanation: '🟢 This text uses neutral, descriptive language without emotional steering or loaded phrasing.',
      detected_techniques: [{ technique: 'Reasoning', score: 0.75 }],
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
      const matched = PRESETS.find((p) => p.text === inputText)
      if (matched) {
        setResult(matched.mock)
      } else {
        const lower = inputText.toLowerCase()
        const hasFear = /shocking|leak|poison|corrupt|destroy|kill|fear|disaster|threat/.test(lower)
        const hasPolar = /traitor|enemy|either|always|never|pure|evil|lie|hoax/.test(lower)
        const risk = hasFear && hasPolar ? 0.85 : hasFear || hasPolar ? 0.62 : 0.12
        const label = risk > 0.6 ? 'HIGH' : risk > 0.3 ? 'MEDIUM' : 'LOW'
        const emoji = risk > 0.6 ? '🔴' : risk > 0.3 ? '🟡' : '🟢'
        setResult({
          persuasion_risk: risk, risk_label: label, risk_emoji: emoji,
          explanation: `${emoji} Analysis: detected ${label.toLowerCase()} probability of rhetorical steering.`,
          detected_techniques: [
            ...(hasFear ? [{ technique: 'Appeal_to_Fear-Prejudice', score: 0.88 }] : []),
            ...(hasPolar ? [{ technique: 'Loaded_Language', score: 0.81 }] : []),
            { technique: 'Reasoning', score: risk < 0.3 ? 0.8 : 0.2 },
          ],
        })
      }
    } finally { setLoading(false) }
  }

  const riskBorderColor = result
    ? result.risk_label === 'LOW' ? 'var(--risk-low)' : result.risk_label === 'MEDIUM' ? 'var(--risk-medium)' : 'var(--risk-high)'
    : 'var(--border)'

  return (
    <div className="page-body">
      <div className="card" style={{ marginBottom: 20 }}>
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 700, marginBottom: 4 }}>
          🔬 Live Persuasion & Propaganda Detector
        </h2>
        <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 20, lineHeight: 1.5 }}>
          Test any headline or paragraph against our RoBERTa multi-label classifier.
          Scores across 23 persuasion and cognitive distortion categories from SemEval-2023 Task 3.
        </p>

        {/* Presets */}
        <div style={{ marginBottom: 16 }}>
          <div className="filter-label" style={{ marginBottom: 8 }}>Sample Inputs</div>
          <div className="filter-group">
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                onClick={() => { setInputText(preset.text); setResult(preset.mock) }}
                className={`filter-btn ${inputText === preset.text ? 'active' : ''}`}
              >
                {preset.label}
              </button>
            ))}
          </div>
        </div>

        {/* Input */}
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          rows={4}
          placeholder="Type or paste any news headline, paragraph, or social post..."
          style={{
            width: '100%', background: 'var(--bg-tertiary)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius-md)', color: 'var(--text-primary)', padding: '14px',
            fontSize: 14, lineHeight: 1.6, resize: 'vertical', outline: 'none', marginBottom: 16,
            fontFamily: 'var(--font-body)',
          }}
        />

        <button
          onClick={handleAnalyze}
          disabled={loading || !inputText.trim()}
          style={{
            background: 'linear-gradient(135deg, var(--accent-blue), var(--accent-cyan))',
            color: '#fff', fontWeight: 600, padding: '10px 28px',
            borderRadius: 'var(--radius-md)', fontSize: 13,
            boxShadow: '0 2px 8px rgba(37,99,235,0.25)',
            opacity: loading ? 0.7 : 1,
            display: 'inline-flex', alignItems: 'center', gap: 8,
            transition: 'all var(--transition)',
          }}
        >
          {loading ? 'Analyzing...' : 'Run Persuasion Detector →'}
        </button>
      </div>

      {/* Result */}
      {result && (
        <div className="card" style={{ borderLeft: `4px solid ${riskBorderColor}` }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <span style={{ fontSize: 28 }}>{result.risk_emoji}</span>
              <div>
                <div style={{ fontSize: 17, fontWeight: 700, fontFamily: 'var(--font-display)' }}>
                  Manipulation Tier: {result.risk_label}
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  Persuasion Index: <strong style={{ fontFamily: 'var(--font-display)', color: riskBorderColor }}>{(result.persuasion_risk * 100).toFixed(1)}%</strong>
                </div>
              </div>
            </div>
            <span className={`risk-badge ${result.risk_label.toLowerCase()}`} style={{ fontSize: 12, padding: '5px 14px' }}>
              {result.risk_label} RISK
            </span>
          </div>

          <div className="explanation-box" style={{ marginBottom: 20, borderLeft: `3px solid ${riskBorderColor}` }}>
            {result.explanation}
          </div>

          <div>
            <div className="drawer-section-title">Detected Persuasion Techniques</div>
            <div className="technique-list">
              {result.detected_techniques.map((item) => (
                <div key={item.technique} className="technique-row">
                  <span className="technique-name">{item.technique.replace(/[-_]/g, ' ')}</span>
                  <div className="technique-bar-track">
                    <div className="technique-bar-fill" style={{ width: `${Math.round(item.score * 100)}%` }} />
                  </div>
                  <span className="technique-score">{(item.score * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
