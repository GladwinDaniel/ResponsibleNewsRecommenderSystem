import { Article } from '../api'

interface ArticleDrawerProps {
  article: Article | null
  onClose: () => void
}

const FULL_LABELS: Record<string, string> = {
  Appeal_to_Authority: 'Appeal to Authority',
  'Appeal_to_Fear-Prejudice': 'Appeal to Fear / Prejudice',
  Bandwagon: 'Bandwagon',
  'Black-and-White_Fallacy': 'Black-and-White Fallacy',
  Causal_Oversimplification: 'Causal Oversimplification',
  Doubt: 'Doubt',
  'Exaggeration-Minimisation': 'Exaggeration / Minimisation',
  'Flag-Waving': 'Flag-Waving',
  Glittering_generalities: 'Glittering Generalities',
  Loaded_Language: 'Loaded Language',
  'Name_Calling-Labeling': 'Name Calling / Labeling',
  Obfuscation: 'Obfuscation',
  Reasoning: 'Reasoning (Constructive)',
  Red_Herring: 'Red Herring',
  Repetition: 'Repetition',
  Slogans: 'Slogans',
  'Thought-terminating_Clichés': 'Thought-terminating Clichés',
  Transfer: 'Transfer',
  Whataboutism: 'Whataboutism',
  Straw_Men: 'Straw Men',
  Appeal_to_Hypocrisy: 'Appeal to Hypocrisy',
  Consequential_Oversimplification: 'Consequential Oversimplification',
  Appeal_to_Time: 'Appeal to Time',
}

export default function ArticleDrawer({ article, onClose }: ArticleDrawerProps) {
  if (!article) return null

  const riskClass  = article.risk_label.toLowerCase()
  const riskColors = { low: 'var(--risk-low)', medium: 'var(--risk-medium)', high: 'var(--risk-high)' }
  const riskColor  = riskColors[riskClass as keyof typeof riskColors] ?? 'var(--text-muted)'

  const allTechniques = [...article.top_techniques].sort((a, b) => b.score - a.score)

  return (
    <>
      <div className="drawer-overlay" onClick={onClose} aria-hidden />
      <aside className="drawer" role="dialog" aria-label="Article detail panel" aria-modal>
        <div className="drawer-header">
          <button className="drawer-close" onClick={onClose} id="drawer-close-btn" aria-label="Close panel">✕</button>

          <h2 style={{ fontFamily: 'var(--font-editorial)', fontSize: 20, fontWeight: 600, lineHeight: 1.35, marginBottom: 14, color: 'var(--text-primary)' }}>
            {article.title}
          </h2>

          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <span className={`risk-badge ${riskClass}`} style={{ fontSize: 12, padding: '4px 12px' }}>
              {article.risk_emoji} {article.risk_label} RISK
            </span>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              Score: <strong style={{ color: 'var(--accent-blue)', fontFamily: 'var(--font-display)' }}>
                {article.responsible_score.toFixed(3)}
              </strong>
            </span>
            {article.url && (
              <a href={article.url} target="_blank" rel="noopener noreferrer"
                style={{ fontSize: 11, color: 'var(--accent-cyan)', marginLeft: 'auto', fontWeight: 600 }}>
                Open ↗
              </a>
            )}
          </div>
        </div>

        <div className="drawer-body">
          {/* Score Breakdown */}
          <div className="drawer-section">
            <div className="drawer-section-title">Score Breakdown</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {[
                { label: 'Responsible Score (final)', val: article.responsible_score, color: 'var(--accent-cyan)' },
                { label: 'Relevance (GNN)',            val: article.relevance,         color: 'var(--accent-blue)' },
                { label: 'Persuasion Risk',            val: article.persuasion_risk,   color: riskColor },
              ].map(({ label, val, color }) => (
                <div key={label}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                    <span style={{ fontSize: 12, color: 'var(--text-secondary)', fontWeight: 500 }}>{label}</span>
                    <span style={{ fontSize: 14, fontWeight: 700, color, fontFamily: 'var(--font-display)' }}>
                      {(val * 100).toFixed(1)}
                    </span>
                  </div>
                  <div className="technique-bar-track" style={{ height: 8 }}>
                    <div className="technique-bar-fill" style={{ width: `${val * 100}%`, background: color, borderRadius: 4 }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Explanation */}
          <div className="drawer-section">
            <div className="drawer-section-title">AI Explanation</div>
            <div className="explanation-box" style={{ borderLeft: `3px solid ${riskColor}` }}>
              {article.explanation || 'No explanation available.'}
            </div>
          </div>

          {/* Persuasion Techniques */}
          {allTechniques.length > 0 && (
            <div className="drawer-section">
              <div className="drawer-section-title">
                Detected Persuasion Techniques ({allTechniques.filter(t => t.score > 0.3).length} above threshold)
              </div>
              <div className="technique-list">
                {allTechniques.map(t => (
                  <div key={t.technique} className="technique-row">
                    <div className="technique-name" title={t.technique}>
                      {FULL_LABELS[t.technique] ?? t.technique.replace(/_/g, ' ')}
                    </div>
                    <div className="technique-bar-track">
                      <div className="technique-bar-fill" style={{ width: `${t.score * 100}%`, opacity: t.score > 0.3 ? 1 : 0.3 }} />
                    </div>
                    <div className="technique-score" style={{ color: t.score > 0.3 ? riskColor : 'var(--text-muted)' }}>
                      {t.score.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* SDG16 Note */}
          <div className="drawer-section">
            <div className="drawer-section-title">SDG 16 Alignment</div>
            <div className="explanation-box" style={{ fontSize: 12, lineHeight: 1.7, borderLeft: '3px solid var(--accent-blue)' }}>
              <strong>SDG 16.10</strong> — Article evaluated for informational quality.<br />
              <strong>SDG 16.6</strong> — Recommendation includes traceable explanation.<br />
              <strong>SDG 16.b</strong> — Provider diversity enforced at feed level.<br />
              <br />
              <em style={{ color: 'var(--text-muted)' }}>
                Scores computed by RoBERTa fine-tuned on SemEval-2023 Task 3. Always verify important claims independently.
              </em>
            </div>
          </div>
        </div>
      </aside>
    </>
  )
}
