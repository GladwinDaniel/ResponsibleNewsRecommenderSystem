import { Article } from '../api'

interface NewsCardProps {
  article: Article
  rank:    number
  onClick: (article: Article) => void
}

const TECHNIQUES_LABELS: Record<string, string> = {
  'Appeal_to_Fear-Prejudice': 'Fear Appeal',
  Loaded_Language: 'Loaded Language',
  Bandwagon: 'Bandwagon',
  'Black-and-White_Fallacy': 'False Dilemma',
  'Name_Calling-Labeling': 'Name Calling',
  'Thought-terminating_Clichés': 'Cliché',
  Doubt: 'Doubt',
  Reasoning: 'Reasoning',
  'Exaggeration-Minimisation': 'Exaggeration',
  Appeal_to_Authority: 'Authority',
  Causal_Oversimplification: 'Causal Oversimp.',
  'Flag-Waving': 'Flag-Waving',
}

export default function NewsCard({ article, rank, onClick }: NewsCardProps) {
  const riskClass = article.risk_label.toLowerCase()

  return (
    <div
      className={`news-card risk-${riskClass}`}
      onClick={() => onClick(article)}
      role="button"
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && onClick(article)}
      aria-label={`Article ${rank}: ${article.title}`}
    >
      <div className="news-card-header">
        <span className="news-card-rank">#{rank}</span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="news-card-title">{article.title}</div>
          <div className="news-card-meta" style={{ marginTop: 6 }}>
            <span className="news-card-category">{article.category}</span>
            {article.url && (
              <span className="news-card-source">
                {new URL(article.url).hostname.replace('www.', '')}
              </span>
            )}
            <span className={`risk-badge ${riskClass}`}>
              {article.risk_emoji} {article.risk_label}
            </span>
          </div>
        </div>
      </div>

      {/* Score bars */}
      <div className="score-bars">
        {[
          { label: 'Relevance', val: article.relevance, cls: 'relevance', color: 'var(--accent-blue)' },
          { label: 'Risk', val: article.persuasion_risk, cls: 'risk', color: 'var(--risk-high)' },
          { label: 'Score', val: article.responsible_score, cls: 'responsible', color: 'var(--accent-cyan)' },
        ].map(({ label, val, cls, color }) => (
          <div key={label} className="score-bar-item">
            <div className="score-bar-label">
              <span>{label}</span>
              <span className="score-bar-value" style={{ color }}>{(val * 100).toFixed(0)}</span>
            </div>
            <div className="score-bar-track">
              <div className={`score-bar-fill ${cls}`} style={{ width: `${val * 100}%` }} />
            </div>
          </div>
        ))}
      </div>

      {/* Technique chips */}
      {article.top_techniques.length > 0 && (
        <div className="technique-chips">
          {article.top_techniques
            .filter(t => t.score > 0.3)
            .slice(0, 3)
            .map(t => (
              <span key={t.technique} className="technique-chip">
                {TECHNIQUES_LABELS[t.technique] ?? t.technique.replace(/_/g, ' ')}
              </span>
            ))}
        </div>
      )}

      {/* Explanation preview */}
      {article.explanation && (
        <div className="news-card-explanation">{article.explanation}</div>
      )}
    </div>
  )
}
