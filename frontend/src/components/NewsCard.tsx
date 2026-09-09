import { Article } from '../api'

interface NewsCardProps {
  article: Article
  rank:    number
  onClick: (article: Article) => void
}

const TECHNIQUES_LABELS: Record<string, string> = {
  Appeal_to_Fear_Prejudice: 'Fear Appeal',
  'Appeal_to_Fear-Prejudice': 'Fear Appeal',
  Loaded_Language: 'Loaded Lang.',
  Bandwagon: 'Bandwagon',
  'Black-and-White_Fallacy': 'False Dilemma',
  Name_Calling_Labeling: 'Name Calling',
  'Name_Calling-Labeling': 'Name Calling',
  'Thought-terminating_Clichés': 'Cliché',
  Doubt: 'Doubt',
  Reasoning: 'Reasoning',
  'Exaggeration-Minimisation': 'Exaggeration',
}

export default function NewsCard({ article, rank, onClick }: NewsCardProps) {
  const riskClass = article.risk_label.toLowerCase()

  const scoreBarFill = (val: number) => `${Math.round(val * 100)}%`

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
            <span>{article.risk_emoji}</span>
            <span
              className={`risk-badge ${riskClass}`}
              aria-label={`Risk: ${article.risk_label}`}
            >
              {article.risk_label}
            </span>
            {article.url && (
              <span style={{ marginLeft: 'auto', fontSize: 11 }}>
                {new URL(article.url).hostname.replace('www.', '')}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Score bars */}
      <div className="score-bars">
        <div className="score-bar-item">
          <div className="score-bar-label">Relevance</div>
          <div className="score-bar-track">
            <div
              className="score-bar-fill relevance"
              style={{ width: scoreBarFill(article.relevance) }}
            />
          </div>
        </div>
        <div className="score-bar-item">
          <div className="score-bar-label">Risk</div>
          <div className="score-bar-track">
            <div
              className="score-bar-fill risk"
              style={{ width: scoreBarFill(article.persuasion_risk) }}
            />
          </div>
        </div>
        <div className="score-bar-item">
          <div className="score-bar-label">Score</div>
          <div className="score-bar-track">
            <div
              className="score-bar-fill responsible"
              style={{ width: scoreBarFill(article.responsible_score) }}
            />
          </div>
        </div>
      </div>

      {/* Technique chips */}
      {article.top_techniques.length > 0 && (
        <div className="technique-chips">
          {article.top_techniques
            .filter(t => t.score > 0.3)
            .slice(0, 4)
            .map(t => (
              <span key={t.technique} className="technique-chip">
                {TECHNIQUES_LABELS[t.technique] ?? t.technique.replace(/_/g, ' ')}
              </span>
            ))}
        </div>
      )}
    </div>
  )
}
