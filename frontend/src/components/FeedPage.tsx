import { useState, useEffect } from 'react'
import { Article, SDGMetrics, getRecommendations, getSDGMetrics, DEMO_ARTICLES, DEMO_SDG } from '../api'
import NewsCard from './NewsCard'
import SDGMetricsStrip from './SDGMetricsStrip'

interface FeedPageProps {
  userId: string
  alpha: number
  beta: number
  onSelectArticle: (article: Article) => void
}

export default function FeedPage({ userId, alpha, beta, onSelectArticle }: FeedPageProps) {
  const [articles, setArticles] = useState<Article[]>(DEMO_ARTICLES)
  const [sdgMetrics, setSdgMetrics] = useState<SDGMetrics>(DEMO_SDG)
  const [loading, setLoading] = useState(false)
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [riskFilter, setRiskFilter] = useState<'all' | 'LOW' | 'MEDIUM' | 'HIGH'>('all')

  useEffect(() => {
    let isMounted = true
    setLoading(true)

    getRecommendations(userId, 10, alpha, beta)
      .then((data) => {
        if (isMounted && data && data.length > 0) setArticles(data)
      })
      .catch(() => {
        if (isMounted) {
          const recalculated = DEMO_ARTICLES.map((art) => {
            const responsible_score = Math.max(0, Math.min(1, alpha * art.relevance - beta * art.persuasion_risk))
            return { ...art, responsible_score }
          }).sort((a, b) => b.responsible_score - a.responsible_score)
          setArticles(recalculated)
        }
      })
      .finally(() => { if (isMounted) setLoading(false) })

    getSDGMetrics()
      .then((metrics) => { if (isMounted && metrics) setSdgMetrics(metrics) })
      .catch(() => { if (isMounted) setSdgMetrics(DEMO_SDG) })

    return () => { isMounted = false }
  }, [userId, alpha, beta])

  const categories = ['all', ...Array.from(new Set(articles.map((a) => a.category)))]

  const filteredArticles = articles.filter((art) => {
    const matchCat = selectedCategory === 'all' || art.category.toLowerCase() === selectedCategory.toLowerCase()
    const matchRisk = riskFilter === 'all' || art.risk_label === riskFilter
    return matchCat && matchRisk
  })

  const heroArticle = filteredArticles[0]
  const gridArticles = filteredArticles.slice(1)

  const riskFilterClass = (r: string) => {
    if (riskFilter !== r) return 'filter-btn'
    if (r === 'LOW') return 'filter-btn active-low'
    if (r === 'MEDIUM') return 'filter-btn active-medium'
    if (r === 'HIGH') return 'filter-btn active-high'
    return 'filter-btn active'
  }

  return (
    <div className="page-body">
      {/* SDG Metrics Overview */}
      <SDGMetricsStrip metrics={sdgMetrics} loading={loading} />

      {/* Filter Bar */}
      <div className="filter-bar">
        <div className="filter-group">
          <span className="filter-label">Category</span>
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`filter-btn ${selectedCategory === cat ? 'active' : ''}`}
            >
              {cat}
            </button>
          ))}
        </div>

        <div className="filter-group">
          <span className="filter-label">Risk</span>
          {(['all', 'LOW', 'MEDIUM', 'HIGH'] as const).map((r) => (
            <button
              key={r}
              onClick={() => setRiskFilter(r)}
              className={riskFilterClass(r)}
            >
              {r === 'all' ? 'All' : r === 'LOW' ? '🟢 Low' : r === 'MEDIUM' ? '🟡 Medium' : '🔴 High'}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="loading-container">
          <div className="spinner" />
          <span>Loading recommendations...</span>
        </div>
      )}

      {!loading && filteredArticles.length === 0 && (
        <div className="empty-state">No articles match the selected filters.</div>
      )}

      {/* ── Hero Article (Top Ranked) ─────────────────────────── */}
      {!loading && heroArticle && (
        <div className="hero-card" onClick={() => onSelectArticle(heroArticle)}>
          <div className="hero-inner">
            <div className="hero-content">
              <div className="hero-rank">
                #1 Top Recommended · {heroArticle.risk_emoji} {heroArticle.risk_label} Risk
              </div>
              <h2 className="hero-title">{heroArticle.title}</h2>
              {heroArticle.abstract && (
                <p className="hero-abstract">{heroArticle.abstract}</p>
              )}
              <div className="hero-meta">
                <span className="news-card-category">{heroArticle.category}</span>
                {heroArticle.url && (
                  <span className="news-card-source">
                    {new URL(heroArticle.url).hostname.replace('www.', '')}
                  </span>
                )}
                <span className={`risk-badge ${heroArticle.risk_label.toLowerCase()}`}>
                  {heroArticle.risk_label}
                </span>
                {heroArticle.top_techniques.filter(t => t.score > 0.3).slice(0, 3).map(t => (
                  <span key={t.technique} className="technique-chip">
                    {t.technique.replace(/[-_]/g, ' ')}
                  </span>
                ))}
              </div>
            </div>

            <div className="hero-scores">
              <div style={{ fontFamily: 'var(--font-display)', fontSize: 12, fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase' as const, marginBottom: 4 }}>
                Score Breakdown
              </div>
              {[
                { label: 'Responsible Score', val: heroArticle.responsible_score, color: 'var(--accent-cyan)' },
                { label: 'Relevance (GNN)', val: heroArticle.relevance, color: 'var(--accent-blue)' },
                { label: 'Persuasion Risk', val: heroArticle.persuasion_risk, color: heroArticle.risk_label === 'LOW' ? 'var(--risk-low)' : heroArticle.risk_label === 'MEDIUM' ? 'var(--risk-medium)' : 'var(--risk-high)' },
              ].map(({ label, val, color }) => (
                <div key={label} className="hero-score-item">
                  <div className="hero-score-label">
                    <span className="hero-score-name">{label}</span>
                    <span className="hero-score-val" style={{ color }}>{(val * 100).toFixed(0)}</span>
                  </div>
                  <div className="hero-bar-track">
                    <div className="hero-bar-fill" style={{ width: `${val * 100}%`, background: color }} />
                  </div>
                </div>
              ))}
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4, lineHeight: 1.4, borderTop: '1px solid var(--border)', paddingTop: 10 }}>
                Formula: <span style={{ fontFamily: 'var(--font-display)', color: 'var(--accent-blue)' }}>α·Rel</span>
                {' − '}
                <span style={{ fontFamily: 'var(--font-display)', color: 'var(--risk-high)' }}>β·Risk</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Magazine Grid (Remaining Articles) ────────────────── */}
      {!loading && gridArticles.length > 0 && (
        <div className="magazine-grid">
          {gridArticles.map((article, idx) => (
            <NewsCard
              key={article.news_id}
              article={article}
              rank={idx + 2}
              onClick={onSelectArticle}
            />
          ))}
        </div>
      )}
    </div>
  )
}
