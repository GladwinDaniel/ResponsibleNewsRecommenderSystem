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

  // Fetch or re-compute recommendations
  useEffect(() => {
    let isMounted = true
    setLoading(true)

    getRecommendations(userId, 10, alpha, beta)
      .then((data) => {
        if (isMounted && data && data.length > 0) {
          setArticles(data)
        }
      })
      .catch(() => {
        // Fallback to client-side recalculated demo articles
        if (isMounted) {
          const recalculated = DEMO_ARTICLES.map((art) => {
            const responsible_score = Math.max(0, Math.min(1, alpha * art.relevance - beta * art.persuasion_risk))
            return { ...art, responsible_score }
          }).sort((a, b) => b.responsible_score - a.responsible_score)
          setArticles(recalculated)
        }
      })
      .finally(() => {
        if (isMounted) setLoading(false)
      })

    getSDGMetrics()
      .then((metrics) => {
        if (isMounted && metrics) setSdgMetrics(metrics)
      })
      .catch(() => {
        if (isMounted) setSdgMetrics(DEMO_SDG)
      })

    return () => {
      isMounted = false
    }
  }, [userId, alpha, beta])

  const categories = ['all', ...Array.from(new Set(articles.map((a) => a.category)))]

  const filteredArticles = articles.filter((art) => {
    const matchCat = selectedCategory === 'all' || art.category.toLowerCase() === selectedCategory.toLowerCase()
    const matchRisk = riskFilter === 'all' || art.risk_label === riskFilter
    return matchCat && matchRisk
  })

  return (
    <div className="page-body">
      {/* SDG strip overview */}
      <SDGMetricsStrip metrics={sdgMetrics} loading={loading} />

      {/* Filter and Status Controls */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 12,
          marginBottom: 20,
          background: 'var(--bg-secondary)',
          padding: '12px 18px',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border)',
        }}
      >
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600 }}>CATEGORY:</span>
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              style={{
                background: selectedCategory === cat ? 'var(--accent-glow)' : 'var(--bg-tertiary)',
                color: selectedCategory === cat ? 'var(--accent-blue)' : 'var(--text-secondary)',
                border: `1px solid ${selectedCategory === cat ? 'var(--accent-blue)' : 'var(--border)'}`,
                padding: '4px 10px',
                borderRadius: 'var(--radius-sm)',
                fontSize: 12,
                textTransform: 'capitalize',
                transition: 'all var(--transition)',
              }}
            >
              {cat}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600 }}>RISK FILTER:</span>
          {(['all', 'LOW', 'MEDIUM', 'HIGH'] as const).map((r) => (
            <button
              key={r}
              onClick={() => setRiskFilter(r)}
              style={{
                background: riskFilter === r ? 'var(--bg-tertiary)' : 'transparent',
                color:
                  riskFilter === r
                    ? r === 'LOW'
                      ? 'var(--risk-low)'
                      : r === 'MEDIUM'
                      ? 'var(--risk-medium)'
                      : r === 'HIGH'
                      ? 'var(--risk-high)'
                      : 'var(--text-primary)'
                    : 'var(--text-muted)',
                border: `1px solid ${riskFilter === r ? 'var(--border-focus)' : 'transparent'}`,
                padding: '4px 10px',
                borderRadius: 'var(--radius-sm)',
                fontSize: 12,
                fontWeight: 600,
                transition: 'all var(--transition)',
              }}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {/* Feed list */}
      <div className="news-feed">
        {filteredArticles.length === 0 ? (
          <div className="empty-state">No articles match the selected filters.</div>
        ) : (
          filteredArticles.map((article, idx) => (
            <NewsCard
              key={article.news_id}
              article={article}
              rank={idx + 1}
              onClick={onSelectArticle}
            />
          ))
        )}
      </div>
    </div>
  )
}
