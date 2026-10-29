import { SDGMetrics } from '../api'

interface SDGMetricsStripProps {
  metrics: SDGMetrics
  loading?: boolean
}

interface MetricCard {
  label:     string
  value:     string
  sub:       string
  target:    string
  targetMet: boolean
  color:     string
}

export default function SDGMetricsStrip({ metrics, loading = false }: SDGMetricsStripProps) {
  const cards: MetricCard[] = [
    {
      label:     'PER',
      value:     loading ? '—' : `${(metrics.per * 100).toFixed(1)}%`,
      sub:       'Persuasion Exposure Reduction',
      target:    '≥ 15%',
      targetMet: metrics.per >= 0.15,
      color:     'sdg-blue',
    },
    {
      label:     'Parity Gap',
      value:     loading ? '—' : `${(metrics.provider_parity_gap * 100).toFixed(1)}%`,
      sub:       'Provider share imbalance',
      target:    '< 5%',
      targetMet: metrics.provider_parity_gap < 0.05,
      color:     'sdg-cyan',
    },
    {
      label:     'Explainability',
      value:     loading ? '—' : `${(metrics.explainability_cover * 100).toFixed(0)}%`,
      sub:       'Articles with explanations',
      target:    '≥ 95%',
      targetMet: metrics.explainability_cover >= 0.95,
      color:     'sdg-green',
    },
    {
      label:     'High Risk',
      value:     loading ? '—' : `${metrics.high_risk_articles}`,
      sub:       `of ${metrics.total_articles} articles`,
      target:    '< 20%',
      targetMet: metrics.total_articles > 0 && (metrics.high_risk_articles / metrics.total_articles) < 0.20,
      color:     'sdg-amber',
    },
    {
      label:     'Avg Risk',
      value:     loading ? '—' : metrics.avg_persuasion_risk.toFixed(3),
      sub:       'Mean persuasion score',
      target:    '< 0.40',
      targetMet: metrics.avg_persuasion_risk < 0.40,
      color:     'sdg-violet',
    },
  ]

  return (
    <div className="sdg-strip" role="region" aria-label="SDG 16 Metrics">
      {cards.map(card => (
        <div key={card.label} className={`sdg-metric ${card.color}`}>
          <div className="sdg-metric-label">{card.label}</div>
          <div className="sdg-metric-value">{card.value}</div>
          <div className="sdg-metric-sub">{card.sub}</div>
          <div className={`sdg-target-badge ${card.targetMet ? 'sdg-target-met' : 'sdg-target-missed'}`}>
            {card.targetMet ? '✓' : '✗'} Target: {card.target}
          </div>
        </div>
      ))}
    </div>
  )
}
