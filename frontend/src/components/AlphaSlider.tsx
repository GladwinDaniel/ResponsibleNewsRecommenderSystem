interface AlphaSliderProps {
  alpha:    number
  beta:     number
  onAlpha:  (v: number) => void
  onBeta:   (v: number) => void
}

export default function AlphaSlider({ alpha, beta, onAlpha, onBeta }: AlphaSliderProps) {
  return (
    <div className="tuning-bar" role="group" aria-label="Ranking weight controls">
      {/* Alpha: relevance weight */}
      <div className="tuning-slider">
        <span className="tuning-label">α Relevance</span>
        <input
          id="alpha-slider"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={alpha}
          onChange={e => onAlpha(parseFloat(e.target.value))}
          aria-label="Alpha: relevance weight"
        />
        <span className="tuning-value">{alpha.toFixed(2)}</span>
      </div>

      {/* Beta: risk penalty weight */}
      <div className="tuning-slider">
        <span className="tuning-label" style={{ color: 'var(--risk-medium)' }}>β Risk</span>
        <input
          id="beta-slider"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={beta}
          onChange={e => onBeta(parseFloat(e.target.value))}
          aria-label="Beta: risk penalty weight"
          style={{ '--thumb-color': 'var(--risk-medium)' } as React.CSSProperties}
        />
        <span className="tuning-value" style={{ color: 'var(--risk-medium)' }}>
          {beta.toFixed(2)}
        </span>
      </div>

      {/* Visual formula */}
      <span style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
        Score = <span style={{ color: 'var(--accent-blue)' }}>{alpha.toFixed(2)}·Rel</span>
        {' − '}
        <span style={{ color: 'var(--risk-medium)' }}>{beta.toFixed(2)}·Risk</span>
      </span>
    </div>
  )
}
