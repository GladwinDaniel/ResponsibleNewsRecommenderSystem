import { useState, useEffect } from 'react'
import { UserProfile, getUserProfile, DEMO_PROFILE } from '../api'
import PersuasionRadar from './PersuasionRadar'

interface ProfilePageProps {
  userId: string
  onUserChange: (userId: string) => void
}

const PRESET_USERS = [
  { id: 'user_demo',      label: 'Balanced User',      note: 'Moderate susceptibility', icon: '👤' },
  { id: 'user_sensitive',  label: 'Vulnerable Profile',  note: 'High emotional sensitivity', icon: '⚠️' },
  { id: 'user_skeptic',    label: 'Analytical Reader',   note: 'High resistance', icon: '🛡️' },
]

export default function ProfilePage({ userId, onUserChange }: ProfilePageProps) {
  const [profile, setProfile] = useState<UserProfile>(DEMO_PROFILE)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let isMounted = true
    setLoading(true)

    getUserProfile(userId)
      .then((data) => { if (isMounted && data) setProfile(data) })
      .catch(() => {
        if (isMounted) {
          if (userId === 'user_sensitive') {
            setProfile({
              user_id: 'user_sensitive', susceptibility: 0.76, n_updates: 142,
              vector: [0.3, 0.88, 0.65, 0.5, 0.4, 0.82, 0.3, 0.2, 0.2, 0.94, 0.81, 0.1, 0.4, 0.2, 0.3, 0.2, 0.5, 0.2, 0.4, 0.6, 0.3, 0.2, 0.1],
              top_techniques: [
                { technique: 'Loaded_Language', score: 0.94 },
                { technique: 'Appeal_to_Fear-Prejudice', score: 0.88 },
                { technique: 'Doubt', score: 0.82 },
                { technique: 'Name_Calling-Labeling', score: 0.81 },
              ],
            })
          } else if (userId === 'user_skeptic') {
            setProfile({
              user_id: 'user_skeptic', susceptibility: 0.15, n_updates: 210,
              vector: new Array(23).fill(0.12),
              top_techniques: [
                { technique: 'Reasoning', score: 0.78 },
                { technique: 'Appeal_to_Authority', score: 0.25 },
              ],
            })
          } else {
            setProfile(DEMO_PROFILE)
          }
        }
      })
      .finally(() => { if (isMounted) setLoading(false) })

    return () => { isMounted = false }
  }, [userId])

  const susceptibilityPercent = Math.round(profile.susceptibility * 100)
  const susceptibilityColor =
    profile.susceptibility < 0.3 ? 'var(--risk-low)' : profile.susceptibility < 0.6 ? 'var(--risk-medium)' : 'var(--risk-high)'
  const susceptibilityLabel =
    profile.susceptibility < 0.3 ? 'Resilient Reader' : profile.susceptibility < 0.6 ? 'Moderate Susceptibility' : 'High Vulnerability'

  return (
    <div className="page-body">
      {/* Header with persona switcher */}
      <div className="card" style={{ marginBottom: 20, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16 }}>
        <div>
          <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 700, marginBottom: 4 }}>
            User Susceptibility Profile
          </h2>
          <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            EMA-tracked across 23 persuasion techniques · Updated in real-time from reading behavior
          </p>
        </div>
        <div className="filter-group">
          <span className="filter-label">Persona</span>
          {PRESET_USERS.map((u) => (
            <button
              key={u.id}
              onClick={() => onUserChange(u.id)}
              className={`filter-btn ${userId === u.id ? 'active' : ''}`}
              title={u.note}
            >
              {u.icon} {u.label}
            </button>
          ))}
        </div>
      </div>

      <div className="profile-grid">
        {/* Left: Susceptibility Index */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="card">
            <div className="card-title">Overall Susceptibility Index</div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px 12px', textAlign: 'center' }}>
              <div style={{
                width: 130, height: 130, borderRadius: '50%',
                border: `7px solid ${susceptibilityColor}`,
                boxShadow: `0 0 0 4px ${susceptibilityColor}15`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexDirection: 'column', marginBottom: 16,
              }}>
                <span style={{ fontSize: 36, fontWeight: 800, fontFamily: 'var(--font-display)', color: susceptibilityColor, lineHeight: 1 }}>
                  {susceptibilityPercent}%
                </span>
                <span style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', marginTop: 3, fontWeight: 600 }}>
                  Susceptibility
                </span>
              </div>

              <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>{susceptibilityLabel}</div>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4, maxWidth: 260 }}>
                {profile.susceptibility < 0.3
                  ? 'Strong critical resistance against loaded rhetoric.'
                  : profile.susceptibility < 0.6
                  ? 'Moderate interaction with emotionally charged content. Penalties applied.'
                  : 'Active deprioritization of manipulative framing to protect autonomy.'}
              </p>

              <div style={{ display: 'flex', gap: 20, marginTop: 20, paddingTop: 14, borderTop: '1px solid var(--border)', width: '100%', justifyContent: 'space-around' }}>
                <div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>Interactions</div>
                  <div style={{ fontSize: 18, fontWeight: 800, fontFamily: 'var(--font-display)', color: 'var(--text-primary)' }}>{profile.n_updates}</div>
                </div>
                <div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>EMA Rate</div>
                  <div style={{ fontSize: 18, fontWeight: 800, fontFamily: 'var(--font-display)', color: 'var(--accent-cyan)' }}>0.15</div>
                </div>
                <div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase' }}>Protection</div>
                  <div style={{ fontSize: 18, fontWeight: 800, fontFamily: 'var(--font-display)', color: 'var(--risk-low)' }}>Active</div>
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-title">Top Vulnerabilities</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {profile.top_techniques.map((item, idx) => (
                <div key={item.technique} style={{ background: 'var(--bg-tertiary)', padding: '10px 14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                      #{idx + 1} {item.technique.replace(/[-_]/g, ' ')}
                    </span>
                    <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--risk-high)', fontFamily: 'var(--font-display)' }}>
                      {(item.score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="technique-bar-track">
                    <div className="technique-bar-fill" style={{ width: `${Math.round(item.score * 100)}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: 23-Dimension Radar */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-title" style={{ justifyContent: 'space-between' }}>
            <span>23-Technique Persuasion Vector</span>
            <span style={{ fontSize: 11, color: 'var(--accent-cyan)', fontWeight: 500 }}>SemEval-2023 Task 3</span>
          </div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16, lineHeight: 1.5 }}>
            Radar chart shows user susceptibility across all 23 propaganda categories. High values trigger stronger risk penalties during reranking.
          </p>

          <div style={{ flex: 1, minHeight: 380, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <PersuasionRadar profile={profile} />
          </div>

          <div style={{
            marginTop: 16, padding: '12px 16px', background: 'var(--bg-tertiary)',
            borderRadius: 'var(--radius-md)', fontSize: 12, color: 'var(--text-secondary)',
            borderLeft: '3px solid var(--accent-violet)',
          }}>
            <strong>SDG 16:</strong> Transparent susceptibility tracking empowers readers against subtle editorial steering and echo chambers.
          </div>
        </div>
      </div>
    </div>
  )
}
