import { useState, useEffect } from 'react'
import { UserProfile, getUserProfile, DEMO_PROFILE } from '../api'
import PersuasionRadar from './PersuasionRadar'

interface ProfilePageProps {
  userId: string
  onUserChange: (userId: string) => void
}

const PRESET_USERS = [
  { id: 'user_demo', label: 'Balanced User (Demo)', note: 'Moderate fear & doubt susceptibility' },
  { id: 'user_sensitive', label: 'Vulnerable Profile', note: 'High emotional & loaded language sensitivity' },
  { id: 'user_skeptic', label: 'Analytical Reader', note: 'High resistance, facts-focused' },
]

export default function ProfilePage({ userId, onUserChange }: ProfilePageProps) {
  const [profile, setProfile] = useState<UserProfile>(DEMO_PROFILE)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let isMounted = true
    setLoading(true)

    getUserProfile(userId)
      .then((data) => {
        if (isMounted && data) setProfile(data)
      })
      .catch(() => {
        if (isMounted) {
          if (userId === 'user_sensitive') {
            setProfile({
              user_id: 'user_sensitive',
              susceptibility: 0.76,
              n_updates: 142,
              vector: [
                0.3, 0.88, 0.65, 0.5, 0.4, 0.82, 0.3, 0.2, 0.2, 0.94, 0.81, 0.1, 0.4, 0.2, 0.3, 0.2, 0.5, 0.2, 0.4, 0.6, 0.3, 0.2, 0.1,
              ],
              top_techniques: [
                { technique: 'Loaded_Language', score: 0.94 },
                { technique: 'Appeal_to_Fear-Prejudice', score: 0.88 },
                { technique: 'Doubt', score: 0.82 },
                { technique: 'Name_Calling-Labeling', score: 0.81 },
              ],
            })
          } else if (userId === 'user_skeptic') {
            setProfile({
              user_id: 'user_skeptic',
              susceptibility: 0.15,
              n_updates: 210,
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
      .finally(() => {
        if (isMounted) setLoading(false)
      })

    return () => {
      isMounted = false
    }
  }, [userId])

  const susceptibilityPercent = Math.round(profile.susceptibility * 100)
  const susceptibilityColor =
    profile.susceptibility < 0.3 ? 'var(--risk-low)' : profile.susceptibility < 0.6 ? 'var(--risk-medium)' : 'var(--risk-high)'

  return (
    <div className="page-body">
      {/* Header bar with user profile switcher */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 16,
          background: 'var(--bg-card)',
          padding: '16px 24px',
          borderRadius: 'var(--radius-lg)',
          border: '1px solid var(--border)',
          marginBottom: 24,
        }}
      >
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 700, fontFamily: 'var(--font-display)', marginBottom: 4 }}>
            User Susceptibility Profile
          </h2>
          <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Dynamic Exponential Moving Average (EMA) tracking across 23 cognitive & propaganda manipulation techniques.
          </p>
        </div>

        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600 }}>SWITCH PERSONA:</span>
          {PRESET_USERS.map((u) => (
            <button
              key={u.id}
              onClick={() => onUserChange(u.id)}
              style={{
                background: userId === u.id ? 'var(--accent-glow)' : 'var(--bg-tertiary)',
                color: userId === u.id ? 'var(--accent-blue)' : 'var(--text-secondary)',
                border: `1px solid ${userId === u.id ? 'var(--accent-blue)' : 'var(--border)'}`,
                padding: '6px 12px',
                borderRadius: 'var(--radius-sm)',
                fontSize: 12,
                fontWeight: 600,
                transition: 'all var(--transition)',
              }}
              title={u.note}
            >
              {u.label}
            </button>
          ))}
        </div>
      </div>

      <div className="profile-grid">
        {/* Left Column: Susceptibility Index & Top Vulnerabilities */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div className="card">
            <div className="card-title">Overall Susceptibility Index</div>
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                padding: '24px 12px',
                textAlign: 'center',
              }}
            >
              <div
                style={{
                  width: 140,
                  height: 140,
                  borderRadius: '50%',
                  border: `8px solid ${susceptibilityColor}`,
                  boxShadow: `0 0 25px ${susceptibilityColor}33`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexDirection: 'column',
                  marginBottom: 16,
                }}
              >
                <span
                  style={{
                    fontSize: 38,
                    fontWeight: 800,
                    fontFamily: 'var(--font-display)',
                    color: susceptibilityColor,
                    lineHeight: 1,
                  }}
                >
                  {susceptibilityPercent}%
                </span>
                <span style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', marginTop: 4 }}>
                  Susceptibility
                </span>
              </div>

              <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
                {profile.susceptibility < 0.3
                  ? 'Resilient Reader'
                  : profile.susceptibility < 0.6
                  ? 'Moderate Susceptibility'
                  : 'High Persuasion Vulnerability'}
              </div>
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4, maxWidth: 280 }}>
                {profile.susceptibility < 0.3
                  ? 'User demonstrates strong critical resistance against loaded rhetoric and fear appeals.'
                  : profile.susceptibility < 0.6
                  ? 'User interacts moderately with emotionally charged headlines. Penalties applied to protect neutrality.'
                  : 'Recommender actively deprioritizes manipulative framing to protect user autonomy.'}
              </p>

              <div
                style={{
                  display: 'flex',
                  gap: 16,
                  marginTop: 20,
                  paddingTop: 16,
                  borderTop: '1px solid var(--border)',
                  width: '100%',
                  justifyContent: 'space-around',
                }}
              >
                <div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>MIND Interactions</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>{profile.n_updates}</div>
                </div>
                <div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>EMA Alpha Rate</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--accent-cyan)' }}>0.15</div>
                </div>
                <div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Protection State</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--risk-low)' }}>Active</div>
                </div>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-title">Top Detected Vulnerabilities</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {profile.top_techniques.map((item, idx) => (
                <div
                  key={item.technique}
                  style={{
                    background: 'var(--bg-tertiary)',
                    padding: '10px 14px',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border)',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                      #{idx + 1} {item.technique.replace(/[-_]/g, ' ')}
                    </span>
                    <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--risk-high)', fontFamily: 'monospace' }}>
                      {(item.score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="technique-bar-track">
                    <div
                      className="technique-bar-fill"
                      style={{ width: `${Math.round(item.score * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: 23-Dimension Persuasion Radar */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>23-Technique Persuasion Vector</span>
            <span style={{ fontSize: 11, color: 'var(--accent-cyan)', fontWeight: 500 }}>
              SemEval 2020 Task 11 Hierarchy
            </span>
          </div>
          <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16 }}>
            The radar visualizes the user's susceptibility across all 23 propaganda categories. When an article contains high
            concentrations of a technique where the user shows susceptibility, the Responsible Recommender penalizes the article's rank.
          </p>

          <div style={{ flex: 1, minHeight: 380, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <PersuasionRadar profile={profile} />
          </div>

          <div
            style={{
              marginTop: 16,
              padding: '12px 16px',
              background: 'var(--bg-tertiary)',
              borderRadius: 'var(--radius-md)',
              fontSize: 12,
              color: 'var(--text-secondary)',
              borderLeft: '3px solid var(--accent-violet)',
            }}
          >
            <strong>SDG 16 Alignment Note:</strong> Transparent susceptibility tracking empowers readers by shedding light on subtle editorial steering, combating echo chambers and deceptive cognitive exploitation.
          </div>
        </div>
      </div>
    </div>
  )
}
