import { useState } from 'react'
import { Article } from './api'
import FeedPage from './components/FeedPage'
import ProfilePage from './components/ProfilePage'
import SDGDashboard from './components/SDGDashboard'
import LiveScorer from './components/LiveScorer'
import ArticleDrawer from './components/ArticleDrawer'
import AlphaSlider from './components/AlphaSlider'

type Tab = 'feed' | 'profile' | 'sdg' | 'scorer'

export default function App() {
  const [currentTab, setCurrentTab] = useState<Tab>('feed')
  const [userId, setUserId] = useState<string>('user_demo')
  const [alpha, setAlpha] = useState<number>(0.7)
  const [beta, setBeta] = useState<number>(0.3)
  const [selectedArticle, setSelectedArticle] = useState<Article | null>(null)

  const tabTitles: Record<Tab, string> = {
    feed: '📰 Responsible News Recommendations',
    profile: '🎯 Persuasion Susceptibility Profile',
    sdg: '🏛️ SDG 16 Governance & Accountability Audit',
    scorer: '⚡ Live Persuasion & Manipulation Detector',
  }

  return (
    <div className="app-layout">
      {/* Left Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <h1>ResponsiNews</h1>
          <span>SDG 16 Responsible AI</span>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-section-label">RECOMMENDATION ENGINE</div>
          <button
            className={`nav-item ${currentTab === 'feed' ? 'active' : ''}`}
            onClick={() => setCurrentTab('feed')}
            style={{ width: '100%', textAlign: 'left', background: currentTab === 'feed' ? 'var(--accent-glow)' : 'transparent' }}
          >
            <span>📰</span>
            <span>Live Feed & Reranker</span>
          </button>

          <button
            className={`nav-item ${currentTab === 'profile' ? 'active' : ''}`}
            onClick={() => setCurrentTab('profile')}
            style={{ width: '100%', textAlign: 'left', background: currentTab === 'profile' ? 'var(--accent-glow)' : 'transparent' }}
          >
            <span>🎯</span>
            <span>Susceptibility Profile</span>
          </button>

          <div className="nav-section-label" style={{ marginTop: 18 }}>
            GOVERNANCE & AUDIT
          </div>
          <button
            className={`nav-item ${currentTab === 'sdg' ? 'active' : ''}`}
            onClick={() => setCurrentTab('sdg')}
            style={{ width: '100%', textAlign: 'left', background: currentTab === 'sdg' ? 'var(--accent-glow)' : 'transparent' }}
          >
            <span>🏛️</span>
            <span>SDG 16 Dashboard</span>
          </button>

          <button
            className={`nav-item ${currentTab === 'scorer' ? 'active' : ''}`}
            onClick={() => setCurrentTab('scorer')}
            style={{ width: '100%', textAlign: 'left', background: currentTab === 'scorer' ? 'var(--accent-glow)' : 'transparent' }}
          >
            <span>⚡</span>
            <span>Live Text Detector</span>
          </button>
        </nav>

        {/* Sidebar Footer */}
        <div
          style={{
            padding: '16px 20px',
            borderTop: '1px solid var(--border)',
            background: 'var(--bg-card)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: 'var(--risk-low)',
                boxShadow: '0 0 8px var(--risk-low)',
              }}
            />
            <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)' }}>
              PIPELINE ONLINE
            </span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.4 }}>
            MIND GNN + RoBERTa-23
            <br />
            Provider Diversity Cap: 20%
          </div>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {/* Top Header */}
        <header className="page-header">
          <div className="header-title">{tabTitles[currentTab]}</div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            {/* Live Tuner visible on Feed */}
            {currentTab === 'feed' && (
              <AlphaSlider alpha={alpha} beta={beta} onAlpha={setAlpha} onBeta={setBeta} />
            )}

            <div
              className="user-badge"
              onClick={() => setCurrentTab('profile')}
              title="Click to view persuasion susceptibility vector"
            >
              <span style={{ fontSize: 14 }}>👤</span>
              <span>{userId === 'user_demo' ? 'Demo User' : userId}</span>
              <span
                style={{
                  fontSize: 10,
                  background: 'var(--bg-primary)',
                  padding: '2px 6px',
                  borderRadius: 'var(--radius-sm)',
                  color: 'var(--accent-cyan)',
                }}
              >
                PROFILED
              </span>
            </div>
          </div>
        </header>

        {/* Page Views */}
        {currentTab === 'feed' && (
          <FeedPage
            userId={userId}
            alpha={alpha}
            beta={beta}
            onSelectArticle={setSelectedArticle}
          />
        )}
        {currentTab === 'profile' && (
          <ProfilePage userId={userId} onUserChange={setUserId} />
        )}
        {currentTab === 'sdg' && <SDGDashboard />}
        {currentTab === 'scorer' && <LiveScorer />}
      </main>

      {/* Article Detail Drawer */}
      <ArticleDrawer
        article={selectedArticle}
        onClose={() => setSelectedArticle(null)}
      />
    </div>
  )
}
