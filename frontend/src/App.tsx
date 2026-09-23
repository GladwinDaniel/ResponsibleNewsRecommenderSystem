import { useState } from 'react'
import { Article } from './api'
import FeedPage from './components/FeedPage'
import ProfilePage from './components/ProfilePage'
import SDGDashboard from './components/SDGDashboard'
import LiveScorer from './components/LiveScorer'
import ArticleDrawer from './components/ArticleDrawer'
import AlphaSlider from './components/AlphaSlider'

type Tab = 'feed' | 'profile' | 'sdg' | 'scorer'

const NAV_ITEMS: { tab: Tab; icon: string; label: string; section?: string }[] = [
  { tab: 'feed',    icon: '📰', label: 'News Feed',           section: 'RECOMMENDATION' },
  { tab: 'profile', icon: '👤', label: 'User Profile' },
  { tab: 'sdg',     icon: '📊', label: 'SDG 16 Analytics',    section: 'GOVERNANCE' },
  { tab: 'scorer',  icon: '🔬', label: 'Live Detector' },
]

const TAB_TITLES: Record<Tab, string> = {
  feed:    'Responsible News Feed',
  profile: 'Persuasion Susceptibility Profile',
  sdg:     'SDG 16 Governance Dashboard',
  scorer:  'Live Persuasion Detector',
}

export default function App() {
  const [currentTab, setCurrentTab] = useState<Tab>('feed')
  const [userId, setUserId] = useState<string>('user_demo')
  const [alpha, setAlpha] = useState<number>(0.7)
  const [beta, setBeta] = useState<number>(0.3)
  const [selectedArticle, setSelectedArticle] = useState<Article | null>(null)

  return (
    <div className="app-layout">
      {/* ── Sidebar ──────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <h1>ResponsiNews</h1>
          <span>SDG 16 · Responsible AI</span>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => (
            <div key={item.tab}>
              {item.section && (
                <div className="nav-section-label">{item.section}</div>
              )}
              <button
                className={`nav-item ${currentTab === item.tab ? 'active' : ''}`}
                onClick={() => setCurrentTab(item.tab)}
                style={{ width: '100%', textAlign: 'left', background: currentTab === item.tab ? 'var(--accent-glow)' : 'transparent' }}
              >
                <span>{item.icon}</span>
                <span>{item.label}</span>
              </button>
            </div>
          ))}
        </nav>

        {/* Sidebar Footer */}
        <div style={{
          padding: '14px 16px',
          borderTop: '1px solid var(--border)',
          background: 'var(--bg-tertiary)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%',
              background: 'var(--risk-low)',
              boxShadow: '0 0 6px var(--risk-low)',
            }} />
            <span style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-muted)', letterSpacing: '0.05em' }}>
              SYSTEM ONLINE
            </span>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
            GNN + RoBERTa-23 Pipeline
            <br />
            Provider Diversity: 30% cap
          </div>
        </div>
      </aside>

      {/* ── Main ─────────────────────────────────────────────────── */}
      <main className="main-content">
        <header className="page-header">
          <div className="header-title">{TAB_TITLES[currentTab]}</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            {currentTab === 'feed' && (
              <AlphaSlider alpha={alpha} beta={beta} onAlpha={setAlpha} onBeta={setBeta} />
            )}
            <div
              className="user-badge"
              onClick={() => setCurrentTab('profile')}
              title="View persuasion susceptibility profile"
            >
              <span style={{ fontSize: 13 }}>👤</span>
              <span style={{ fontSize: 13, fontWeight: 500 }}>
                {userId === 'user_demo' ? 'Demo User' : userId}
              </span>
            </div>
          </div>
        </header>

        {currentTab === 'feed' && (
          <FeedPage userId={userId} alpha={alpha} beta={beta} onSelectArticle={setSelectedArticle} />
        )}
        {currentTab === 'profile' && (
          <ProfilePage userId={userId} onUserChange={setUserId} />
        )}
        {currentTab === 'sdg' && <SDGDashboard />}
        {currentTab === 'scorer' && <LiveScorer />}
      </main>

      {/* ── Article Detail Drawer ────────────────────────────────── */}
      <ArticleDrawer article={selectedArticle} onClose={() => setSelectedArticle(null)} />
    </div>
  )
}
