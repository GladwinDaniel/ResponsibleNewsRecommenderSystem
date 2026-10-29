// API client — talks to FastAPI backend via /api proxy (or direct on :8000)
import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || '/api'

export const api = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
})

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Technique {
  technique: string
  score:     number
  description?: string
}

export interface Article {
  news_id:           string
  title:             string
  abstract:          string
  category:          string
  url:               string
  persuasion_risk:   number
  risk_label:        'LOW' | 'MEDIUM' | 'HIGH'
  risk_emoji:        string
  risk_color:        string
  responsible_score: number
  relevance:         number
  explanation:       string
  top_techniques:    Technique[]
}

export interface UserProfile {
  user_id:        string
  susceptibility: number
  n_updates:      number
  vector:         number[]
  top_techniques: Technique[]
}

export interface SDGMetrics {
  per:                  number
  provider_parity_gap:  number
  explainability_cover: number
  high_risk_articles:   number
  total_articles:       number
  avg_persuasion_risk:  number
}

export interface ScoreResponse {
  persuasion_risk:     number
  risk_label:          string
  risk_emoji:          string
  explanation:         string
  detected_techniques: Technique[]
}

// ── API calls ─────────────────────────────────────────────────────────────────

export const getRecommendations = (userId: string, topK = 10, alpha?: number, beta?: number) =>
  api.get<Article[]>(`/recommend/${userId}`, {
    params: { top_k: topK, alpha, beta },
  }).then(r => r.data)

export const getArticle = (newsId: string, userId = 'user_demo') =>
  api.get<Article>(`/article/${newsId}`, { params: { user_id: userId } }).then(r => r.data)

export const getUserProfile = (userId: string) =>
  api.get<UserProfile>(`/user/${userId}/profile`).then(r => r.data)

export const getSDGMetrics = () =>
  api.get<SDGMetrics>('/metrics/sdg').then(r => r.data)

export const scoreText = (text: string) =>
  api.post<ScoreResponse>('/score', { text }).then(r => r.data)

export const sendFeedback = (userId: string, newsId: string, action: string) =>
  api.post('/feedback', { user_id: userId, news_id: newsId, action }).then(r => r.data)

// ── Demo data (when API is unavailable) ──────────────────────────────────────

export const DEMO_ARTICLES: Article[] = [
  {
    news_id: 'N000000', title: 'SHOCKING: Government HIDING the truth about vaccines!',
    abstract: 'Anonymous sources claim officials have suppressed data for years.',
    category: 'politics', url: 'https://example-site.com/news/1',
    persuasion_risk: 0.82, risk_label: 'HIGH', risk_emoji: '🔴', risk_color: '#ef4444',
    responsible_score: 0.24, relevance: 0.78, explanation:
      '🔴 This article uses strong persuasion techniques. Detected: Appeal to Fear-Prejudice, Loaded Language, Doubt.',
    top_techniques: [
      { technique: 'Appeal_to_Fear-Prejudice', score: 0.91 },
      { technique: 'Loaded_Language', score: 0.85 },
      { technique: 'Doubt', score: 0.72 },
    ],
  },
  {
    news_id: 'N000001', title: 'Scientists publish new findings on climate resilience',
    abstract: 'A new study from MIT shows coastal ecosystems may adapt faster than predicted.',
    category: 'science', url: 'https://reuters.com/news/2',
    persuasion_risk: 0.08, risk_label: 'LOW', risk_emoji: '🟢', risk_color: '#22c55e',
    responsible_score: 0.85, relevance: 0.80,
    explanation: '🟢 This article uses minimal persuasion techniques and appears informational.',
    top_techniques: [{ technique: 'Reasoning', score: 0.70 }],
  },
  {
    news_id: 'N000002', title: 'New AI regulations proposed by European Parliament',
    abstract: 'MEPs debate comprehensive framework for high-risk AI systems.',
    category: 'technology', url: 'https://bbc.com/news/3',
    persuasion_risk: 0.12, risk_label: 'LOW', risk_emoji: '🟢', risk_color: '#22c55e',
    responsible_score: 0.82, relevance: 0.76,
    explanation: '🟢 This article uses minimal persuasion techniques and appears informational.',
    top_techniques: [{ technique: 'Reasoning', score: 0.65 }],
  },
  {
    news_id: 'N000003', title: 'Everyone agrees: this is the ONLY way forward on housing',
    abstract: 'Policy experts claim a single solution will fix the housing crisis.',
    category: 'politics', url: 'https://cnn.com/news/4',
    persuasion_risk: 0.60, risk_label: 'HIGH', risk_emoji: '🔴', risk_color: '#ef4444',
    responsible_score: 0.41, relevance: 0.72,
    explanation: '🔴 This article uses strong persuasion techniques. Detected: Bandwagon, Black-and-White Fallacy.',
    top_techniques: [
      { technique: 'Bandwagon', score: 0.75 },
      { technique: 'Black-and-White_Fallacy', score: 0.65 },
    ],
  },
  {
    news_id: 'N000004', title: 'Parliament passes new infrastructure investment bill',
    abstract: '£40bn allocated for road, rail, and broadband improvements over five years.',
    category: 'politics', url: 'https://theguardian.com/news/5',
    persuasion_risk: 0.05, risk_label: 'LOW', risk_emoji: '🟢', risk_color: '#22c55e',
    responsible_score: 0.88, relevance: 0.84,
    explanation: '🟢 This article uses minimal persuasion techniques and appears informational.',
    top_techniques: [],
  },
  {
    news_id: 'N000005', title: 'Moderate concern: new drug shows promise but questions remain',
    abstract: 'Phase 2 trials reveal positive results; researchers urge cautious optimism.',
    category: 'health', url: 'https://apnews.com/news/6',
    persuasion_risk: 0.32, risk_label: 'MEDIUM', risk_emoji: '🟡', risk_color: '#f59e0b',
    responsible_score: 0.67, relevance: 0.71,
    explanation: '🟡 This article uses moderate persuasion techniques. Some exaggeration detected.',
    top_techniques: [
      { technique: 'Exaggeration-Minimisation', score: 0.45 },
      { technique: 'Appeal_to_Authority', score: 0.38 },
    ],
  },
]

export const DEMO_PROFILE: UserProfile = {
  user_id: 'user_demo',
  susceptibility: 0.42,
  n_updates: 87,
  vector: [
    0.12, 0.65, 0.31, 0.18, 0.22, 0.44, 0.15, 0.08, 0.10,
    0.72, 0.55, 0.06, 0.28, 0.19, 0.14, 0.11, 0.33, 0.09,
    0.21, 0.38, 0.17, 0.13, 0.07,
  ],
  top_techniques: [
    { technique: 'Loaded_Language',           score: 0.72 },
    { technique: 'Appeal_to_Fear-Prejudice',  score: 0.65 },
    { technique: 'Name_Calling-Labeling',     score: 0.55 },
    { technique: 'Doubt',                     score: 0.44 },
    { technique: 'Thought-terminating_Clichés', score: 0.33 },
  ],
}

export const DEMO_SDG: SDGMetrics = {
  per:                  0.18,
  provider_parity_gap:  0.04,
  explainability_cover: 0.97,
  high_risk_articles:   3,
  total_articles:       6,
  avg_persuasion_risk:  0.33,
}
