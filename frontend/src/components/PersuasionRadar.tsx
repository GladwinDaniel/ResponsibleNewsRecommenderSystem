import { useEffect, useRef } from 'react'
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js'
import { Radar } from 'react-chartjs-2'
import { UserProfile } from '../api'

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend)

const TECHNIQUE_LABELS_SHORT = [
  'Auth.', 'Fear', 'Bandwagon', 'False Dil.', 'Causal O.',
  'Doubt', 'Exagger.', 'Flag', 'Glitter.', 'Loaded',
  'Name Call.', 'Obfusc.', 'Reasoning', 'Red H.', 'Repetit.',
  'Slogans', 'Cliché', 'Transfer', 'Whatabout.', 'Straw M.',
  'Hypocrisy', 'Conseq. O.', 'Time',
]

interface PersuasionRadarProps {
  profile: UserProfile
  compact?: boolean
}

export default function PersuasionRadar({ profile, compact = false }: PersuasionRadarProps) {
  const vector = profile.vector.length > 0
    ? profile.vector.slice(0, 23)
    : new Array(23).fill(0)

  const data = {
    labels: TECHNIQUE_LABELS_SHORT,
    datasets: [
      {
        label: 'Susceptibility',
        data: vector,
        backgroundColor: 'rgba(124, 58, 237, 0.15)',
        borderColor: 'rgba(124, 58, 237, 0.8)',
        borderWidth: 2,
        pointBackgroundColor: 'rgba(124, 58, 237, 0.9)',
        pointBorderColor: 'transparent',
        pointRadius: compact ? 2 : 3,
        fill: true,
      },
    ],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: true,
    scales: {
      r: {
        beginAtZero: true,
        min: 0,
        max: 1,
        ticks: {
          stepSize: 0.25,
          color: '#64748b',
          font: { size: 9 },
          backdropColor: 'transparent',
        },
        grid: {
          color: 'rgba(255,255,255,0.06)',
        },
        angleLines: {
          color: 'rgba(255,255,255,0.06)',
        },
        pointLabels: {
          color: '#94a3b8',
          font: { size: compact ? 8 : 9 },
        },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1a2235',
        titleColor: '#f1f5f9',
        bodyColor: '#94a3b8',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        callbacks: {
          label: (ctx: any) => ` ${(ctx.raw as number).toFixed(3)}`,
        },
      },
    },
  }

  return <Radar data={data} options={options} />
}
