import { useCallback, useRef, useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const CLASS_INFO = {
  MEL: { label: 'Melanoma', risk: 'high' },
  NV: { label: 'Melanocytic Nevus', risk: 'low' },
  BCC: { label: 'Basal Cell Carcinoma', risk: 'high' },
  AKIEC: { label: 'Actinic Keratosis', risk: 'high' },
  BKL: { label: 'Benign Keratosis', risk: 'low' },
  DF: { label: 'Dermatofibroma', risk: 'low' },
  VASC: { label: 'Vascular Lesion', risk: 'low' },
}

function Dropzone({ onFile, previewUrl }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault()
      setDragOver(false)
      const file = e.dataTransfer.files?.[0]
      if (file) onFile(file)
    },
    [onFile],
  )

  return (
    <div
      className={`dropzone ${dragOver ? 'dropzone--over' : ''} ${previewUrl ? 'dropzone--filled' : ''}`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFile(file)
        }}
      />
      {!previewUrl && (
        <>
          <span className="dropzone__eyebrow">STEP 01</span>
          <p className="dropzone__title">Drop a dermoscopic image</p>
          <p className="dropzone__hint">or click to browse — JPG or PNG</p>
        </>
      )}
    </div>
  )
}

function Lens({ imageUrl, gradcamUrl, showGradcam, scanning }) {
  return (
    <div className="lens">
      <div className="lens__ring">
        <div className="lens__viewport">
          {imageUrl && (
            <img
              src={showGradcam && gradcamUrl ? gradcamUrl : imageUrl}
              alt={showGradcam ? 'Grad-CAM heatmap overlay' : 'Uploaded lesion'}
              className="lens__image"
            />
          )}
          {scanning && <div className="lens__scanline" />}
        </div>
      </div>
      {gradcamUrl && <span className="lens__caption">{showGradcam ? 'Grad-CAM overlay' : 'Original image'}</span>}
    </div>
  )
}

function ProbabilityBars({ probabilities, predictedClass }) {
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1])
  return (
    <div className="bars">
      {entries.map(([code, value]) => {
        const info = CLASS_INFO[code] || { label: code, risk: 'low' }
        const isPred = code === predictedClass
        return (
          <div className="bars__row" key={code}>
            <div className="bars__label">
              <span className="bars__code">{code}</span>
              <span className="bars__name">{info.label}</span>
            </div>
            <div className="bars__track">
              <div
                className={`bars__fill bars__fill--${info.risk} ${isPred ? 'bars__fill--pred' : ''}`}
                style={{ width: `${Math.max(value * 100, 1.5)}%` }}
              />
            </div>
            <span className="bars__value">{(value * 100).toFixed(1)}%</span>
          </div>
        )
      })}
    </div>
  )
}

function generateDemoLesion() {
  const size = 260
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')

  const skinGrad = ctx.createRadialGradient(size * 0.4, size * 0.35, 10, size * 0.5, size * 0.5, size * 0.8)
  skinGrad.addColorStop(0, '#e8b894')
  skinGrad.addColorStop(1, '#c1875f')
  ctx.fillStyle = skinGrad
  ctx.fillRect(0, 0, size, size)

  for (let i = 0; i < 250; i++) {
    ctx.fillStyle = `rgba(90,50,30,${Math.random() * 0.05})`
    ctx.beginPath()
    ctx.arc(Math.random() * size, Math.random() * size, Math.random() * 1.4, 0, Math.PI * 2)
    ctx.fill()
  }

  ctx.save()
  ctx.translate(size * 0.5, size * 0.52)
  ctx.beginPath()
  const points = 12
  for (let i = 0; i <= points; i++) {
    const angle = (i / points) * Math.PI * 2
    const r = 52 + Math.sin(angle * 3) * 9 + Math.cos(angle * 5) * 5
    const x = Math.cos(angle) * r
    const y = Math.sin(angle) * r * 0.9
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.closePath()
  const lesionGrad = ctx.createRadialGradient(0, 0, 4, 0, 0, 58)
  lesionGrad.addColorStop(0, '#241109')
  lesionGrad.addColorStop(0.55, '#452314')
  lesionGrad.addColorStop(1, '#6b3a22')
  ctx.fillStyle = lesionGrad
  ctx.fill()
  ctx.restore()

  const originalUrl = canvas.toDataURL('image/png')

  const heatGrad = ctx.createRadialGradient(size * 0.5, size * 0.53, 4, size * 0.5, size * 0.53, size * 0.5)
  heatGrad.addColorStop(0, 'rgba(255,30,20,0.85)')
  heatGrad.addColorStop(0.35, 'rgba(255,170,20,0.55)')
  heatGrad.addColorStop(0.6, 'rgba(130,220,90,0.28)')
  heatGrad.addColorStop(0.82, 'rgba(40,120,220,0.14)')
  heatGrad.addColorStop(1, 'rgba(40,120,220,0)')
  ctx.fillStyle = heatGrad
  ctx.fillRect(0, 0, size, size)

  const gradcamUrl = canvas.toDataURL('image/png')

  return { originalUrl, gradcamBase64: gradcamUrl.split(',')[1] }
}

const DEMO_RESULT = {
  predicted_class: 'MEL',
  confidence: 0.87,
  probabilities: { MEL: 0.87, NV: 0.04, BCC: 0.03, AKIEC: 0.02, BKL: 0.02, DF: 0.01, VASC: 0.01 },
  shap_top_features: [
    { feature: 'fused_dim_146', shap_value: 0.31 },
    { feature: 'fused_dim_212', shap_value: 0.18 },
    { feature: 'fused_dim_58', shap_value: -0.12 },
    { feature: 'fused_dim_301', shap_value: 0.09 },
    { feature: 'fused_dim_77', shap_value: -0.07 },
  ],
}

function ShapChart({ features }) {
  if (!features || features.length === 0) return null
  const maxAbs = Math.max(...features.map((f) => Math.abs(f.shap_value)), 0.001)
  return (
    <div className="shap">
      <h3 className="shap__title">Feature attribution (SHAP)</h3>
      <p className="shap__hint">
        Contribution of the top fused CNN+ViT feature dimensions to this prediction.
      </p>
      <div className="shap__rows">
        {features.map((f) => {
          const pct = (Math.abs(f.shap_value) / maxAbs) * 100
          const positive = f.shap_value >= 0
          return (
            <div className="shap__row" key={f.feature}>
              <span className="shap__feature">{f.feature}</span>
              <div className="shap__track">
                <div
                  className={`shap__fill ${positive ? 'shap__fill--pos' : 'shap__fill--neg'}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="shap__value">{f.shap_value >= 0 ? '+' : ''}{f.shap_value.toFixed(3)}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [showGradcam, setShowGradcam] = useState(true)
  const [isDemo, setIsDemo] = useState(false)
  // Optional clinical metadata (the deployed checkpoint was trained with it).
  const [age, setAge] = useState('')
  const [sex, setSex] = useState('')
  const [localization, setLocalization] = useState('')

  const runAnalysis = useCallback(async (f) => {
    if (!f) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const formData = new FormData()
      formData.append('file', f)
      if (age !== '') formData.append('age', age)
      if (sex) formData.append('sex', sex)
      if (localization) formData.append('localization', localization)
      const res = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Request failed (${res.status})`)
      }
      const data = await res.json()
      setResult(data)
      setShowGradcam(true)
    } catch (e) {
      setError(e.message || 'Something went wrong reaching the model server.')
    } finally {
      setLoading(false)
    }
  }, [age, sex, localization])

  const handleFile = useCallback((f) => {
    setFile(f)
    setResult(null)
    setError(null)
    setIsDemo(false)
    setPreviewUrl(URL.createObjectURL(f))
    runAnalysis(f) // analyse automatically as soon as an image is chosen
  }, [runAnalysis])

  const handleDemo = useCallback(() => {
    setFile(null)
    setError(null)
    setResult(null)
    setIsDemo(true)
    setLoading(true)
    setShowGradcam(true)
    const { originalUrl, gradcamBase64 } = generateDemoLesion()
    setPreviewUrl(originalUrl)
    window.setTimeout(() => {
      setResult({ ...DEMO_RESULT, gradcam_image_base64: gradcamBase64 })
      setLoading(false)
    }, 900)
  }, [])

  const gradcamUrl = result?.gradcam_image_base64
    ? `data:image/png;base64,${result.gradcam_image_base64}`
    : null

  const predictedInfo = result ? CLASS_INFO[result.predicted_class] || {} : null

  return (
    <div className="page">
      <header className="header">
        <span className="header__eyebrow">XAI-SKINNET · DIAGNOSTIC ASSISTANT</span>
        <h1 className="header__title">Hybrid CNN–ViT lesion classifier, explained</h1>
        <p className="header__sub">
          Upload a dermoscopic image to see the model's prediction alongside the visual and
          feature-level reasoning behind it.
        </p>
      </header>

      <main className="layout">
        <section className="panel panel--left">
          <Dropzone onFile={handleFile} previewUrl={previewUrl} />
          {!previewUrl && (
            <button type="button" className="demo-btn" onClick={handleDemo}>
              ▶ No checkpoint yet? Try a demo sample
            </button>
          )}
          {previewUrl && (
            <>
              <Lens
                imageUrl={previewUrl}
                gradcamUrl={gradcamUrl}
                showGradcam={showGradcam}
                scanning={loading}
              />
              {gradcamUrl && (
                <div className="toggle">
                  <button
                    className={!showGradcam ? 'toggle__btn toggle__btn--active' : 'toggle__btn'}
                    onClick={() => setShowGradcam(false)}
                  >
                    Original
                  </button>
                  <button
                    className={showGradcam ? 'toggle__btn toggle__btn--active' : 'toggle__btn'}
                    onClick={() => setShowGradcam(true)}
                  >
                    Grad-CAM
                  </button>
                </div>
              )}
              {!isDemo && (
                <div className="meta">
                  <p className="meta__title">Patient details <span>(optional, improves accuracy)</span></p>
                  <div className="meta__row">
                    <input
                      type="number" min="0" max="100" placeholder="Age"
                      value={age} onChange={(e) => setAge(e.target.value)}
                    />
                    <select value={sex} onChange={(e) => setSex(e.target.value)}>
                      <option value="">Sex</option>
                      <option value="male">Male</option>
                      <option value="female">Female</option>
                    </select>
                    <select value={localization} onChange={(e) => setLocalization(e.target.value)}>
                      <option value="">Body site</option>
                      {['back','lower extremity','trunk','upper extremity','abdomen','face','chest','foot','neck','scalp','hand','ear','genital','acral'].map((l) => (
                        <option key={l} value={l}>{l}</option>
                      ))}
                    </select>
                  </div>
                </div>
              )}
              <button
                className="analyze-btn"
                onClick={isDemo ? handleDemo : () => runAnalysis(file)}
                disabled={loading}
              >
                {loading
                  ? 'Analyzing…'
                  : isDemo
                    ? 'Try another sample'
                    : 'Re-analyze with these details'}
              </button>
            </>
          )}
          {error && <p className="error">{error}</p>}
        </section>

        <section className="panel panel--right">
          {!result && !loading && (
            <div className="placeholder">
              <span className="placeholder__eyebrow">STEP 02</span>
              <p>Results will appear here once an image is analyzed.</p>
            </div>
          )}

          {loading && (
            <div className="placeholder">
              <span className="placeholder__eyebrow">RUNNING INFERENCE</span>
              <p>CNN + ViT branches → fusion → classification → Grad-CAM → SHAP…</p>
            </div>
          )}

          {result && !loading && (
            <div className="result">
              {isDemo && (
                <p className="demo-note">
                  Simulated demo — synthetic image and mock output, not a real model inference.
                </p>
              )}
              <div className="result__headline">
                <span className="result__eyebrow">
                  PREDICTED CLASS
                  {predictedInfo?.risk === 'high' && <span className="risk-tag">FLAG FOR REVIEW</span>}
                </span>
                <div className="result__class">
                  <span className="result__code">{result.predicted_class}</span>
                  <span className="result__name">{predictedInfo?.label}</span>
                </div>
                <span className="result__confidence">
                  {(result.confidence * 100).toFixed(1)}% confidence
                </span>
              </div>

              <ProbabilityBars
                probabilities={result.probabilities}
                predictedClass={result.predicted_class}
              />

              {result.branch_weights && (
                <p className="branch-note">
                  Model relied on: CNN {Math.round(result.branch_weights.cnn * 100)}% ·
                  ViT {Math.round(result.branch_weights.vit * 100)}%
                  {result.branch_weights.metadata > 0 &&
                    ` · Patient details ${Math.round(result.branch_weights.metadata * 100)}%`}
                  . The heatmap blends both image branches in these proportions.
                </p>
              )}

              <ShapChart features={result.shap_top_features} />
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        Research/educational demonstration only — not a certified diagnostic device, and not a
        substitute for evaluation by a dermatologist.
      </footer>
    </div>
  )
}
