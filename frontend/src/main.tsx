import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BookOpen,
  FileText,
  Loader2,
  Menu,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  Send,
  Sun,
  Trash2,
} from "lucide-react";
import {
  askQuestion,
  clearPaperHistory,
  evaluateCorpus,
  fetchCorpusStatus,
  fetchPaperHistory,
  fetchPapers,
} from "./api";
import type { ChatResponse, CorpusStatus, CorrectiveAnalytics, EvaluationResponse, Message, Paper } from "./types";
import "./styles.css";

function App() {
  const [dark, setDark] = useState(() => localStorage.getItem("theme") === "dark");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sessionId] = useState(() => {
    const existing = localStorage.getItem("ragSessionId");
    if (existing) return existing;
    const created = `session-${crypto.randomUUID()}`;
    localStorage.setItem("ragSessionId", created);
    return created;
  });
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedPaperId, setSelectedPaperId] = useState<number | undefined>();
  const [corpusStatus, setCorpusStatus] = useState<CorpusStatus | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [evaluationText, setEvaluationText] = useState("What is the main objective? | objective, method\nWhat limitations are reported? | limitation, future");
  const [evaluation, setEvaluation] = useState<EvaluationResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState({
    papers: true,
    history: false,
    chat: false,
    evaluate: false,
    clearHistory: false,
  });

  const selectedPaper = papers.find((paper) => paper.id === selectedPaperId);
  const latestAnswer = [...messages].reverse().find((message) => message.role === "assistant")?.response;

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("theme", dark ? "dark" : "light");
  }, [dark]);

  useEffect(() => {
    Promise.all([fetchPapers(), fetchCorpusStatus()])
      .then(([paperData, status]) => {
        setPapers(paperData);
        setCorpusStatus(status);
        setSelectedPaperId(undefined);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading((state) => ({ ...state, papers: false })));
  }, []);

  useEffect(() => {
    if (papers.length === 0) {
      setMessages([]);
      return;
    }
    setLoading((state) => ({ ...state, history: true }));
    fetchPaperHistory(selectedPaperId, sessionId)
      .then(setMessages)
      .finally(() => setLoading((state) => ({ ...state, history: false })));
  }, [selectedPaperId, sessionId, papers.length]);

  const stats = useMemo(
    () => [
      { label: "Documents", value: papers.length.toString() },
      { label: "Pages", value: papers.reduce((total, paper) => total + paper.page_count, 0).toString() },
      { label: "Chunks", value: papers.reduce((total, paper) => total + paper.chunk_count, 0).toString() },
    ],
    [papers]
  );

  async function handleClearHistory() {
    if (!selectedPaperId) return;
    const confirmed = window.confirm("Clear chat history for this paper?");
    if (!confirmed) return;
    setLoading((state) => ({ ...state, clearHistory: true }));
    try {
      await clearPaperHistory(selectedPaperId, sessionId);
      setMessages([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not clear history");
    } finally {
      setLoading((state) => ({ ...state, clearHistory: false }));
    }
  }

  async function handleAsk(event: React.FormEvent) {
    event.preventDefault();
    if (!question.trim()) return;
    const prompt = question.trim();
    setQuestion("");
    setError("");
    setMessages((current) => [...current, { role: "user", content: prompt }]);
    setLoading((state) => ({ ...state, chat: true }));
    try {
      const response = await askQuestion(prompt, selectedPaperId, sessionId);
      setMessages((current) => [...current, { role: "assistant", content: response.formatted, response }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not answer question");
    } finally {
      setLoading((state) => ({ ...state, chat: false }));
    }
  }

  async function handleEvaluate() {
    const questions = evaluationText
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const [prompt, terms = ""] = line.split("|");
        return {
          question: prompt.trim(),
          expected_answer_terms: terms.split(",").map((term) => term.trim()).filter(Boolean),
        };
      })
      .filter((item) => item.question);
    if (!questions.length) return;
    setLoading((state) => ({ ...state, evaluate: true }));
    setError("");
    try {
      setEvaluation(await evaluateCorpus(questions, `${sessionId}-evaluation`));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Evaluation failed");
    } finally {
      setLoading((state) => ({ ...state, evaluate: false }));
    }
  }

  return (
    <main className="min-h-screen bg-paper text-ink transition dark:bg-[#101722] dark:text-slate-100">
      <div className={`app-shell ${sidebarOpen ? "app-shell-open" : "app-shell-closed"}`}>
        <aside className={`sidebar ${sidebarOpen ? "block" : "hidden lg:block"}`}>
          <div className="mb-5 flex items-center justify-between">
            <div className="flex items-center gap-2 font-semibold">
              <BookOpen className="h-5 w-5 text-coral" />
              {sidebarOpen && "Cancer RAG"}
            </div>
            <button className="icon-button" onClick={() => setSidebarOpen(!sidebarOpen)} title="Toggle sidebar">
              {sidebarOpen ? <PanelLeftClose /> : <PanelLeftOpen />}
            </button>
          </div>

          {sidebarOpen && (
            <>
              {corpusStatus && <CorpusStatusCard status={corpusStatus} />}

              <div className="mt-6 flex items-center justify-between text-sm font-semibold">
                <span>Backend Source Documents</span>
              </div>
              <div className="mt-3 space-y-2">
                {loading.papers && <EmptyLine text="Loading indexed corpus..." />}
                {!loading.papers && papers.length === 0 && <EmptyLine text="No backend documents indexed yet." />}
                {papers.length > 0 && (
                  <button className={`paper-row ${selectedPaperId === undefined ? "paper-button-active" : ""}`} onClick={() => setSelectedPaperId(undefined)}>
                    <span className="flex items-center gap-2">
                      <BookOpen className="h-4 w-4 shrink-0" />
                      <span className="font-medium">All corpus documents</span>
                    </span>
                  </button>
                )}
                {papers.map((paper) => (
                  <PaperButton key={paper.id} paper={paper} active={selectedPaperId === paper.id} onSelect={() => setSelectedPaperId(paper.id)} />
                ))}
              </div>
            </>
          )}
        </aside>

        <section className="min-w-0 p-4">
          <header className="workspace-header">
            <button className="icon-button lg:hidden" onClick={() => setSidebarOpen(!sidebarOpen)} title="Menu">
              <Menu />
            </button>
            <div className="min-w-0 flex-1">
              <p className="text-sm text-slate-500 dark:text-slate-400">{corpusStatus?.domain_name ?? "Cancer research corpus"}</p>
              <h1 className="max-w-5xl text-xl font-semibold leading-snug lg:text-2xl">Cancer Research RAG Chatbot with Retrieval Accuracy and Answer Quality Evaluation</h1>
              {selectedPaper ? (
                <p className="text-xs text-slate-500">
                  Indexed {formatDate(selectedPaper.created_at)} - {selectedPaper.page_count} pages - {selectedPaper.chunk_count} chunks
                </p>
              ) : (
                corpusStatus && (
                  <p className="text-xs text-slate-500">
                    Backend corpus: {corpusStatus.uploaded_documents}/{corpusStatus.required_documents} PDFs - {corpusStatus.total_chunks} chunks
                  </p>
                )
              )}
            </div>
            <button className="icon-button" onClick={() => setDark(!dark)} title="Toggle theme">
              {dark ? <Sun /> : <Moon />}
            </button>
          </header>

          <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
            <div className="min-w-0 space-y-4">
              <div className="grid gap-3 sm:grid-cols-3">
                {stats.map((item) => (
                  <MetricCard key={item.label} label={item.label} value={item.value} />
                ))}
              </div>

              <section className="panel">
                <div className="panel-header">
                  <div>
                    <h2 className="font-semibold">Chat</h2>
                    <p className="text-xs text-slate-500">Corpus retrieval - grading - correction - cited answer</p>
                  </div>
                  <button className="secondary-button" disabled={!selectedPaperId || loading.clearHistory} onClick={handleClearHistory}>
                    {loading.clearHistory ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                    Clear history
                  </button>
                </div>
                <div className="chat-window">
                  {loading.history && <EmptyLine text="Loading chat history..." />}
                  {!loading.history && messages.length === 0 && (
                    <EmptyLine text={papers.length ? "Ask across the full backend corpus, or select one document for paper-specific questions." : "Index backend source PDFs to start chatting."} />
                  )}
                  {messages.map((message, index) => (
                    <div key={message.id ?? index} className={`message ${message.role === "user" ? "message-user" : "message-assistant"}`}>
                      <pre>{message.content}</pre>
                    </div>
                  ))}
                  {loading.chat && (
                    <div className="message message-assistant flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Running corpus RAG...
                    </div>
                  )}
                </div>
                <form onSubmit={handleAsk} className="chat-form">
                  <input
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    className="chat-input"
                    placeholder="Ask a citation-grounded corpus question"
                    disabled={papers.length === 0}
                  />
                  <button className="primary-button" disabled={loading.chat || !question.trim()}>
                    <Send className="h-4 w-4" />
                    Ask
                  </button>
                </form>
              </section>
            </div>

            <aside className="space-y-4">
              <CorrectivePanel analytics={latestAnswer?.analytics} />
              <EvaluationPanel value={evaluationText} result={evaluation} loading={loading.evaluate} onChange={setEvaluationText} onRun={handleEvaluate} />
              <CitationPanel response={latestAnswer} />
              <section className="panel p-4 text-sm">
                <h2 className="mb-2 font-semibold">Sample Queries</h2>
                <ul className="space-y-2 text-slate-600 dark:text-slate-300">
                  <li>Summarize the common methods across the corpus.</li>
                  <li>What limitations are reported?</li>
                  <li>Compare findings across the documents.</li>
                  <li>Create viva questions from this domain.</li>
                </ul>
              </section>
              {error && <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{error}</div>}
            </aside>
          </div>
        </section>
      </div>
    </main>
  );
}

function CorpusStatusCard({ status }: { status: CorpusStatus }) {
  const progress = Math.min(100, Math.round((status.uploaded_documents / status.required_documents) * 100));
  return (
    <section className="rounded-lg border border-black/10 bg-white p-4 text-sm dark:border-white/10 dark:bg-slate-900">
      <div className="flex items-center justify-between gap-3">
        <span className="font-semibold">50-PDF Corpus</span>
        <span className={status.ready ? "text-green-600" : "text-coral"}>{status.uploaded_documents}/{status.required_documents}</span>
      </div>
      <ProgressBar value={progress} text={status.ready ? "Ready for evaluation" : `${status.remaining_documents} backend PDF(s) still needed`} />
      <p className="mt-3 text-xs text-slate-500">Put PDFs in data/source_documents and run backend ingestion.</p>
    </section>
  );
}

function PaperButton({ paper, active, onSelect }: { paper: Paper; active: boolean; onSelect: () => void }) {
  return (
    <div className={`paper-row ${active ? "paper-button-active" : ""}`}>
      <button onClick={onSelect} className="min-w-0 flex-1 text-left">
        <span className="flex items-center gap-2">
          <FileText className="h-4 w-4 shrink-0" />
          <span className="truncate font-medium">{paper.name}</span>
        </span>
        <span className="mt-1 block text-xs text-slate-500">
          {formatDate(paper.created_at)} - {paper.page_count}p - {paper.chunk_count} chunks
        </span>
      </button>
    </div>
  );
}

function EvaluationPanel({
  value,
  result,
  loading,
  onChange,
  onRun,
}: {
  value: string;
  result: EvaluationResponse | null;
  loading: boolean;
  onChange: (value: string) => void;
  onRun: () => void;
}) {
  return (
    <section className="panel p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="font-semibold">RAG Evaluation</h2>
        <button className="secondary-button" onClick={onRun} disabled={loading}>
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
          Run
        </button>
      </div>
      <textarea
        className="min-h-28 w-full rounded-md border border-black/10 bg-white p-3 text-sm outline-none ring-coral/30 focus:ring-4 dark:border-white/10 dark:bg-slate-950"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
      {result && (
        <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
          <MetricCard label="Retrieval accuracy" value={`${Math.round(result.retrieval_accuracy * 100)}%`} />
          <MetricCard label="Answer quality" value={`${Math.round(result.answer_quality * 100)}%`} />
        </div>
      )}
    </section>
  );
}

function CorrectivePanel({ analytics }: { analytics?: CorrectiveAnalytics | null }) {
  const rows = analytics
    ? [
        ["Retrieval confidence", analytics.retrieval_confidence.toString()],
        ["Correction attempts", analytics.correction_attempts.toString()],
        ["Final confidence", analytics.final_confidence.toString()],
        ["Grounding", analytics.grounding_check_result],
        ["Accepted chunks", analytics.chunks_accepted.toString()],
        ["Rejected chunks", analytics.chunks_rejected.toString()],
        ["Response time", `${analytics.response_time_ms} ms`],
      ]
    : [];
  return (
    <section className="panel p-4">
      <h2 className="mb-3 font-semibold">Corrective RAG Details</h2>
      {!analytics && <EmptyLine text="Details appear after an answer." />}
      {analytics && (
        <div className="space-y-2 text-sm">
          {rows.map(([label, value]) => (
            <div key={label} className="flex justify-between gap-3 rounded-md bg-slate-50 px-3 py-2 dark:bg-slate-950">
              <span className="text-slate-500">{label}</span>
              <span className="text-right font-medium">{value}</span>
            </div>
          ))}
          {analytics.rewritten_query && (
            <div className="rounded-md bg-slate-50 px-3 py-2 dark:bg-slate-950">
              <p className="text-slate-500">Rewritten query</p>
              <p className="mt-1 text-sm">{analytics.rewritten_query}</p>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function CitationPanel({ response }: { response?: ChatResponse }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h2 className="font-semibold">Citations</h2>
      </div>
      <div className="space-y-3 p-4">
        {!response?.citations.length && <EmptyLine text="Citations appear after an answer." />}
        {response?.citations.map((citation) => (
          <div key={citation.chunk_id} className="rounded-md border border-black/10 p-3 text-sm dark:border-white/10">
            <p className="font-medium">{citation.paper_name}</p>
            <p className="text-slate-500 dark:text-slate-400">Page {citation.page_number}</p>
            <p className="break-words text-xs text-slate-500 dark:text-slate-400">{citation.chunk_id}</p>
            <p className="mt-1 text-xs">Score {citation.score}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-black/10 bg-white p-4 dark:border-white/10 dark:bg-slate-900">
      <p className="text-sm text-slate-500 dark:text-slate-400">{label}</p>
      <p className="text-2xl font-semibold">{value}</p>
    </div>
  );
}

function SummaryBody({ loading, error, text }: { loading: boolean; error: string; text: string }) {
  if (loading) {
    return (
      <div className="flex items-center gap-2 p-4 text-sm text-slate-600 dark:text-slate-300">
        <Loader2 className="h-5 w-5 animate-spin text-coral" />
        Generating summary...
      </div>
    );
  }
  if (error) return <p className="m-4 rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{error}</p>;
  return <pre className="max-h-[460px] overflow-y-auto whitespace-pre-wrap break-words p-5 font-sans text-sm leading-6 text-slate-600 dark:text-slate-300">{text}</pre>;
}

function ProgressBar({ value, text }: { value: number; text: string }) {
  return (
    <div className="mt-3">
      <div className="h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
        <div className="h-full bg-coral transition-all" style={{ width: `${value}%` }} />
      </div>
      <p className="mt-1 text-xs text-slate-500">{text}</p>
    </div>
  );
}

function EmptyLine({ text }: { text: string }) {
  return <p className="rounded-md border border-dashed border-black/10 p-3 text-sm text-slate-500 dark:border-white/10 dark:text-slate-400">{text}</p>;
}

function formatDate(value: string) {
  return new Date(value).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

createRoot(document.getElementById("root")!).render(<App />);
