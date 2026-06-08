export type Paper = {
  id: number;
  name: string;
  page_count: number;
  chunk_count: number;
  created_at: string;
};

export type CorpusStatus = {
  domain_name: string;
  required_documents: number;
  uploaded_documents: number;
  remaining_documents: number;
  total_pages: number;
  total_chunks: number;
  ready: boolean;
};

export type Citation = {
  paper_name: string;
  page_number: number;
  chunk_id: string;
  score: number;
};

export type ChatResponse = {
  answer: string;
  citations: Citation[];
  simple_explanation: string;
  key_insights: string[];
  confidence: number;
  formatted: string;
  analytics?: CorrectiveAnalytics | null;
};

export type Message = {
  id?: number;
  role: "user" | "assistant";
  content: string;
  created_at?: string;
  response?: ChatResponse;
};

export type CorrectiveChunkGrade = {
  chunk_id: string;
  page_number: number;
  grade: string;
  score: number;
  reason: string;
};

export type CorrectiveAnalytics = {
  retrieval_confidence: number;
  correction_attempts: number;
  final_confidence: number;
  chunks_accepted: number;
  chunks_rejected: number;
  grounding_check_result: string;
  chunk_grades: CorrectiveChunkGrade[];
  rewritten_query?: string | null;
  response_time_ms: number;
};

export type EvaluationQuestion = {
  question: string;
  expected_answer_terms: string[];
  expected_paper_id?: number;
};

export type EvaluationResult = {
  question: string;
  answer: string;
  retrieval_hit: boolean;
  answer_quality: number;
  confidence: number;
  matched_terms: string[];
  citations: Citation[];
};

export type EvaluationResponse = {
  retrieval_accuracy: number;
  answer_quality: number;
  total_questions: number;
  results: EvaluationResult[];
};
