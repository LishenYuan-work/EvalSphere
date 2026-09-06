// The deployed frontend must never try to call the visitor's own localhost.
// Keep localhost convenient for local development, while using the public API
// as a production fallback when Vercel variables were not injected.
// Accept either an API origin or an origin that already ends in `/api`.
// Every client path below includes its own `/api` prefix, so normalize the
// environment value to prevent production requests such as `/api/api/...`.
const configuredBase = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/+$/, "").replace(/\/api$/, "");
// The Render service was renamed; keep stale Vercel variables from routing
// production traffic through the retired service alias.
const canonicalBase = configuredBase?.replace(
  /^https:\/\/review-platform-api\.onrender\.com$/i,
  "https://1-0plp.onrender.com",
);
const BASE = (canonicalBase || (
  process.env.NODE_ENV === "production"
    ? "https://1-0plp.onrender.com"
    : "http://localhost:8000"
)).replace(/\/$/, "");

function connectionError(): Error {
  return new Error(
    `无法连接评审服务（${BASE}）。请检查后端是否在线，或联系管理员确认 NEXT_PUBLIC_API_URL 配置。`,
  );
}

const FIELD_LABELS: Record<string, string> = {
  email: "邮箱",
  password: "密码",
  display_name: "姓名",
  organization_name: "组织名称",
  topic: "调研主题",
  max_round: "评审轮次",
};

function formatApiError(payload: unknown, fallback: string): string {
  if (typeof payload === "string" && payload.trim()) {
    return payload.trim().toLowerCase() === "not found" ? "评审服务暂未部署最新版本，请稍后重试" : payload;
  }
  if (!payload || typeof payload !== "object") return fallback;

  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string" && detail.trim()) {
    return detail.trim().toLowerCase() === "not found" ? "评审服务暂未部署最新版本，请稍后重试" : detail;
  }
  if (Array.isArray(detail)) {
    const messages = detail.map((item) => {
      if (!item || typeof item !== "object") return String(item);
      const issue = item as { type?: unknown; msg?: unknown; loc?: unknown; ctx?: Record<string, unknown> };
      const location = Array.isArray(issue.loc)
        ? issue.loc.find((part) => typeof part === "string" && part !== "body")
        : undefined;
      const label = typeof location === "string" ? FIELD_LABELS[location] || location : undefined;
      let message = typeof issue.msg === "string" ? issue.msg : "请求参数无效";
      if (issue.type === "string_too_short" && typeof issue.ctx?.min_length === "number") {
        message = `至少需要 ${issue.ctx.min_length} 个字符`;
      } else if (issue.type === "string_too_long" && typeof issue.ctx?.max_length === "number") {
        message = `不能超过 ${issue.ctx.max_length} 个字符`;
      } else if (issue.type === "missing") {
        message = "不能为空";
      }
      return `${label ? `${label}：` : ""}${message}`;
    }).filter(Boolean);
    if (messages.length) return messages.join("；");
  }
  return fallback;
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  org?: string,
): Promise<T> {
  const headers: Record<string, string> = {};
  if (org) headers["X-Organization-ID"] = org;
  if (typeof document !== "undefined" && !["GET", "HEAD", "OPTIONS"].includes(method)) {
    let csrf = document.cookie
      .split(";")
      .map((item) => item.trim())
      .find((item) => item.startsWith("review_csrf="))
      ?.split("=")
      .slice(1)
      .join("=");
    // The CSRF cookie is scoped to the API origin and is not readable from
    // the separately deployed frontend. Ask the API for the matching value.
    if (!csrf) {
      let csrfResponse: Response;
      try {
        csrfResponse = await fetch(`${BASE}/api/auth/csrf`, { credentials: "include" });
      } catch {
        throw connectionError();
      }
      if (csrfResponse.ok) {
        const payload = (await csrfResponse.json()) as { csrf_token?: string };
        csrf = payload.csrf_token;
      }
    }
    if (csrf) headers["X-CSRF-Token"] = decodeURIComponent(csrf);
  }
  if (body instanceof FormData) {
    /* browser sets multipart boundary */
  } else if (body !== undefined) headers["Content-Type"] = "application/json";
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      headers,
      credentials: "include",
      body:
        body instanceof FormData
          ? body
          : body === undefined
            ? undefined
            : JSON.stringify(body),
    });
  } catch {
    throw connectionError();
  }
  if (!response.ok) {
    const payload = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(formatApiError(payload, response.statusText || "请求失败"));
  }
  if (response.status === 204) return undefined as T;
  return response.json();
}

export const api = {
  auth: {
    guest: () => request<AuthResponse>("POST", "/api/auth/guest"),
    register: (body: RegisterInput) =>
      request<AuthResponse>("POST", "/api/auth/register", body),
    login: (body: { email: string; password: string; remember_me: boolean }) =>
      request<AuthResponse>("POST", "/api/auth/login", body),
    supabaseExchange: (body: {
      access_token: string;
      display_name?: string;
      organization_name?: string;
      invite_token?: string;
    }) => request<AuthResponse>("POST", "/api/auth/supabase/exchange", body),
    supabaseLogin: (body: { email: string; password: string; remember_me: boolean }) =>
      request<AuthResponse>("POST", "/api/auth/supabase/login", body),
    verify: (token: string) =>
      request<{ status: string }>("POST", "/api/auth/verify-email", { token }),
    resend: (email: string) =>
      request<{ status: string }>("POST", "/api/auth/resend-verification", {
        email,
      }),
    forgot: (email: string) =>
      request<{ status: string }>("POST", "/api/auth/forgot-password", {
        email,
      }),
    reset: (token: string, password: string) =>
      request<{ status: string }>("POST", "/api/auth/reset-password", {
        token,
        password,
      }),
    me: () => request<UserProfile>("GET", "/api/auth/me"),
    refresh: () => request<AuthResponse>("POST", "/api/auth/refresh"),
    logout: () => request<{ status: string }>("POST", "/api/auth/logout"),
  },
  reviews: {
    list: (organizationId: string) =>
      request<ReviewSummary[]>(
        "GET",
        `/api/reviews?organization_id=${encodeURIComponent(organizationId)}`,
      ),
    create: (body: {
      organization_id: string;
      topic?: string;
      max_round: number;
    }) =>
      request<ReviewSummary>(
        "POST",
        "/api/reviews",
        body,
        body.organization_id,
      ),
    get: (id: string, org?: string) =>
      request<ReviewDetail>("GET", `/api/reviews/${id}`, undefined, org),
    upload: (id: string, file: File, org?: string) => {
      const form = new FormData();
      form.append("file", file);
      return request<ReviewDocument>(
        "POST",
        `/api/reviews/${id}/documents`,
        form,
        org,
      );
    },
    start: (id: string, org?: string) =>
      request<ReviewProgress>(
        "POST",
        `/api/reviews/${id}/start`,
        undefined,
        org,
      ),
    humanReview: (id: string, approved: boolean, note?: string, org?: string) =>
      request<ReviewProgress>(
        "POST",
        `/api/reviews/${id}/human-review`,
        { approved, note },
        org,
      ),
    evidence: (id: string, org?: string) =>
      request<EvidenceItem[]>(
        "GET",
        `/api/reviews/${id}/evidence`,
        undefined,
        org,
      ),
    report: (id: string, org?: string) =>
      request<{ markdown: string }>(
        "GET",
        `/api/reviews/${id}/report`,
        undefined,
        org,
      ),
    downloadUrl: (id: string) => `${BASE}/api/reviews/${id}/report.md`,
    delete: (id: string, org?: string) =>
      request<void>("DELETE", `/api/reviews/${id}`, undefined, org),
  },
  guestReviews: {
    create: (body: { topic?: string; max_round: number }) =>
      request<ReviewSummary>("POST", "/api/guest/reviews", body),
    get: (id: string) => request<ReviewDetail>("GET", `/api/guest/reviews/${id}`),
    upload: (id: string, file: File) => {
      const form = new FormData();
      form.append("file", file);
      return request<ReviewDocument>("POST", `/api/guest/reviews/${id}/documents`, form);
    },
    start: (id: string) => request<ReviewProgress>("POST", `/api/guest/reviews/${id}/start`),
    evidence: (id: string) => request<EvidenceItem[]>("GET", `/api/guest/reviews/${id}/evidence`),
    report: (id: string) => request<{ markdown: string }>("GET", `/api/guest/reviews/${id}/report`),
    downloadUrl: (id: string) => `${BASE}/api/guest/reviews/${id}/report.md`,
  },
  organizations: {
    invite: (id: string, email: string) =>
      request("POST", `/api/organizations/${id}/invites`, { email }),
    removeMember: (id: string, userId: string) =>
      request<void>("DELETE", `/api/organizations/${id}/members/${userId}`),
  },
};

export function authToken() {
  return null;
}
export function apiBase() {
  return BASE;
}
export interface RegisterInput {
  email: string;
  password: string;
  display_name?: string;
  organization_name?: string;
  invite_token?: string;
}
export interface Organization {
  id: string;
  name: string;
  role: string;
}
export interface UserProfile {
  id: string;
  email: string;
  display_name: string | null;
  email_verified: boolean;
  organizations: Organization[];
  created_at: string;
  is_guest?: boolean;
}
export interface AuthResponse {
  access_token?: string;
  refresh_token?: string;
  user: UserProfile;
  token_type: string;
}
export interface ReviewEvent {
  type: string;
  session_id: string;
  sequence: number;
  timestamp?: string;
  [key: string]: unknown;
}
export interface ReviewSummary {
  id: string;
  organization_id: string;
  topic: string | null;
  max_round: number;
  current_round: number;
  current_stage: string;
  status: string;
  document_count: number;
  evidence_count: number;
  creator_id: string;
  creator_name: string | null;
  created_at: string;
  updated_at: string;
}
export interface ReviewDocument {
  id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  parse_status: string;
  parse_error?: string | null;
}
export interface ReviewOutput {
  id: string;
  agent_role: string;
  round_num: number;
  sequence: number;
  content_markdown: string;
  structured_data?: Record<string, unknown> | null;
  created_at: string;
}
export interface ReviewDetail extends ReviewSummary {
  documents: ReviewDocument[];
  outputs: ReviewOutput[];
  report_markdown?: string | null;
  error_message?: string | null;
}
export interface ReviewProgress {
  session_id: string;
  status: string;
  current_stage: string;
  current_round: number;
  max_round: number;
  output_count: number;
  evidence_count: number;
  report_ready: boolean;
  error_message?: string | null;
}
export interface EvidenceItem {
  id: string;
  round_num: number;
  argument_role: string;
  claim_text: string;
  verdict: "verified" | "contradicted" | "uncertain";
  rationale: string;
  sources: {
    id: string;
    title: string;
    url: string;
    snippet?: string | null;
  }[];
}
