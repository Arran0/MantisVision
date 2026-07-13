// Server-only helper around the GitHub Contents/Git Data REST API, used to
// stage admin-uploaded dataset photos on a dedicated branch (dataset-staging)
// instead of a Supabase Storage bucket. The retrain job (ml/scripts/
// retrain_and_report.py, triggered from apps/web/src/app/api/admin/retrain)
// checks that branch out, materializes the staged photos into the training
// dataset, archives them to Kaggle, then resets the branch back to main's
// tip — so this is a transient landing zone, not the durable store.
//
// Reuses the same GITHUB_TOKEN/GITHUB_REPO already wired up for dispatching
// the retrain workflow (apps/web/src/app/api/admin/retrain/route.ts).

const API_BASE = "https://api.github.com";
export const STAGING_BRANCH = "dataset-staging";

function repoSlug(): string {
  const repo = process.env.GITHUB_REPO;
  if (!repo) throw new Error("GITHUB_REPO must be set.");
  return repo;
}

function authHeaders(): HeadersInit {
  const token = process.env.GITHUB_TOKEN;
  if (!token) throw new Error("GITHUB_TOKEN must be set.");
  return { Authorization: `Bearer ${token}`, Accept: "application/vnd.github+json" };
}

// Encode each path segment separately so slashes stay literal in the URL.
function encodePath(path: string): string {
  return path.split("/").map(encodeURIComponent).join("/");
}

async function gh(path: string, init?: RequestInit): Promise<Response> {
  return fetch(`${API_BASE}/repos/${repoSlug()}${path}`, {
    ...init,
    headers: { ...authHeaders(), "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
}

// Creates the staging branch (pointed at main's current tip) the first time
// it's needed. A 422 on creation means another concurrent upload just won,
// which is fine — the branch exists either way.
export async function ensureStagingBranch(baseBranch = "main"): Promise<void> {
  const existing = await gh(`/git/ref/heads/${STAGING_BRANCH}`);
  if (existing.ok) return;

  const baseRef = await gh(`/git/ref/heads/${baseBranch}`);
  if (!baseRef.ok) {
    throw new Error(`Could not read base branch ${baseBranch}: ${await baseRef.text()}`);
  }
  const { object } = await baseRef.json();

  const create = await gh(`/git/refs`, {
    method: "POST",
    body: JSON.stringify({ ref: `refs/heads/${STAGING_BRANCH}`, sha: object.sha }),
  });
  if (!create.ok && create.status !== 422) {
    throw new Error(`Could not create staging branch: ${await create.text()}`);
  }
}

export async function getFile(
  path: string,
  branch: string = STAGING_BRANCH
): Promise<{ content: string; sha: string } | null> {
  const response = await gh(`/contents/${encodePath(path)}?ref=${branch}`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`GitHub getFile(${path}) failed: ${await response.text()}`);
  const data = await response.json();
  return { content: Buffer.from(data.content, "base64").toString("utf-8"), sha: data.sha };
}

export async function getFileRaw(
  path: string,
  branch: string = STAGING_BRANCH
): Promise<Buffer | null> {
  const response = await gh(`/contents/${encodePath(path)}?ref=${branch}`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`GitHub getFileRaw(${path}) failed: ${await response.text()}`);
  const data = await response.json();
  return Buffer.from(data.content, "base64");
}

export async function putFile(
  path: string,
  content: Buffer | string,
  message: string,
  branch: string = STAGING_BRANCH,
  sha?: string
): Promise<void> {
  const body: Record<string, unknown> = {
    message,
    content: Buffer.isBuffer(content) ? content.toString("base64") : Buffer.from(content, "utf-8").toString("base64"),
    branch,
  };
  if (sha) body.sha = sha;

  const response = await gh(`/contents/${encodePath(path)}`, { method: "PUT", body: JSON.stringify(body) });
  if (!response.ok) throw new Error(`GitHub putFile(${path}) failed: ${await response.text()}`);
}

export async function listDir(
  path: string,
  branch: string = STAGING_BRANCH
): Promise<{ name: string; path: string; type: string }[]> {
  const response = await gh(`/contents/${encodePath(path)}?ref=${branch}`);
  if (response.status === 404) return [];
  if (!response.ok) throw new Error(`GitHub listDir(${path}) failed: ${await response.text()}`);
  const data = await response.json();
  return Array.isArray(data) ? data : [];
}
