export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error;
    throw new Error('The local engine is unreachable. Start the API on port 8765, then retry.');
  }
  const text = await response.text();
  let body: unknown;
  try { body = text ? JSON.parse(text) : null; } catch { body = null; }
  if (!response.ok) {
    const info = body as { detail?: string | { message?: string }; error?: string | { message?: string } } | null;
    const detail = info?.detail ?? info?.error;
    const message = typeof detail === 'string' ? detail : detail?.message;
    throw new Error(message || `The local engine returned ${response.status}. Check that the API is running, then retry.`);
  }
  if (body === null) throw new Error('The local engine returned an empty or unreadable response. Retry after checking the API.');
  return body as T;
}

export const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong. Please retry.';
